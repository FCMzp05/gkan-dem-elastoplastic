# ============================================================================
# Fang et al. (2020) §4.2: L-shaped plate (Winkler test), elasto-plastic phase-field
#   fracture with a length-scale insensitive model (Comput Mech 66:931-961)
#
# Model:
#   - Perfect plasticity (no hardening) in effective stress space with a Rankine
#     (maximum principal stress) yield surface; tension-controlled quasi-brittle failure
#   - Effective yield stress σy0 calibrated per lc by quartic Eq. (31) so that nominal peak = σt
#   - AT2 phase field, hybrid formulation: H+ = max(tensile elastic energy + σy0·λ) (Eq. A16)
#   - Nominal stress without split: σ = g(d)·σ0, g = (1-d)² + k
#
# Geometry (Winkler 2001): 500×500 L-shaped plate (250×250 lower-right cut-out),
#   thickness 100 mm, plane stress.
#   Vertices: (0,0)-(250,0)-(250,250)-(500,250)-(500,500)-(0,500)
#   Bottom y=0 clamped; upward displacement on a 10 mm strip at y=250, x=470.
# Material: E0=25.85 GPa, ν=0.18, gf=0.095 N/mm, σt=2.7 MPa
# Units: mm, N, MPa (force reported in kN).
#
# Usage: julia lplate.jl   (scans LC_SCAN; finished cases are skipped)
# ============================================================================

using Gmsh: gmsh
using Gridap
using Gridap.Geometry
using Gridap.Geometry: get_cell_coordinates
using Gridap.TensorValues
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Algebra
using GridapGmsh
using LinearAlgebra
using Printf
using DelimitedFiles
ENV["GKSwstype"] = "100"
using Plots
gr(show=false)

# ---------------- Material (paper §4.2) ----------------
const E0   = 25_850.0    # MPa
const ν0   = 0.18
const gf   = 0.095       # N/mm
const sigt = 2.7         # MPa
const THICK = 100.0      # mm
const η_stab = 1.0e-6
const H_kin  = 5.0e-5 * E0   # tiny regularizing hardening for perfect plasticity

# Plane-stress elasticity
const C11 = E0 / (1 - ν0^2)
const C12 = ν0 * E0 / (1 - ν0^2)
const μ0  = E0 / (2 * (1 + ν0))
const λps = ν0 * E0 / (1 - ν0^2)   # plane-stress λ* (λ*+2μ = E/(1-ν²))

const C_tensor = SymFourthOrderTensorValue(
    C11, 0.0, C12,
    0.0, μ0,  0.0,
    C12, 0.0, C11)

# ---------------- Loading / solver control ----------------
const LC_SCAN = [10.0, 5.0, 7.5, 15.0]
const U_MAX   = 1.0          # mm
const N_STEPS = 250
const Δu      = U_MAX / N_STEPS
const MAX_STAGGER_OUTER = 50
const STAGGER_QUIET_TOL = 1.0e-3
const STAGGER_ACTIVE_TOL = 1.0e-3
const MAX_PLASTIC_FP = 200
const PLASTIC_FP_TOL = 2.0e-4
const PLASTIC_FP_RELAX = parse(Float64, get(ENV, "FANG2020_PLASTIC_FP_RELAX", "0.35"))
0.0 < PLASTIC_FP_RELAX <= 1.0 ||
    error("FANG2020_PLASTIC_FP_RELAX must be in (0,1], got $PLASTIC_FP_RELAX")
const EARLY_STOP_FRAC = 0.10   # stop once force falls below 10% of peak
const OUTPUT_DIR = joinpath(get(ENV, "DEM_REVISION_OUTPUT_ROOT", "outputs"), "fang2020_lplate")
const SNAPSHOT_DAMAGE_LEVELS = [0.3, 0.5, 0.7, 0.9, 0.99]
const VTK_EVERY = 10           # VTK snapshot every N steps (plus damage-threshold snapshots)
const SAVE_INTERMEDIATE_FIELDS =
    lowercase(get(ENV, "FANG2020_SAVE_INTERMEDIATE_FIELDS", "true")) in ("1", "true", "yes")

lc_star() = 0.0429 * gf * E0 / sigt^2

# ---------------- σy0 calibration (Eq. 31) ----------------
function calibrate_sigy0(lc)
    f(s) = s^4 + (2 * E0 * gf / lc) * s^2 - (E0^2 * gf^2 / (sigt * lc^2)) * s + E0^2 * gf^2 / lc^2
    lo = sigt
    hi = 2.0 * sigt
    s_prev, f_prev = lo, f(lo)
    found = false
    for s in range(lo, hi; length=4001)
        fs = f(s)
        if f_prev * fs <= 0 && s > s_prev
            lo, hi = s_prev, s
            found = true
            break
        end
        s_prev, f_prev = s, fs
    end
    found || error("σy0 root not bracketed for lc=$lc")
    flo = f(lo)
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        if f(mid) * flo <= 0
            hi = mid
        else
            lo = mid
            flo = f(lo)
        end
    end
    return 0.5 * (lo + hi)
end

# ---------------- Plastic state ----------------
mutable struct PlasticStateData
    εp::SymTensorValue{2,Float64,3}
    λ::Float64
end
PlasticStateData() = PlasticStateData(SymTensorValue{2,Float64}(0.0, 0.0, 0.0), 0.0)
copy_ps(ps) = PlasticStateData(ps.εp, ps.λ)

# ---------------- Rankine perfect-plasticity return mapping (plane stress, principal space) ----------------
# Input: total strain and start-of-step plastic state; returns (effective stress σ0, tensile elastic energy ψ+)
function rankine_return!(εt::SymTensorValue{2,Float64,3}, ps::PlasticStateData,
                         sy0::Float64; update::Bool=true)
    εe11 = εt[1, 1] - ps.εp[1, 1]
    εe22 = εt[2, 2] - ps.εp[2, 2]
    εe12 = εt[1, 2] - ps.εp[1, 2]

    σ11 = C11 * εe11 + C12 * εe22
    σ22 = C12 * εe11 + C11 * εe22
    σ12 = 2 * μ0 * εe12

    c = 0.5 * (σ11 + σ22)
    R = sqrt((0.5 * (σ11 - σ22))^2 + σ12^2)
    σ1t = c + R
    σ2t = c - R
    sy = sy0 + H_kin * ps.λ

    if σ1t <= sy
        # Elastic: ψ+ from spectral split of principal elastic strains
        ε1 = (σ1t - ν0 * σ2t) / E0
        ε2 = (σ2t - ν0 * σ1t) / E0
        trp = max(ε1 + ε2, 0.0)
        ψp = 0.5 * λps * trp^2 + μ0 * (max(ε1, 0.0)^2 + max(ε2, 0.0)^2)
        σ0 = SymTensorValue{2,Float64}(σ11, σ12, σ22)
        return σ0, ψp
    end

    # Principal directions
    θ = 0.5 * atan(2 * σ12, σ11 - σ22)
    cθ, sθ = cos(θ), sin(θ)

    # Single-surface return (σ1 = sy)
    Δλ1 = (σ1t - sy) / (C11 + H_kin)
    Δλ2 = 0.0
    σ1n = σ1t - C11 * Δλ1
    σ2n = σ2t - C12 * Δλ1
    if σ2n > sy0 + H_kin * (ps.λ + Δλ1)
        # Corner return (σ1 = σ2 = sy)
        A = C11 + H_kin
        B = C12 + H_kin
        f1 = σ1t - sy
        f2 = σ2t - sy
        det = A^2 - B^2
        Δλ1 = (A * f1 - B * f2) / det
        Δλ2 = (A * f2 - B * f1) / det
        σ1n = σ1t - C11 * Δλ1 - C12 * Δλ2
        σ2n = σ2t - C12 * Δλ1 - C11 * Δλ2
    end

    # Principal axes -> xy
    n1_11, n1_22, n1_12 = cθ^2, sθ^2, cθ * sθ
    n2_11, n2_22, n2_12 = sθ^2, cθ^2, -cθ * sθ
    σ0 = SymTensorValue{2,Float64}(
        σ1n * n1_11 + σ2n * n2_11,
        σ1n * n1_12 + σ2n * n2_12,
        σ1n * n1_22 + σ2n * n2_22)

    if update
        Δεp = SymTensorValue{2,Float64}(
            Δλ1 * n1_11 + Δλ2 * n2_11,
            Δλ1 * n1_12 + Δλ2 * n2_12,
            Δλ1 * n1_22 + Δλ2 * n2_22)
        ps.εp = ps.εp + Δεp
        ps.λ = ps.λ + Δλ1 + Δλ2
    end

    # Elastic principal strains after return, and ψ+
    ε1 = (σ1n - ν0 * σ2n) / E0
    ε2 = (σ2n - ν0 * σ1n) / E0
    trp = max(ε1 + ε2, 0.0)
    ψp = 0.5 * λps * trp^2 + μ0 * (max(ε1, 0.0)^2 + max(ε2, 0.0)^2)
    return σ0, ψp
end

# ---------------- Mesh ----------------
const XLOAD1, XLOAD2 = 465.0, 475.0   # load strip (centered at x=470, 30 mm from right edge)

function create_mesh(case_dir, lc)
    h_f = lc / 5          # h/lc = 0.2 as in the paper
    h_c = 15.0
    println("  Mesh: h_fine=$(h_f) mm (lc/5), h_coarse=$(h_c) mm")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Lplate")
    geo = gmsh.model.geo

    h_load = min(h_c, 4.0)   # local refinement so the 10 mm strip has interior nodes
    p1 = geo.addPoint(0.0, 0.0, 0.0, h_c)
    p2 = geo.addPoint(250.0, 0.0, 0.0, h_c)
    p3 = geo.addPoint(250.0, 250.0, 0.0, h_f)     # re-entrant corner (crack initiation)
    p4 = geo.addPoint(XLOAD1, 250.0, 0.0, h_load)
    p5 = geo.addPoint(XLOAD2, 250.0, 0.0, h_load)
    p6 = geo.addPoint(500.0, 250.0, 0.0, h_c)
    p7 = geo.addPoint(500.0, 500.0, 0.0, h_c)
    p8 = geo.addPoint(0.0, 500.0, 0.0, h_c)

    l_bot   = geo.addLine(p1, p2)   # clamped
    l_r1    = geo.addLine(p2, p3)
    l_arm1  = geo.addLine(p3, p4)
    l_load  = geo.addLine(p4, p5)   # load strip
    l_arm2  = geo.addLine(p5, p6)
    l_right = geo.addLine(p6, p7)
    l_top   = geo.addLine(p7, p8)
    l_left  = geo.addLine(p8, p1)

    loop = geo.addCurveLoop([l_bot, l_r1, l_arm1, l_load, l_arm2, l_right, l_top, l_left])
    surf = geo.addPlaneSurface([loop])
    geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.addPhysicalGroup(1, [l_bot], 1)
    gmsh.model.setPhysicalName(1, 1, "fixed")
    gmsh.model.addPhysicalGroup(1, [l_load], 2)
    gmsh.model.setPhysicalName(1, 2, "load")
    # Include end points so curve end nodes receive the Dirichlet tags
    gmsh.model.addPhysicalGroup(0, [p1, p2], 3)
    gmsh.model.setPhysicalName(0, 3, "fixed_pts")
    gmsh.model.addPhysicalGroup(0, [p4, p5], 4)
    gmsh.model.setPhysicalName(0, 4, "load_pts")

    # Box refinement along the expected crack path from the re-entrant corner
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_f)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_c)
    gmsh.model.mesh.field.setNumber(1, "XMin", 0.0)
    gmsh.model.mesh.field.setNumber(1, "XMax", 310.0)
    gmsh.model.mesh.field.setNumber(1, "YMin", 230.0)
    gmsh.model.mesh.field.setNumber(1, "YMax", 400.0)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 30.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.model.mesh.generate(2)

    num_nodes = length(gmsh.model.mesh.getNodes()[1])
    println("  Nodes: $(num_nodes)")

    mesh_file = joinpath(case_dir, "mesh.msh")
    gmsh.write(mesh_file)
    gmsh.finalize()
    return mesh_file
end

# ---------------- Displacement solve (d and start-of-step plastic state frozen; εp fixed-point) ----------------
function compute_stress!(uh, dh, plastic_states, cell_points; update=false)
    nc = length(plastic_states)
    ε_qp = evaluate(ε(uh), cell_points)
    σ_vals = Vector{SymTensorValue{2,Float64,3}}(undef, nc)
    ψp_vals = zeros(nc)
    for i in 1:nc
        σ_vals[i], ψp_vals[i] = rankine_return!(ε_qp[i][1], plastic_states[i], SIGY0[]; update=update)
    end
    return σ_vals, ψp_vals
end

const SIGY0 = Ref(0.0)   # calibrated σy0 of the current case

function solve_displacement(dh, plastic_states, u_val, V_u, dΩ, Ω, uh_prev, cell_points)
    g_d(d) = (1 - d)^2 + η_stab
    nc = length(plastic_states)
    work = [copy_ps(ps) for ps in plastic_states]
    ps_start = [copy_ps(ps) for ps in work]

    zerov = VectorValue(0.0, 0.0)
    loadv = VectorValue(0.0, u_val)
    U_u = TrialFESpace(V_u, [zerov, zerov, loadv, loadv])

    # Stiffness depends only on dh: assemble and factorize once; fixed-point updates the RHS only
    a(u, v) = ∫((g_d ∘ dh) * ((C_tensor ⊙ ε(u)) ⊙ ε(v))) * dΩ
    l0(v) = ∫(zerov ⋅ v) * dΩ
    op = AffineFEOperator(a, l0, U_u, V_u)
    A = Gridap.Algebra.get_matrix(op)
    b0 = Gridap.Algebra.get_vector(op)
    Fact = lu(A)

    uh = uh_prev
    x = copy(get_free_dof_values(uh))
    fp_iters = 0
    fp_rel = Inf
    fp_converged = false
    for it in 1:MAX_PLASTIC_FP
        σp_vals = [C_tensor ⊙ work[i].εp for i in 1:nc]
        σp_field = CellField(σp_vals, Ω)
        lp(v) = ∫((g_d ∘ dh) * (σp_field ⊙ ε(v))) * dΩ
        bp = assemble_vector(lp, V_u)
        x_raw = Fact \ (b0 .+ bp)
        # Convergence on the unrelaxed Picard defect; ω changes the path, not the fixed point.
        fp_rel = norm(x_raw - x) / (norm(x_raw) + 1e-12)
        fp_iters = it
        if fp_rel < PLASTIC_FP_TOL
            x = x_raw
            uh = FEFunction(U_u, x)
            for i in 1:nc
                work[i].εp = ps_start[i].εp
                work[i].λ = ps_start[i].λ
            end
            compute_stress!(uh, dh, work, cell_points; update=true)
            fp_converged = true
            break
        end
        x = x .+ PLASTIC_FP_RELAX .* (x_raw .- x)
        uh = FEFunction(U_u, x)
        for i in 1:nc
            work[i].εp = ps_start[i].εp
            work[i].λ = ps_start[i].λ
        end
        compute_stress!(uh, dh, work, cell_points; update=true)
    end
    for i in 1:nc
        plastic_states[i].εp = work[i].εp
        plastic_states[i].λ = work[i].λ
    end
    fp_converged || @warn "Plastic fixed-point reached iteration limit" iterations=fp_iters residual=fp_rel relaxation=PLASTIC_FP_RELAX
    return uh, fp_iters, fp_rel, fp_converged
end

# ---------------- Phase-field solve ----------------
function solve_phase_field(dh_committed, H_field, V_d, U_d, dΩ, lc)
    a(d, φ) = ∫((gf / lc + 2.0 * H_field) * d * φ + gf * lc * (∇(d) ⋅ ∇(φ))) * dΩ
    b(φ) = ∫(2.0 * H_field * φ) * dΩ
    op = AffineFEOperator(a, b, U_d, V_d)
    dh_solved = solve(op)
    d_new = get_free_dof_values(dh_solved)
    d_old = get_free_dof_values(dh_committed)
    for i in eachindex(d_new)
        isfinite(d_new[i]) || (d_new[i] = d_old[i])
        d_new[i] = clamp(max(d_new[i], d_old[i]), 0.0, 1.0)
    end
    return FEFunction(U_d, d_new)
end

function project_to_fe(q, V, dΩ)
    a(u, v) = ∫(u * v) * dΩ
    b(v) = ∫(v * q) * dΩ
    return solve(AffineFEOperator(a, b, V, V))
end

# ---------------- VTK field snapshots (for ParaView) ----------------
function save_vtk(case_dir, tag, Ω, uh, dh, plastic_states, H_arr, σ_cell)
    vdir = joinpath(case_dir, "vtk")
    mkpath(vdir)
    nc = length(plastic_states)
    λv = [plastic_states[i].λ for i in 1:nc]
    epxx = [plastic_states[i].εp[1, 1] for i in 1:nc]
    epyy = [plastic_states[i].εp[2, 2] for i in 1:nc]
    epxy = [plastic_states[i].εp[1, 2] for i in 1:nc]
    sxx = [σ_cell[i][1, 1] for i in 1:nc]
    syy = [σ_cell[i][2, 2] for i in 1:nc]
    sxy = [σ_cell[i][1, 2] for i in 1:nc]
    writevtk(Ω, joinpath(vdir, "state_$tag"),
             cellfields=["u" => uh, "d" => dh],
             celldata=["lambda" => λv, "H" => H_arr,
                       "ep_xx" => epxx, "ep_yy" => epyy, "ep_xy" => epxy,
                       "sig_xx" => sxx, "sig_yy" => syy, "sig_xy" => sxy])
end

# ---------------- Damage plot (cell-value scatter) ----------------
function plot_damage_png(dh, Ω, cell_points, case_dir, tag, lc)
    d_qp = evaluate(dh, cell_points)
    nc = length(d_qp)
    coords = get_cell_coordinates(Ω)
    xs = zeros(nc)
    ys = zeros(nc)
    dv = zeros(nc)
    for i in 1:nc
        pts = coords[i]
        xs[i] = sum(p[1] for p in pts) / length(pts)
        ys[i] = sum(p[2] for p in pts) / length(pts)
        dv[i] = clamp(d_qp[i][1], 0.0, 1.0)
    end
    p = scatter(xs, ys, marker_z=dv, ms=1.6, msw=0, c=:jet, clims=(0, 1),
                aspect_ratio=:equal, legend=false, colorbar=true,
                title="damage ($tag, lc=$(lc)mm)", xlim=(0, 500), ylim=(0, 500))
    savefig(p, joinpath(case_dir, "damage_$(tag).png"))
end

# ---------------- Single case ----------------
function run_case(lc, case_dir)
    mkpath(case_dir)
    sy0 = calibrate_sigy0(lc)
    SIGY0[] = sy0

    println("\n" * "=" ^ 76)
    @printf("Fang2020 L-shaped plate | lc = %g mm (lc* = %.2f mm)\n", lc, lc_star())
    @printf("  E0=%.0f MPa, ν=%.2f, gf=%.3f N/mm, σt=%.2f MPa\n", E0, ν0, gf, sigt)
    @printf("  σy0 = %.4f MPa (%.3f σt), Rankine perfect plasticity + AT2 phase field (hybrid)\n", sy0, sy0 / sigt)
    @printf("  plane stress, thickness %.0f mm; U_max=%.2f mm, %d steps; plastic FP ω=%.3f, tol=%.1e\n",
            THICK, U_MAX, N_STEPS, PLASTIC_FP_RELAX, PLASTIC_FP_TOL)
    println("=" ^ 76)

    t0 = time()
    mesh_file = create_mesh(case_dir, lc)
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    Γl = BoundaryTriangulation(model, tags="load")
    dΓl = Measure(Γl, 2)
    nΓl = get_normal_vector(Γl)

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    V_u = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags=["fixed", "fixed_pts", "load", "load_pts"],
                      dirichlet_masks=[(true, true), (true, true), (false, true), (false, true)])
    # Reaction via virtual work with a harmonic lift (same definition as the DEM φ):
    # vector Laplace with load strip = (0,1), fixed edge = (0,0), natural BCs elsewhere.
    V_lift = TestFESpace(model, reffe_u, conformity=:H1,
                         dirichlet_tags=["fixed", "fixed_pts", "load", "load_pts"])
    U_lift = TrialFESpace(V_lift, [VectorValue(0.0, 0.0), VectorValue(0.0, 0.0),
                                   VectorValue(0.0, 1.0), VectorValue(0.0, 1.0)])
    a_lift(w, q) = ∫(∇(w) ⊙ ∇(q)) * dΩ
    l_lift(q) = 0.0
    v_lift = solve(AffineFEOperator(a_lift, l_lift, U_lift, V_lift))

    reffe_d = ReferenceFE(lagrangian, Float64, 1)
    V_d = TestFESpace(model, reffe_d, conformity=:H1)
    U_d = TrialFESpace(V_d)
    reffe_dg0 = ReferenceFE(lagrangian, Float64, 0)
    V_dg0 = FESpace(model, reffe_dg0, conformity=:L2)
    V_L2 = FESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)

    quad = CellQuadrature(Ω, 2)
    cell_points = get_cell_points(quad)
    nc = num_cells(Ω)
    println("  Cells: $nc")

    plastic_states = [PlasticStateData() for _ in 1:nc]
    H_arr = zeros(nc)
    dh = interpolate_everywhere(x -> 0.0, U_d)
    zv = VectorValue(0.0, 0.0)
    uh = zero(TrialFESpace(V_u, [zv, zv, zv, zv]))

    hist = Vector{Vector{Float64}}()
    peak_kN = -Inf
    snapshot_done = falses(length(SNAPSHOT_DAMAGE_LEVELS))
    case_converged = true

    for step in 1:N_STEPS
        u_val = step * Δu

        local σ_cell
        dh_committed = dh
        d_committed = copy(get_free_dof_values(dh_committed))
        ps_committed = [copy_ps(ps) for ps in plastic_states]
        H_committed = copy(H_arr)
        dh_trial = dh_committed
        uh_trial = uh
        step_converged = false
        outer_used = 0
        stagger_res = Inf
        fp_iters = 0
        fp_rel = Inf
        fp_converged = false

        for outer in 1:MAX_STAGGER_OUTER
            outer_used = outer
            plastic_trial = [copy_ps(ps) for ps in ps_committed]
            uh_trial, fp_iters, fp_rel, fp_converged =
                solve_displacement(dh_trial, plastic_trial, u_val, V_u, dΩ, Ω, uh_trial, cell_points)
            if !fp_converged
                stagger_res = fp_rel
                break
            end

            _, ψp_cell = compute_stress!(uh_trial, dh_trial, plastic_trial, cell_points; update=false)
            H_trial = similar(H_committed)
            for i in 1:nc
                λi = plastic_trial[i].λ
                driver = ψp_cell[i] + sy0 * λi + 0.5 * H_kin * λi^2
                H_trial[i] = max(H_committed[i], driver)
            end
            H_field = project_to_fe(FEFunction(V_dg0, H_trial), V_L2, dΩ)
            dh_candidate = solve_phase_field(dh_committed, H_field, V_d, U_d, dΩ, lc)
            d_candidate = get_free_dof_values(dh_candidate)
            stagger_res = norm(d_candidate - get_free_dof_values(dh_trial), Inf)
            damage_active = norm(d_candidate - d_committed, Inf) > STAGGER_QUIET_TOL
            stagger_tol = damage_active ? STAGGER_ACTIVE_TOL : STAGGER_QUIET_TOL

            if stagger_res <= stagger_tol
                plastic_corrected = [copy_ps(ps) for ps in ps_committed]
                uh_corrected, fp_iters, fp_rel, fp_converged =
                    solve_displacement(dh_candidate, plastic_corrected, u_val, V_u, dΩ, Ω, uh_trial, cell_points)
                if !fp_converged
                    stagger_res = fp_rel
                    break
                end
                σ_corrected, ψp_corrected =
                    compute_stress!(uh_corrected, dh_candidate, plastic_corrected, cell_points; update=false)
                H_corrected = similar(H_committed)
                for i in 1:nc
                    λi = plastic_corrected[i].λ
                    driver = ψp_corrected[i] + sy0 * λi + 0.5 * H_kin * λi^2
                    H_corrected[i] = max(H_committed[i], driver)
                end
                H_corrected_field = project_to_fe(FEFunction(V_dg0, H_corrected), V_L2, dΩ)
                dh_check = solve_phase_field(dh_committed, H_corrected_field, V_d, U_d, dΩ, lc)
                d_correction_res = norm(get_free_dof_values(dh_check) - d_candidate, Inf)
                H_correction_res = norm(H_corrected - H_trial, Inf) /
                                   max(norm(H_corrected, Inf), gf / lc, eps(Float64))
                stagger_res = max(d_correction_res, H_correction_res)
                if stagger_res <= stagger_tol
                    dh = dh_candidate
                    uh = uh_corrected
                    plastic_states = plastic_corrected
                    H_arr = H_corrected
                    σ_cell = σ_corrected
                    step_converged = true
                    break
                end
                dh_trial = dh_check
                uh_trial = uh_corrected
            else
                dh_trial = dh_candidate
            end
        end

        if !step_converged
            case_converged = false
            d_max = maximum(d_committed)
            λ_max = maximum(ps.λ for ps in ps_committed)
            d_int = sum(∫(dh_committed) * dΩ)
            wp_int = sum(sy0 * ps_committed[i].λ + 0.5 * H_kin * ps_committed[i].λ^2 for i in 1:nc)
            push!(hist, [u_val, NaN, d_max, λ_max, d_int, wp_int,
                         outer_used, stagger_res, 0.0, fp_iters, fp_rel, fp_converged ? 1.0 : 0.0])
            @error "Load step not converged; keeping committed state and stopping case" step=step outer=outer_used stagger_res=stagger_res fp_res=fp_rel
            break
        end

        # Reaction by virtual work (more accurate than a boundary flux integral of cell-constant stress)
        σ_field = CellField(σ_cell, Ω)
        g_d(d) = (1 - d)^2 + η_stab
        reaction_vw = sum(∫((g_d ∘ dh) * (σ_field ⊙ ε(v_lift))) * dΩ)
        F_kN = reaction_vw * THICK / 1000.0

        d_max = maximum(get_free_dof_values(dh))
        λ_max = maximum(ps.λ for ps in plastic_states)
        d_int = sum(∫(dh) * dΩ)                       # ∫d dΩ (damage volume)
        wp_int = sum(sy0 * plastic_states[i].λ + 0.5 * H_kin * plastic_states[i].λ^2 for i in 1:nc)
        push!(hist, [u_val, F_kN, d_max, λ_max, d_int, wp_int,
                     outer_used, stagger_res, 1.0, fp_iters, fp_rel, fp_converged ? 1.0 : 0.0])
        peak_kN = max(peak_kN, F_kN)

        if step % 10 == 0 || step <= 3
            @printf("  step %3d/%d | u=%.4f mm | F=%.4f kN | d_max=%.4f | stag=%d/%.2e | fp=%d/%.2e | t=%.0fs\n",
                    step, N_STEPS, u_val, F_kN, d_max, outer_used, stagger_res,
                    fp_iters, fp_rel, time() - t0)
        end

        if SAVE_INTERMEDIATE_FIELDS && step % VTK_EVERY == 0
            save_vtk(case_dir, @sprintf("step%04d", step), Ω, uh, dh, plastic_states, H_arr, σ_cell)
        end
        if SAVE_INTERMEDIATE_FIELDS
            for (i, thr) in enumerate(SNAPSHOT_DAMAGE_LEVELS)
                if !snapshot_done[i] && d_max >= thr
                    tag = @sprintf("d%02d", round(Int, 100 * thr))
                    plot_damage_png(dh, Ω, cell_points, case_dir, tag, lc)
                    save_vtk(case_dir, tag, Ω, uh, dh, plastic_states, H_arr, σ_cell)
                    snapshot_done[i] = true
                end
            end
        end

        if d_max > 0.95 && F_kN < EARLY_STOP_FRAC * peak_kN
            println("  Early stop: F=$(round(F_kN, digits=3)) kN < $(EARLY_STOP_FRAC)×peak, d_max=$(round(d_max, digits=3))")
            break
        end
    end

    # Output
    open(joinpath(case_dir, "force_displacement.csv"), "w") do io
        println(io, "u_mm,F_kN,d_max,lambda_max,int_d_dOmega,plastic_work_ind,stagger_iters,stagger_res,step_converged,plastic_fp_iters,plastic_fp_res,plastic_fp_converged")
        for row in hist
            @printf(io, "%.6f,%.6f,%.6f,%.6e,%.6e,%.6e,%.0f,%.6e,%.0f,%.0f,%.6e,%.0f\n", row...)
        end
    end
    plot_damage_png(dh, Ω, cell_points, case_dir, "final", lc)
    let (σ_cell_f, _) = compute_stress!(uh, dh, plastic_states, cell_points; update=false)
        save_vtk(case_dir, "final", Ω, uh, dh, plastic_states, H_arr, σ_cell_f)
    end
    case_converged || error("L-plate lc=$lc stopped at an unconverged load step; DONE not written")

    us = [r[1] for r in hist]
    fs = [r[2] for r in hist]
    p = plot(us, fs, lw=2, xlabel="u (mm)", ylabel="F (kN)",
             title=@sprintf("L-plate lc=%g mm (peak %.2f kN)", lc, maximum(fs)), legend=false)
    savefig(p, joinpath(case_dir, "F_u_curve.png"))

    open(joinpath(case_dir, "DONE"), "w") do io
        @printf(io, "lc=%g peak_kN=%.4f steps=%d\n", lc, maximum(fs), length(hist))
    end
    @printf("\nFinished lc=%g: peak = %.3f kN, runtime = %.0f s\n", lc, maximum(fs), time() - t0)
end

# ---------------- Scan driver ----------------
function main()
    mkpath(OUTPUT_DIR)
    for lc in LC_SCAN
        cdir = joinpath(OUTPUT_DIR, "lc$(lc)")
        if isfile(joinpath(cdir, "DONE"))
            println("[skip] lc=$lc already done")
            continue
        end
        run_case(lc, cdir)
    end
    # Combined plot
    p = plot(xlabel="u (mm)", ylabel="F (kN)", title="Fang2020 L-plate: lc scan", legend=:topright)
    for lc in LC_SCAN
        csv = joinpath(OUTPUT_DIR, "lc$(lc)", "force_displacement.csv")
        isfile(csv) || continue
        data = readdlm(csv, ',', skipstart=1)
        plot!(p, data[:, 1], data[:, 2], lw=2, label="lc = $(lc) mm")
    end
    savefig(p, joinpath(OUTPUT_DIR, "F_u_scan.png"))
    println("Combined plot -> $OUTPUT_DIR/F_u_scan.png")
end

main()

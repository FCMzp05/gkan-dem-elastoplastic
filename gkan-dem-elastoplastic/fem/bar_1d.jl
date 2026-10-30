# ============================================================================
# Fang et al. (2020) "Phase field fracture in elasto-plastic solids: a
# length-scale insensitive model for quasi-brittle materials", Comput Mech.
#
# 1D bar in tension (L = 100 mm), §3.2/§3.4: coupled elasto-plastic phase-field model.
#   - Perfect plasticity (no hardening) solved in effective stress space (Eqs. 6-7)
#   - Effective yield stress σy0 calibrated by quartic Eq. (31) so that nominal peak = σt
#   - AT2 phase field: (gf/lc + 2H)d - gf*lc*d'' = 2H (hybrid formulation, Eqs. A17/A18)
#   - History field H = max(½E0εe² + σy0·λ) (Eq. A15; no split needed in 1D tension)
#
# Units: mm, N, MPa. Parameters are constants below; lc is scanned in main().
# Plain 1D linear FE (no Gridap), staggered scheme; the central element's σy0 is
# slightly reduced to trigger localization.
# Usage: julia bar_1d.jl
# ============================================================================

using LinearAlgebra
using Printf
using DelimitedFiles
ENV["GKSwstype"] = "100"
using Plots
gr(show=false)

# ---------------- Material (concrete, paper §3) ----------------
const E0   = 31_000.0     # MPa
const gf   = 0.059        # N/mm
const sigt = 2.86         # MPa, tensile strength
const L    = 100.0        # mm
const k_res = 1.0e-6      # residual stiffness
const H_kin = 1.0e-4 * E0 # tiny hardening, numerical regularization of the perfectly plastic tangent

const OUTDIR = "outputs/fang2020_bar"

# lc* = 0.0429 gf E0 / σt² (Eq. 30)
lc_star() = 0.0429 * gf * E0 / sigt^2

# ---------------- σy0 calibration (quartic Eq. 31, bisection) ----------------
function calibrate_sigy0(lc)
    f(s) = s^4 + (2 * E0 * gf / lc) * s^2 - (E0^2 * gf^2 / (sigt * lc^2)) * s + E0^2 * gf^2 / lc^2
    lo, hi = sigt, 1.5 * sigt
    flo = f(lo)
    fhi = f(hi)
    # Root lies in [σt, 1.5σt] (σy0 ∈ [σt, 1.11σt] for lc ≤ lc*)
    if flo * fhi > 0
        # For small lc the root approaches σt; scan for a sign change
        s_prev, f_prev = lo, flo
        found = false
        for s in range(lo, 2.0 * sigt; length=2001)
            fs = f(s)
            if f_prev * fs <= 0
                lo, hi, flo = s_prev, s, f_prev
                found = true
                break
            end
            s_prev, f_prev = s, fs
        end
        found || error("σy0 root not bracketed for lc=$lc")
    end
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

# ---------------- 1D FE mesh (uniform, h = 0.05 lc, paper Fig 10) ----------------
struct BarMesh
    x::Vector{Float64}     # node coordinates
    h::Vector{Float64}     # element lengths
    nel::Int
end

function make_mesh(lc)
    h_target = 0.05 * lc
    nel = ceil(Int, L / h_target)
    isodd(nel) && (nel += 1)   # even element count: x=0 is a node; one adjacent element is weakened
    x = collect(range(-L / 2, L / 2; length=nel + 1))
    h = diff(x)
    return BarMesh(x, h, nel)
end

# ---------------- Element state from nominal stress s (perfect plasticity in effective stress space) ----------------
# Returns (effective stress σ0, new εp, new λ, elastic strain εe)
function state_from_stress(s, g, epsp0, lam0, sy0)
    sig0 = s / g
    sy = sy0 + H_kin * lam0
    if sig0 > sy
        dlam = (sig0 - sy) / H_kin
        return sig0, epsp0 + dlam, lam0 + dlam, sig0 / E0
    elseif sig0 < -sy
        dlam = (-sig0 - sy) / H_kin
        return sig0, epsp0 - dlam, lam0 + dlam, sig0 / E0
    else
        return sig0, epsp0, lam0, sig0 / E0
    end
end

# ---------------- Displacement subproblem: scalar stress root (1D equilibrium = uniform nominal stress) ----------------
# For a given nominal stress s, element strains follow from the constitutive law (d and start-of-step plastic state frozen):
#   σ0 = s/g; |σ0| ≤ sy -> elastic ε = εp0 + σ0/E; otherwise plastic Δλ = (|σ0|-sy)/H_kin.
# ε(s) is monotone in s, so the total elongation U(s) is monotone and is solved by bisection.
function strain_from_stress(s, g, epsp0, lam0, sy0)
    sig0 = s / g
    sy = sy0 + H_kin * lam0
    if sig0 > sy
        dlam = (sig0 - sy) / H_kin
        return epsp0 + dlam + sig0 / E0
    elseif sig0 < -sy
        dlam = (-sig0 - sy) / H_kin
        return epsp0 - dlam + sig0 / E0
    else
        return epsp0 + sig0 / E0
    end
end

function total_elongation(s, d_elem, epsp0, lam0, sy0_elem, mesh)
    total = 0.0
    for e in 1:mesh.nel
        g = (1.0 - d_elem[e])^2 + k_res
        total += mesh.h[e] * strain_from_stress(s, g, epsp0[e], lam0[e], sy0_elem[e])
    end
    return total
end

function solve_stress(d_elem, epsp0, lam0, sy0_elem, mesh, U; s_bracket=10.0)
    lo, hi = -s_bracket, s_bracket
    flo = total_elongation(lo, d_elem, epsp0, lam0, sy0_elem, mesh) - U
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        fm = total_elongation(mid, d_elem, epsp0, lam0, sy0_elem, mesh) - U
        if flo * fm <= 0
            hi = mid
        else
            lo = mid
            flo = fm
        end
        (hi - lo) < 1.0e-14 * s_bracket && break
    end
    return 0.5 * (lo + hi)
end

# ---------------- Phase-field solve (linear, P1 tridiagonal) ----------------
function solve_damage(H_elem, lc, mesh)
    n = length(mesh.x)
    K = zeros(n, n)
    rhs = zeros(n)
    for e in 1:mesh.nel
        he = mesh.h[e]
        He = H_elem[e]
        coef = gf / lc + 2.0 * He
        # Consistent mass he/6·[2 1;1 2], gradient gf·lc/he·[1 -1;-1 1]
        m = he / 6.0 .* [2.0 1.0; 1.0 2.0]
        kg = gf * lc / he .* [1.0 -1.0; -1.0 1.0]
        ke = coef .* m .+ kg
        fe = 2.0 * He .* (m * [1.0, 1.0])
        idx = [e, e + 1]
        K[idx, idx] .+= ke
        rhs[idx] .+= fe
    end
    return K \ rhs
end

# ---------------- Single case ----------------
function run_case(lc; U_max=0.15, n_steps=1500)
    sy0 = calibrate_sigy0(lc)
    mesh = make_mesh(lc)
    n = length(mesh.x)
    @printf("  lc=%.4f mm (lc*/%.1f)  σy0=%.4f MPa (%.3fσt)  nel=%d h=%.4f\n",
            lc, lc_star() / lc, sy0, sy0 / sigt, mesh.nel, mesh.h[1])

    # Reduce σy0 of the central element by 0.2% to trigger localization at x=0
    sy0_elem = fill(sy0, mesh.nel)
    icen = mesh.nel ÷ 2
    sy0_elem[icen] *= 0.998

    u = zeros(n)
    d_nodes = zeros(n)
    epsp = zeros(mesh.nel)
    lam = zeros(mesh.nel)
    H_elem = zeros(mesh.nel)

    Us = Float64[]
    sigs = Float64[]
    dmaxs = Float64[]
    lammaxs = Float64[]
    # Profile snapshots triggered by damage thresholds
    snap_levels = [0.3, 0.5, 0.7, 0.9, 0.99]
    snap_done = falses(length(snap_levels))
    prof_dir = joinpath(OUTDIR, @sprintf("profiles_lc%.3f", lc))
    mkpath(prof_dir)
    save_profile(tag) = writedlm(joinpath(prof_dir, "profile_$tag.csv"),
        [["x_mm" "d" "epsp" "lambda"];
         [mesh.x d_nodes [epsp; NaN] [lam; NaN]]], ',')
    s = 0.0
    for step in 1:n_steps
        U = U_max * step / n_steps
        # Staggered: stress root (exact 1D equilibrium) -> H update -> d solve, until d converges
        local d_elem
        H_trial = copy(H_elem)
        for it in 1:200
            d_elem = 0.5 .* (d_nodes[1:end - 1] .+ d_nodes[2:end])
            s = solve_stress(d_elem, epsp, lam, sy0_elem, mesh, U)
            # H update: elastic energy + plastic work σy0λ (always tensile in 1D, no split)
            for e in 1:mesh.nel
                g = (1.0 - d_elem[e])^2 + k_res
                _, _, lam_new, ee = state_from_stress(s, g, epsp[e], lam[e], sy0_elem[e])
                psi = 0.5 * E0 * ee^2 + sy0_elem[e] * lam_new
                H_trial[e] = max(H_trial[e], psi)
            end
            d_new = solve_damage(H_trial, lc, mesh)
            d_new = clamp.(max.(d_new, d_nodes), 0.0, 1.0)  # irreversibility
            delta = maximum(abs.(d_new .- d_nodes))
            d_nodes = d_new
            delta < 1.0e-6 && break
        end
        # Commit plastic state and history field
        d_elem = 0.5 .* (d_nodes[1:end - 1] .+ d_nodes[2:end])
        for e in 1:mesh.nel
            g = (1.0 - d_elem[e])^2 + k_res
            _, epsp[e], lam[e], _ = state_from_stress(s, g, epsp[e], lam[e], sy0_elem[e])
        end
        H_elem = H_trial
        push!(Us, U)
        push!(sigs, s)
        push!(dmaxs, maximum(d_nodes))
        push!(lammaxs, maximum(lam))
        for (i, thr) in enumerate(snap_levels)
            if !snap_done[i] && maximum(d_nodes) >= thr
                save_profile(@sprintf("d%02d", round(Int, 100 * thr)))
                snap_done[i] = true
            end
        end
        if step % 250 == 0
            @printf("    step %3d/%d U=%.4f σ=%.4f MPa d_max=%.4f λ_max=%.3e\n",
                    step, n_steps, U, s, maximum(d_nodes), maximum(lam))
        end
    end
    save_profile("final")
    return Us, sigs, dmaxs, lammaxs, sy0, d_nodes, mesh
end

# ---------------- Main: lc scan ----------------
function main()
    mkpath(OUTDIR)
    lcs = lc_star() .* [0.2, 0.5, 1.0]
    println("=" ^ 72)
    @printf("Fang 2020 1D bar: E0=%.0f MPa, gf=%.3f N/mm, σt=%.2f MPa, L=%.0f mm\n",
            E0, gf, sigt, L)
    @printf("lc* = %.4f mm; scanning lc = lc*/5, lc*/2, lc*\n", lc_star())
    println("=" ^ 72)

    results = []
    for lc in lcs
        Us, sigs, dmaxs, lammaxs, sy0, d_prof, mesh = run_case(lc)
        peak = maximum(sigs)
        @printf("  --> peak stress = %.4f MPa (σt=%.2f, error %.2f%%)\n\n",
                peak, sigt, 100 * abs(peak - sigt) / sigt)
        writedlm(joinpath(OUTDIR, @sprintf("bar_lc%.3f.csv", lc)),
                 [["U_mm" "sigma_MPa" "d_max" "lambda_max"]; [Us sigs dmaxs lammaxs]], ',')
        push!(results, (lc, Us, sigs, d_prof, mesh))
    end

    # σ-U curves (paper Fig 8/11a)
    p1 = plot(xlabel="U (mm)", ylabel="stress (MPa)",
              title="Fang2020 1D bar: stress-displacement", legend=:topright)
    for (lc, Us, sigs, _, _) in results
        plot!(p1, Us, sigs, lw=2, label=@sprintf("lc = %.2f mm", lc))
    end
    hline!(p1, [sigt], ls=:dash, c=:gray, label="σt = 2.86 MPa")
    savefig(p1, joinpath(OUTDIR, "stress_displacement.png"))

    # Damage profiles (paper Fig 11b)
    p2 = plot(xlabel="x (mm)", ylabel="d", title="Final damage profiles", legend=:topright)
    for (lc, _, _, d_prof, mesh) in results
        plot!(p2, mesh.x, d_prof, lw=2, label=@sprintf("lc = %.2f mm", lc))
    end
    savefig(p2, joinpath(OUTDIR, "damage_profiles.png"))
    println("plots -> $OUTDIR/stress_displacement.png, damage_profiles.png")
end

main()

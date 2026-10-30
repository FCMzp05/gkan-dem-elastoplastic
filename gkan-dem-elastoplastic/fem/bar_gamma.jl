# ============================================================================
# Fang et al. (2020) §3.5: Γ-convergence study of a reduced-section bar (Fig 13-15)
#
# Geometry: L = 100 mm bar; the central 0.15L has its section reduced from A = 4 mm²
#   to Ar = 2 mm² (step change); displacement loading U. The reduced section triggers localization.
# Model as in bar_1d.jl (1D perfect plasticity + AT2 + σy0 calibration); equilibrium
#   is uniform force F: σ_e = F / A_e, solved for F by bisection.
# Crack surface Γ_lc,h = ∫ (d²/2lc + lc/2·d'²) A dx -> Ar as lc -> 0 (Fig 14).
#
# Units: mm, N, MPa. Usage: julia bar_gamma.jl
# ============================================================================

using LinearAlgebra
using Printf
using DelimitedFiles
ENV["GKSwstype"] = "100"
using Plots
gr(show=false)

const E0   = 31_000.0     # MPa
const gf   = 0.059        # N/mm
const sigt = 2.86         # MPa
const L    = 100.0        # mm
const A_out = 4.0         # mm²
const A_red = 2.0         # mm²
const RED_HALF = 0.075 * L   # half-length of reduced section (0.15L total)
const k_res = 1.0e-6
const H_kin = 1.0e-4 * E0

const OUTDIR = "outputs/fang2020_bar_gamma"

lc_star() = 0.0429 * gf * E0 / sigt^2

function calibrate_sigy0(lc)
    f(s) = s^4 + (2 * E0 * gf / lc) * s^2 - (E0^2 * gf^2 / (sigt * lc^2)) * s + E0^2 * gf^2 / lc^2
    lo, hi = sigt, 2.0 * sigt
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

struct BarMesh
    x::Vector{Float64}
    h::Vector{Float64}
    A::Vector{Float64}   # element cross-section areas
    nel::Int
end

function make_mesh(lc)
    h_target = 0.05 * lc
    nel = ceil(Int, L / h_target)
    isodd(nel) && (nel += 1)
    x = collect(range(-L / 2, L / 2; length=nel + 1))
    h = diff(x)
    xc = 0.5 .* (x[1:end - 1] .+ x[2:end])
    A = [abs(c) < RED_HALF ? A_red : A_out for c in xc]
    return BarMesh(x, h, A, nel)
end

# Strain from force F (σ0 = F/(A·g)), closed-form perfect plasticity
function strain_from_force(F, A, g, epsp0, lam0, sy0)
    sig0 = F / (A * g)
    sy = sy0 + H_kin * lam0
    if sig0 > sy
        return epsp0 + (sig0 - sy) / H_kin + sig0 / E0
    elseif sig0 < -sy
        return epsp0 - (-sig0 - sy) / H_kin + sig0 / E0
    else
        return epsp0 + sig0 / E0
    end
end

function state_from_force(F, A, g, epsp0, lam0, sy0)
    sig0 = F / (A * g)
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

function total_elongation(F, d_elem, epsp0, lam0, sy0_elem, mesh)
    total = 0.0
    for e in 1:mesh.nel
        g = (1.0 - d_elem[e])^2 + k_res
        total += mesh.h[e] * strain_from_force(F, mesh.A[e], g, epsp0[e], lam0[e], sy0_elem[e])
    end
    return total
end

function solve_force(d_elem, epsp0, lam0, sy0_elem, mesh, U; F_bracket=40.0)
    lo, hi = -F_bracket, F_bracket
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
        (hi - lo) < 1.0e-14 * F_bracket && break
    end
    return 0.5 * (lo + hi)
end

# Phase field: 1D P1 with energy density weighted by cross-section area
function solve_damage(H_elem, lc, mesh)
    n = length(mesh.x)
    K = zeros(n, n)
    rhs = zeros(n)
    for e in 1:mesh.nel
        he = mesh.h[e]
        Ae = mesh.A[e]
        He = H_elem[e]
        coef = gf / lc + 2.0 * He
        m = Ae * he / 6.0 .* [2.0 1.0; 1.0 2.0]
        kg = Ae * gf * lc / he .* [1.0 -1.0; -1.0 1.0]
        ke = coef .* m .+ kg
        fe = 2.0 * He .* (m * [1.0, 1.0])
        idx = [e, e + 1]
        K[idx, idx] .+= ke
        rhs[idx] .+= fe
    end
    return K \ rhs
end

function crack_surface(d_nodes, lc, mesh)
    Γ = 0.0
    for e in 1:mesh.nel
        dm = 0.5 * (d_nodes[e] + d_nodes[e + 1])
        dd = (d_nodes[e + 1] - d_nodes[e]) / mesh.h[e]
        Γ += mesh.A[e] * mesh.h[e] * (dm^2 / (2 * lc) + 0.5 * lc * dd^2)
    end
    return Γ
end

function run_case(lc; U_max=0.6, n_steps=4800)
    sy0 = calibrate_sigy0(lc)
    mesh = make_mesh(lc)
    n = length(mesh.x)
    @printf("  lc=%.4f mm (lc*/%.1f)  σy0=%.4f MPa (%.3fσt)  nel=%d h=%.4f\n",
            lc, lc_star() / lc, sy0, sy0 / sigt, mesh.nel, mesh.h[1])

    sy0_elem = fill(sy0, mesh.nel)
    sy0_elem[mesh.nel ÷ 2] *= 0.998   # weak element inside reduced section

    d_nodes = zeros(n)
    epsp = zeros(mesh.nel)
    lam = zeros(mesh.nel)
    H_elem = zeros(mesh.nel)

    Us = Float64[]; Fs = Float64[]; dmaxs = Float64[]; lammaxs = Float64[]; Γs = Float64[]
    # Profile snapshots (x, d, εp, λ) triggered by damage thresholds
    snap_levels = [0.3, 0.5, 0.7, 0.9, 0.99]
    snap_done = falses(length(snap_levels))
    xc = 0.5 .* (mesh.x[1:end - 1] .+ mesh.x[2:end])
    prof_dir = joinpath(OUTDIR, @sprintf("profiles_lc%.3f", lc))
    mkpath(prof_dir)
    # Nodal d + element quantities (εp/λ/A padded with NaN)
    save_profile(tag) = writedlm(joinpath(prof_dir, "profile_$tag.csv"),
        [["x_mm" "d" "epsp" "lambda" "A_mm2"];
         [mesh.x d_nodes [epsp; NaN] [lam; NaN] [mesh.A; NaN]]], ',')
    F = 0.0
    for step in 1:n_steps
        U = U_max * step / n_steps
        local d_elem
        H_trial = copy(H_elem)
        for it in 1:200
            d_elem = 0.5 .* (d_nodes[1:end - 1] .+ d_nodes[2:end])
            F = solve_force(d_elem, epsp, lam, sy0_elem, mesh, U)
            for e in 1:mesh.nel
                g = (1.0 - d_elem[e])^2 + k_res
                _, _, lam_new, ee = state_from_force(F, mesh.A[e], g, epsp[e], lam[e], sy0_elem[e])
                psi = 0.5 * E0 * ee^2 + sy0_elem[e] * lam_new
                H_trial[e] = max(H_trial[e], psi)
            end
            d_new = solve_damage(H_trial, lc, mesh)
            d_new = clamp.(max.(d_new, d_nodes), 0.0, 1.0)
            delta = maximum(abs.(d_new .- d_nodes))
            d_nodes = d_new
            delta < 1.0e-6 && break
        end
        d_elem = 0.5 .* (d_nodes[1:end - 1] .+ d_nodes[2:end])
        for e in 1:mesh.nel
            g = (1.0 - d_elem[e])^2 + k_res
            _, epsp[e], lam[e], _ = state_from_force(F, mesh.A[e], g, epsp[e], lam[e], sy0_elem[e])
        end
        H_elem = H_trial
        push!(Us, U); push!(Fs, F); push!(dmaxs, maximum(d_nodes))
        push!(lammaxs, maximum(lam)); push!(Γs, crack_surface(d_nodes, lc, mesh))
        for (i, thr) in enumerate(snap_levels)
            if !snap_done[i] && maximum(d_nodes) >= thr
                save_profile(@sprintf("d%02d", round(Int, 100 * thr)))
                snap_done[i] = true
            end
        end
        if step % 400 == 0
            @printf("    step %4d/%d U=%.4f F=%.4f N F/(Ar·σt)=%.4f d_max=%.4f\n",
                    step, n_steps, U, F, F / (A_red * sigt), maximum(d_nodes))
        end
        # Γ_lc,h is evaluated at d_max = 0.99 (Fig 14); stop there
        if maximum(d_nodes) >= 0.99
            @printf("    stop at step %d: d_max=%.4f >= 0.99 (U=%.4f)\n",
                    step, maximum(d_nodes), U)
            break
        end
    end
    Γ = crack_surface(d_nodes, lc, mesh)
    save_profile("final")
    @printf("  --> peak F = %.4f N (Ar·σt=%.3f, ratio %.4f); Γ_lc,h = %.4f mm² (target %.1f)\n\n",
            maximum(Fs), A_red * sigt, maximum(Fs) / (A_red * sigt), Γ, A_red)
    return Us, Fs, dmaxs, lammaxs, Γs, d_nodes, mesh, Γ
end

function main()
    mkpath(OUTDIR)
    lcs = lc_star() .* [0.1, 0.2, 0.5, 1.0]
    println("=" ^ 72)
    @printf("Fang2020 §3.5 reduced-section bar Γ-convergence: E0=%.0f, gf=%.3f, σt=%.2f, A=%.0f/Ar=%.0f mm²\n",
            E0, gf, sigt, A_out, A_red)
    @printf("lc* = %.4f mm; scanning lc = lc*/10, lc*/5, lc*/2, lc*\n", lc_star())
    println("=" ^ 72)

    results = []
    for lc in lcs
        Us, Fs, dmaxs, lammaxs, Γs, d_prof, mesh, Γ = run_case(lc)
        writedlm(joinpath(OUTDIR, @sprintf("gamma_lc%.3f.csv", lc)),
                 [["U_mm" "F_N" "d_max" "lambda_max" "Gamma_mm2"]; [Us Fs dmaxs lammaxs Γs]], ',')
        push!(results, (lc, Us, Fs, d_prof, mesh, Γ))
    end

    p1 = plot(xlabel="U / L", ylabel="F / (Ar·σt)",
              title="Fang2020 reduced-section bar (Fig 15)", legend=:topright)
    for (lc, Us, Fs, _, _, _) in results
        plot!(p1, Us ./ L, Fs ./ (A_red * sigt), lw=2,
              label=@sprintf("lc = lc*/%.0f", lc_star() / lc))
    end
    savefig(p1, joinpath(OUTDIR, "force_displacement.png"))

    p2 = plot(xlabel="x (mm)", ylabel="d", title="Final damage profiles", legend=:topright)
    for (lc, _, _, d_prof, mesh, Γ) in results
        plot!(p2, mesh.x, d_prof, lw=2,
              label=@sprintf("lc*/%.0f: Γ=%.3f mm²", lc_star() / lc, Γ))
    end
    savefig(p2, joinpath(OUTDIR, "damage_profiles.png"))
    # Γ-convergence summary (Fig 14)
    writedlm(joinpath(OUTDIR, "gamma_summary.csv"),
             [["lc_mm" "Gamma_mm2" "peak_F_N" "peak_ratio"];
              [[r[1] for r in results] [r[6] for r in results] [maximum(r[3]) for r in results] [maximum(r[3]) / (A_red * sigt) for r in results]]], ',')
    println("plots -> $OUTDIR/")
end

main()

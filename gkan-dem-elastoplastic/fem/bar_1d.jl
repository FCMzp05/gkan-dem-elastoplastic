using LinearAlgebra
using Printf
using DelimitedFiles
ENV["GKSwstype"] = "100"
using Plots
gr(show=false)

const E0   = 31_000.0
const gf   = 0.059
const sigt = 2.86
const L    = 100.0
const k_res = 1.0e-6
const H_kin = 1.0e-4 * E0

const OUTDIR = "outputs/fang2020_bar"

lc_star() = 0.0429 * gf * E0 / sigt^2

function calibrate_sigy0(lc)
    f(s) = s^4 + (2 * E0 * gf / lc) * s^2 - (E0^2 * gf^2 / (sigt * lc^2)) * s + E0^2 * gf^2 / lc^2
    lo, hi = sigt, 1.5 * sigt
    flo = f(lo)
    fhi = f(hi)
    if flo * fhi > 0
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
        found || error("sigma_y0 root not bracketed for lc=$lc")
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

struct BarMesh
    x::Vector{Float64}
    h::Vector{Float64}
    nel::Int
end

function make_mesh(lc)
    h_target = 0.05 * lc
    nel = ceil(Int, L / h_target)
    isodd(nel) && (nel += 1)
    x = collect(range(-L / 2, L / 2; length=nel + 1))
    h = diff(x)
    return BarMesh(x, h, nel)
end

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

function solve_damage(H_elem, lc, mesh)
    n = length(mesh.x)
    K = zeros(n, n)
    rhs = zeros(n)
    for e in 1:mesh.nel
        he = mesh.h[e]
        He = H_elem[e]
        coef = gf / lc + 2.0 * He
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

function run_case(lc; U_max=0.15, n_steps=1500)
    sy0 = calibrate_sigy0(lc)
    mesh = make_mesh(lc)
    n = length(mesh.x)
    @printf("  lc=%.4f mm (lc*/%.1f)  sigma_y0=%.4f MPa (%.3f sigt)  nel=%d h=%.4f\n",
            lc, lc_star() / lc, sy0, sy0 / sigt, mesh.nel, mesh.h[1])

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
        local d_elem
        H_trial = copy(H_elem)
        for it in 1:200
            d_elem = 0.5 .* (d_nodes[1:end - 1] .+ d_nodes[2:end])
            s = solve_stress(d_elem, epsp, lam, sy0_elem, mesh, U)
            for e in 1:mesh.nel
                g = (1.0 - d_elem[e])^2 + k_res
                _, _, lam_new, ee = state_from_stress(s, g, epsp[e], lam[e], sy0_elem[e])
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
            @printf("    step %3d/%d U=%.4f sigma=%.4f MPa d_max=%.4f lam_max=%.3e\n",
                    step, n_steps, U, s, maximum(d_nodes), maximum(lam))
        end
    end
    save_profile("final")
    return Us, sigs, dmaxs, lammaxs, sy0, d_nodes, mesh
end

function main()
    mkpath(OUTDIR)
    lcs = lc_star() .* [0.2, 0.5, 1.0]
    println("=" ^ 72)
    @printf("1D softening bar: E0=%.0f MPa, gf=%.3f N/mm, sigt=%.2f MPa, L=%.0f mm\n",
            E0, gf, sigt, L)
    @printf("lc* = %.4f mm; scanning lc = lc*/5, lc*/2, lc*\n", lc_star())
    println("=" ^ 72)

    results = []
    for lc in lcs
        Us, sigs, dmaxs, lammaxs, sy0, d_prof, mesh = run_case(lc)
        peak = maximum(sigs)
        @printf("  --> peak stress = %.4f MPa (sigt=%.2f, error %.2f%%)\n\n",
                peak, sigt, 100 * abs(peak - sigt) / sigt)
        writedlm(joinpath(OUTDIR, @sprintf("bar_lc%.3f.csv", lc)),
                 [["U_mm" "sigma_MPa" "d_max" "lambda_max"]; [Us sigs dmaxs lammaxs]], ',')
        push!(results, (lc, Us, sigs, d_prof, mesh))
    end

    p1 = plot(xlabel="U (mm)", ylabel="stress (MPa)",
              title="1D bar: stress-displacement", legend=:topright)
    for (lc, Us, sigs, _, _) in results
        plot!(p1, Us, sigs, lw=2, label=@sprintf("lc = %.2f mm", lc))
    end
    hline!(p1, [sigt], ls=:dash, c=:gray, label="sigt = 2.86 MPa")
    savefig(p1, joinpath(OUTDIR, "stress_displacement.png"))

    p2 = plot(xlabel="x (mm)", ylabel="d", title="Final damage profiles", legend=:topright)
    for (lc, _, _, d_prof, mesh) in results
        plot!(p2, mesh.x, d_prof, lw=2, label=@sprintf("lc = %.2f mm", lc))
    end
    savefig(p2, joinpath(OUTDIR, "damage_profiles.png"))
    println("plots -> $OUTDIR/stress_displacement.png, damage_profiles.png")
end

main()

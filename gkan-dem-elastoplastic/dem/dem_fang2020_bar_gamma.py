"""
Fang et al. (2020) Sec. 3.5 reduced-section bar: Gamma-convergence of the elasto-plastic phase field (hybrid DEM/ChebyKAN).

Geometry: L = 100 mm bar, A = 4 mm^2, reduced to Ar = 2 mm^2 over the central 0.15 L (triggers localization).
Model as in dem_fang2020_bar.py, with all energies/damage/forces weighted by A(x).
Expected: peak F/(Ar sigma_t) ~ 1, independent of lc;
Gamma_lc,h = int (d^2/2lc + lc/2 d'^2) A dx -> Ar = 2 mm^2 (evaluated at d_max >= 0.99).
Usage: python dem_fang2020_bar_gamma.py --device cuda
"""

import os
import sys
import csv
import math
import time

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# 1. Configuration / material (concrete)
# ==========================================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mat_E = 31_000.0     # MPa
    gf = 0.059           # N/mm
    sigt = 2.86          # MPa
    L = 100.0            # mm
    A_out = 4.0          # mm²
    A_red = 2.0          # mm²
    red_half = 7.5       # half-length of reduced section (0.075 L)
    kappa = 1.0e-6
    hk_frac = 1.0e-4
    weaken = 0.998       # slight sigma_y0 weakening of the central element

    h_over_lc = 0.05
    u_max = 0.6          # mm (stops early at d_max >= 0.99)
    n_steps = 2400

    neurons = 48
    hidden_layers = 3
    cheby_degree = 4

    n_epochs_lbfgs = 1
    lbfgs_dmax = 0.01
    n_epochs_rprop = 600
    n_epochs_rprop_active = 1200   # first staggered pass only
    n_epochs_rprop_later = 400     # later passes (warm-started network)
    optim_rel_tol = 1.0e-8
    max_stagger = 10
    max_stagger_active = 25
    stagger_tol = 1.0e-4
    damage_active = 0.05
    seed = 1
    output_dir = "outputs/dem_fang2020_bar_gamma"


conf = Config()


def lc_star():
    return 0.0429 * conf.gf * conf.mat_E / conf.sigt ** 2


def calibrate_sigy0(lc):
    E0, gf, st = conf.mat_E, conf.gf, conf.sigt

    def f(s):
        return (s ** 4 + (2 * E0 * gf / lc) * s ** 2
                - (E0 ** 2 * gf ** 2 / (st * lc ** 2)) * s + E0 ** 2 * gf ** 2 / lc ** 2)

    ss = np.linspace(st, 2.0 * st, 4001)
    fs = f(ss)
    idx = np.where(fs[:-1] * fs[1:] <= 0)[0]
    if len(idx) == 0:
        raise RuntimeError(f"sigy0 root not bracketed for lc={lc}")
    lo, hi = ss[idx[0]], ss[idx[0] + 1]
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) * flo <= 0:
            hi = mid
        else:
            lo = mid
            flo = f(lo)
    return 0.5 * (lo + hi)


# ==========================================
# 2. ChebyKAN
# ==========================================
class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree):
        super().__init__()
        self.degree = degree
        self.linear = nn.Linear(in_features * (degree + 1), out_features)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x_norm = torch.tanh(x)
        t_list = [torch.ones_like(x_norm), x_norm]
        for _ in range(2, self.degree + 1):
            t_list.append(2 * x_norm * t_list[-1] - t_list[-2])
        return self.linear(torch.cat(t_list[: self.degree + 1], dim=-1))


class ChebyKANMLP(nn.Module):
    def __init__(self, input_size, hidden, output_size, n_hidden, degree):
        super().__init__()
        sizes = [input_size] + [hidden] * n_hidden + [output_size]
        self.layers = nn.ModuleList(
            [ChebyKANLayer(sizes[i], sizes[i + 1], degree) for i in range(len(sizes) - 1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


def get_optimizer(params, kind):
    if kind == "LBFGS":
        return optim.LBFGS(params, lr=0.5, max_iter=300, max_eval=1500,
                           history_size=100, line_search_fn="strong_wolfe",
                           tolerance_change=1.0e-14, tolerance_grad=1.0e-14)
    return optim.Rprop(params, lr=1.0e-5, step_sizes=(1.0e-10, 50))


# ==========================================
# 3. 1D elasto-plastic kernel (internal units: E=1)
# ==========================================
def plastic_core_1d(eps, ep0, lam0, sy0_e, hk):
    st = eps - ep0
    sy = sy0_e + hk * lam0
    f = torch.relu(torch.abs(st) - sy)
    dl = f / (1.0 + hk)
    sn = st - torch.sign(st) * dl
    lam_new = lam0 + dl
    psi_e = 0.5 * sn ** 2
    W = psi_e + sy0_e * dl + 0.5 * hk * (lam_new ** 2 - lam0 ** 2)
    driving = psi_e + sy0_e * lam_new + 0.5 * hk * lam_new ** 2
    ep_new = ep0 + torch.sign(st) * dl
    return W, driving, ep_new, lam_new, sn


def elem_strain(u, h_int):
    return (u[1:] - u[:-1]) / h_int


def energy(u, d_nodes, h_int, A_e, ep0, lam0, sy0_e, hk, lc_int):
    eps = elem_strain(u, h_int)
    d_e = 0.5 * (d_nodes[:-1] + d_nodes[1:])
    g = (1.0 - d_e) ** 2 + conf.kappa
    W, _, _, _, _ = plastic_core_1d(eps, ep0, lam0, sy0_e, hk)
    e_el = (h_int * A_e * g * W).sum()
    dgrad = (d_nodes[1:] - d_nodes[:-1]) / h_int
    gf_int = lc_int
    e_d = (h_int * A_e * (gf_int / (2 * lc_int) * d_e ** 2
                          + 0.5 * gf_int * lc_int * dgrad ** 2)).sum()
    return e_el, e_d


# ==========================================
# 4. 1D damage solve (tridiagonal, A-weighted)
# ==========================================
def solve_damage_1d(H_e, d_prev, h_int, A_e, lc_int):
    nel = len(H_e)
    n = nel + 1
    gf_int = lc_int
    main = np.zeros(n)
    off = np.zeros(nel)
    rhs = np.zeros(n)
    coef = gf_int / lc_int + 2.0 * H_e
    for e in range(nel):
        he = h_int[e]
        Ae = A_e[e]
        m11 = Ae * he / 3.0
        m12 = Ae * he / 6.0
        kg = Ae * gf_int * lc_int / he
        main[e] += coef[e] * m11 + kg
        main[e + 1] += coef[e] * m11 + kg
        off[e] += coef[e] * m12 - kg
        fe = 2.0 * H_e[e] * Ae * he / 2.0
        rhs[e] += fe
        rhs[e + 1] += fe
    K = diags([off, main, off], [-1, 0, 1], format="csr")
    d = spsolve(K, rhs)
    return np.clip(np.maximum(d, d_prev), 0.0, 1.0)


def crack_surface(d_nodes, h_np, A_np, lc_int, Ls):
    """Gamma_lc,h = int (d^2/2lc + lc/2 d'^2) A dx, in mm^2."""
    d_e = 0.5 * (d_nodes[:-1] + d_nodes[1:])
    dgrad = (d_nodes[1:] - d_nodes[:-1]) / h_np
    lc_phys = lc_int * Ls
    h_phys = h_np * Ls
    dgrad_phys = dgrad / Ls
    return float(np.sum(A_np * h_phys * (d_e ** 2 / (2 * lc_phys)
                                         + 0.5 * lc_phys * dgrad_phys ** 2)))


# ==========================================
# 5. Displacement field + fitting
# ==========================================
class Field1D:
    def __init__(self, net, x):
        self.net = net
        self.x = x
        self.lift = x + 0.5
        self.bubble = 4.0 * (x + 0.5) * (0.5 - x)
        self.u_comm = torch.zeros_like(x)
        self.dl = torch.zeros((), device=x.device)

    def compute(self, d_nodes):
        feats = torch.stack([self.x, d_nodes], dim=1)
        out = self.net(feats)[:, 0]
        return self.u_comm + self.dl * (self.lift + self.bubble * out)


def fit(field, d_nodes, h_int, A_e, ep0, lam0, sy0_e, hk, lc_int, num_epochs, optimizer, kind):
    losses = []

    def closure():
        optimizer.zero_grad()
        u = field.compute(d_nodes)
        e_el, e_d = energy(u, d_nodes, h_int, A_e, ep0, lam0, sy0_e, hk, lc_int)
        total = e_el + e_d
        if not torch.isfinite(total):
            raise FloatingPointError("invalid energy")
        loss = torch.log10(total.clamp_min(1e-30))
        loss.backward()
        losses.append(loss.item())
        return loss

    if kind == "lbfgs":
        for _ in range(num_epochs):
            optimizer.step(closure)
        return losses
    eps_m = np.finfo(float).eps
    counter, prev = 0, 0.0
    for _ in range(num_epochs):
        closure()
        optimizer.step()
        curr = losses[-1]
        if abs(curr - prev) / (abs(prev) + eps_m) > conf.optim_rel_tol:
            counter = 0
        else:
            counter += 1
            if counter >= 20:
                break
        prev = curr
    return losses


# ==========================================
# 6. Single case
# ==========================================
def run_case(lc):
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    device = torch.device(conf.device)
    dtype = torch.float64

    sy0 = calibrate_sigy0(lc)
    h_t = conf.h_over_lc * lc
    nel = int(math.ceil(conf.L / h_t))
    if nel % 2 == 1:
        nel += 1
    x_phys = np.linspace(-conf.L / 2, conf.L / 2, nel + 1)
    xc = 0.5 * (x_phys[:-1] + x_phys[1:])
    A_np = np.where(np.abs(xc) < conf.red_half, conf.A_red, conf.A_out)

    Ls = conf.L
    non_dim = math.sqrt(conf.mat_E * lc / conf.gf)
    stress_scale = conf.mat_E / non_dim
    lc_int = lc / Ls
    sy0_int = sy0 / stress_scale
    hk = conf.hk_frac

    print(f"\n  lc={lc:.4f} mm (lc*/{lc_star() / lc:.1f})  sy0={sy0:.4f} MPa "
          f"({sy0 / conf.sigt:.3f} sigt)  nel={nel}  non_dim={non_dim:.4g}")

    x = torch.from_numpy(x_phys / Ls).to(dtype).to(device)
    h_int = x[1:] - x[:-1]
    A_e = torch.from_numpy(A_np).to(dtype).to(device)
    sy0_e = torch.full((nel,), sy0_int, dtype=dtype, device=device)
    sy0_e[nel // 2] *= conf.weaken

    net = ChebyKANMLP(2, conf.neurons, 1, conf.hidden_layers, conf.cheby_degree).to(dtype).to(device)
    field = Field1D(net, x)

    d_nodes = torch.zeros(nel + 1, dtype=dtype, device=device)
    ep = torch.zeros(nel, dtype=dtype, device=device)
    lam = torch.zeros(nel, dtype=dtype, device=device)
    H_e = torch.zeros(nel, dtype=dtype, device=device)

    U_all = np.linspace(0.0, conf.u_max, conf.n_steps + 1) / Ls * non_dim
    dU = np.diff(U_all)
    h_np = h_int.cpu().numpy()

    opt_r = get_optimizer(net.parameters(), "RPROP")
    Us, Fs, dmaxs, lammaxs, Gammas = [], [], [], [], []
    # profile snapshots
    snap_levels = [0.3, 0.5, 0.7, 0.9, 0.99]
    snap_done = [False] * len(snap_levels)
    prof_dir = os.path.join(conf.output_dir, f"profiles_lc{lc:.3f}")
    os.makedirs(prof_dir, exist_ok=True)

    def save_profile(tag):
        d_now = d_nodes.cpu().numpy()
        ep_now = np.append(ep.cpu().numpy() / non_dim, np.nan)
        lam_now = np.append(lam.cpu().numpy() / non_dim, np.nan)
        A_col = np.append(A_np, np.nan)
        with open(os.path.join(prof_dir, f"profile_{tag}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_mm", "d", "epsp", "lambda", "A_mm2"])
            w.writerows(zip(x_phys, d_now, ep_now, lam_now, A_col))

    t0 = time.time()
    for step in range(1, conf.n_steps + 1):
        field.dl = torch.tensor(float(dU[step - 1]), dtype=dtype, device=device)
        active = float(d_nodes.max()) >= conf.damage_active
        max_st = conf.max_stagger_active if active else conf.max_stagger
        n_ep = conf.n_epochs_rprop_active if active else conf.n_epochs_rprop
        H_step = H_e.clone()
        d_prev_np = None
        for fp in range(max_st):
            if float(d_nodes.max()) < conf.lbfgs_dmax and conf.n_epochs_lbfgs > 0:
                opt_l = get_optimizer(net.parameters(), "LBFGS")
                fit(field, d_nodes, h_int, A_e, ep, lam, sy0_e, hk, lc_int,
                    conf.n_epochs_lbfgs, opt_l, "lbfgs")
            budget = n_ep if fp == 0 else conf.n_epochs_rprop_later
            fit(field, d_nodes, h_int, A_e, ep, lam, sy0_e, hk, lc_int, budget, opt_r, "rprop")
            with torch.no_grad():
                u = field.compute(d_nodes)
                eps = elem_strain(u, h_int)
                _, driving, _, _, _ = plastic_core_1d(eps, ep, lam, sy0_e, hk)
                H_step = torch.maximum(H_step, driving)
            d_np = solve_damage_1d(H_step.cpu().numpy(), d_nodes.cpu().numpy(),
                                   h_np, A_np, lc_int)
            d_nodes = torch.from_numpy(d_np).to(dtype).to(device)
            if d_prev_np is not None and np.max(np.abs(d_np - d_prev_np)) < conf.stagger_tol:
                break
            d_prev_np = d_np

        with torch.no_grad():
            u = field.compute(d_nodes)
            eps = elem_strain(u, h_int)
            d_e = 0.5 * (d_nodes[:-1] + d_nodes[1:])
            g = (1.0 - d_e) ** 2 + conf.kappa
            _, driving, ep_new, lam_new, sn = plastic_core_1d(eps, ep, lam, sy0_e, hk)
            H_e = torch.maximum(H_step, driving)
            ep, lam = ep_new.clone(), lam_new.clone()
            field.u_comm = u.clone()
            # axial force F = A g sigma (uniform along the bar), N
            F_N = float((A_e * g * sn).mean()) * stress_scale

        U_mm = conf.u_max * step / conf.n_steps
        d_max = float(d_nodes.max())
        Us.append(U_mm)
        Fs.append(F_N)
        dmaxs.append(d_max)
        lammaxs.append(float(lam.max()) / non_dim)
        Gammas.append(crack_surface(d_nodes.cpu().numpy(), h_np, A_np, lc_int, Ls))
        for i, thr in enumerate(snap_levels):
            if not snap_done[i] and d_max >= thr:
                save_profile(f"d{int(100 * thr):02d}")
                snap_done[i] = True
        if step % 100 == 0 or step == 1:
            print(f"    step {step:4d}/{conf.n_steps} U={U_mm:.4f} F={F_N:.4f} N "
                  f"F/(Ar·sigt)={F_N / (conf.A_red * conf.sigt):.4f} d_max={d_max:.4f} "
                  f"t={time.time() - t0:.0f}s")
        if d_max >= 0.99:
            print(f"    stop at step {step}: d_max={d_max:.4f} >= 0.99 (U={U_mm:.4f})")
            break

    save_profile("final")
    Gamma = Gammas[-1]
    peak = max(Fs)
    print(f"  --> peak F = {peak:.4f} N (Ar·sigt={conf.A_red * conf.sigt:.3f}, "
          f"ratio {peak / (conf.A_red * conf.sigt):.4f}); Gamma = {Gamma:.4f} mm² (target {conf.A_red})")
    return Us, Fs, dmaxs, lammaxs, Gammas, x_phys, d_nodes.cpu().numpy(), Gamma


def main():
    if "--device" in sys.argv:
        conf.device = sys.argv[sys.argv.index("--device") + 1]
    os.makedirs(conf.output_dir, exist_ok=True)
    lcs = [lc_star() * f for f in (0.1, 0.2, 0.5, 1.0)]
    print("=" * 72)
    print(f"Fang2020 §3.5 reduced-section bar (DEM/ChebyKAN hybrid) | device={conf.device}")
    print(f"E={conf.mat_E} MPa, gf={conf.gf} N/mm, sigt={conf.sigt} MPa, "
          f"A={conf.A_out}/Ar={conf.A_red} mm²")
    print(f"lc* = {lc_star():.4f} mm; scanning lc = lc*/10, lc*/5, lc*/2, lc*")
    print("=" * 72)

    results = []
    for lc in lcs:
        Us, Fs, dmaxs, lammaxs, Gammas, xg, d_prof, Gamma = run_case(lc)
        with open(os.path.join(conf.output_dir, f"gamma_lc{lc:.3f}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["U_mm", "F_N", "d_max", "lambda_max", "Gamma_mm2"])
            w.writerows(zip(Us, Fs, dmaxs, lammaxs, Gammas))
        results.append((lc, Us, Fs, xg, d_prof, Gamma))

    with open(os.path.join(conf.output_dir, "gamma_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lc_mm", "Gamma_mm2", "peak_F_N", "peak_ratio"])
        for lc, Us, Fs, _, _, Gamma in results:
            w.writerow([lc, Gamma, max(Fs), max(Fs) / (conf.A_red * conf.sigt)])

    fig, ax = plt.subplots(figsize=(7, 5))
    for lc, Us, Fs, _, _, _ in results:
        ax.plot(np.array(Us) / conf.L, np.array(Fs) / (conf.A_red * conf.sigt), lw=2,
                label=f"lc = lc*/{lc_star() / lc:.0f}")
    ax.set_xlabel("U / L")
    ax.set_ylabel("F / (Ar·sigt)")
    ax.set_title("Fang2020 reduced-section bar (DEM, Fig 15)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(conf.output_dir, "force_displacement.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for lc, _, _, xg, d_prof, Gamma in results:
        ax.plot(xg, d_prof, lw=2, label=f"lc*/{lc_star() / lc:.0f}: Gamma={Gamma:.3f} mm²")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("d")
    ax.set_title("Final damage profiles (DEM)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(conf.output_dir, "damage_profiles.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nplots -> {conf.output_dir}/")


if __name__ == "__main__":
    main()

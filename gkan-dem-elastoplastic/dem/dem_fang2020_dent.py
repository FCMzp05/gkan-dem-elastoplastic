"""
Fang et al. (2020) Sec. 4.1 double-edge-notched tension (DENT): elasto-plastic
phase-field fracture with hybrid DEM/ChebyKAN (network solves u, mesh solves d).

Model as in dem_fang2020_lplate.py: Rankine perfect plasticity + AT2, sigma_y0
calibrated from Eq. (31). Units: mm, MPa, N/mm; reaction force in kN.

Geometry: 60x120 mm plate (10 mm thick, plane stress), two edge notches 10 mm
deep and 2 mm high, offset by 5/15 mm (left y=60+off/2, right y=60-off/2).
BCs: bottom u=v=0; top u=0, v=U. Harmonic lift phi (0 bottom / 1 top):
  u = phi(1-phi)*out_u*dl;  v = [phi(1-phi)*out_v + phi]*dl.

Usage: python dem_fang2020_dent.py --device cuda   # sweeps (offset, lc), resumable
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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# TF32 matmul on Ampere+; energy error O(1e-6) is negligible
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import gmsh
except ImportError:
    print("Error: pip install gmsh")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Geometry (mm) ----
    W = 60.0
    H = 120.0
    notch_depth = 10.0
    notch_half_h = 1.0
    thickness = 10.0

    # ---- Material (concrete, paper Sec. 3) ----
    mat_E = 31_000.0
    mat_nu = 0.2
    gf = 0.059
    sigt = 2.86
    kappa = 1.0e-6
    hk_frac = 5.0e-5

    # ---- Case sweep (offset, lc) ----
    # First 4: Fang2020 cases; then offset in {7.5,10,12.5} @ lc=3 (crack-pattern
    # transition), (15,5) for lc sensitivity, (10,10) for large-lc comparison.
    cases = [(5.0, 3.0), (5.0, 10.0), (15.0, 3.0), (15.0, 10.0),
             (7.5, 3.0), (10.0, 3.0), (12.5, 3.0), (15.0, 5.0), (10.0, 10.0)]
    h_over_lc = 0.2
    h_coarse = 4.0

    # ---- Loading ----
    u_max = 0.20
    n_steps = 250

    # ---- Network ----
    neurons = 64
    hidden_layers = 1
    mpnn_layer = 2
    aggregate_fun = "mean"
    activation = "SiLU"
    cheby_degree = 3

    # ---- Optimizer / staggering ----
    n_epochs_lbfgs = 30
    lbfgs_alpha_max = 0.008
    # Epoch counts are upper bounds; early stop: rel_tol over `window` iters after min_iters
    n_epochs_rprop = 1500
    n_epochs_rprop_active = 3000     # fp=1 only (carries the load increment)
    n_epochs_rprop_later = 800       # fp>=2: warm-started, smaller budget
    optim_rel_tol = 3.0e-6
    optim_rel_tol_later = 1.0e-5     # looser tolerance for fp>=2
    optim_stable_window = 50         # consecutive iters below tolerance
    optim_stable_window_later = 30
    optim_min_iters = 300            # minimum iters before early stop
    optim_min_iters_later = 100
    weight_decay = 1.0e-6
    max_stagger = 12
    max_stagger_active = 15
    min_stagger_active = 4
    stagger_abs_tol = 1.0e-3
    stagger_abs_tol_active = 2.0e-4
    stagger_alpha_slack = 5.0        # near-stagnation when dd < slack*tol
    stagger_force_rtol = 1.0e-3      # relative reaction change between fp iterations
    damage_active_alpha = 0.05
    progress_every = 500
    compile_net = False              # enabled by --compile
    n_epochs_polish = 600
    polish_rel_tol = 1.0e-6

    # ---- Output / early stop ----
    save_every_low = 10
    save_every_high = 5
    damage_high_threshold = 0.5
    early_stop_force_frac = 0.15
    early_stop_min_alpha = 0.9

    seed = 1
    _revision_root = os.environ.get("DEM_REVISION_OUTPUT_ROOT")
    output_base = (os.path.join(_revision_root, "dem_fang2020_dent")
                   if _revision_root else "outputs/dem_fang2020_dent")


conf = Config()


# ==========================================
# 2. sigma_y0 calibration (Eq. 31)
# ==========================================
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
# 3. Mesh (same DENT geometry as the FEM reference)
# ==========================================
def _signed_areas(nodes, conn):
    x, y = nodes[:, 0], nodes[:, 1]
    return 0.5 * (
        x[conn[:, 0]] * (y[conn[:, 1]] - y[conn[:, 2]]) +
        x[conn[:, 1]] * (y[conn[:, 2]] - y[conn[:, 0]]) +
        x[conn[:, 2]] * (y[conn[:, 0]] - y[conn[:, 1]]))


def generate_dent_mesh(output_dir, offset, lc):
    h_f = conf.h_over_lc * lc
    h_c = conf.h_coarse
    yl = 60.0 + offset / 2
    yr = 60.0 - offset / 2
    print(f"\n>>> Generating DENT mesh (offset={offset}, h_fine={h_f} mm = lc/5, "
          f"h_coarse={h_c} mm, y_left={yl}, y_right={yr})...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("dent")
    geo = gmsh.model.geo
    W, H = conf.W, conf.H
    nd, nh = conf.notch_depth, conf.notch_half_h

    # Rectangular notches (10 mm deep x 2 mm high, flat tip)
    p_BL = geo.addPoint(0.0, 0.0, 0.0, h_c)
    p_BR = geo.addPoint(W, 0.0, 0.0, h_c)
    p_R_lo = geo.addPoint(W, yr - nh, 0.0, h_f)
    p_R_ti1 = geo.addPoint(W - nd, yr - nh, 0.0, h_f)
    p_R_ti2 = geo.addPoint(W - nd, yr + nh, 0.0, h_f)
    p_R_hi = geo.addPoint(W, yr + nh, 0.0, h_f)
    p_TR = geo.addPoint(W, H, 0.0, h_c)
    p_TL = geo.addPoint(0.0, H, 0.0, h_c)
    p_L_hi = geo.addPoint(0.0, yl + nh, 0.0, h_f)
    p_L_ti1 = geo.addPoint(nd, yl + nh, 0.0, h_f)
    p_L_ti2 = geo.addPoint(nd, yl - nh, 0.0, h_f)
    p_L_lo = geo.addPoint(0.0, yl - nh, 0.0, h_f)

    pts = [(p_BL, p_BR), (p_BR, p_R_lo), (p_R_lo, p_R_ti1), (p_R_ti1, p_R_ti2),
           (p_R_ti2, p_R_hi), (p_R_hi, p_TR), (p_TR, p_TL), (p_TL, p_L_hi),
           (p_L_hi, p_L_ti1), (p_L_ti1, p_L_ti2), (p_L_ti2, p_L_lo), (p_L_lo, p_BL)]
    lines = [geo.addLine(a, b) for a, b in pts]
    loop = geo.addCurveLoop(lines)
    surf = geo.addPlaneSurface([loop])
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [surf], 1)

    # Box refinement: band between the two notch tips
    ylo = min(yl, yr) - 12.0
    yhi = max(yl, yr) + 12.0
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_f)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_c)
    gmsh.model.mesh.field.setNumber(1, "XMin", 0.0)
    gmsh.model.mesh.field.setNumber(1, "XMax", W)
    gmsh.model.mesh.field.setNumber(1, "YMin", ylo)
    gmsh.model.mesh.field.setNumber(1, "YMax", yhi)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 8.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(os.path.join(output_dir, "mesh.msh"))

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(nodeCoords).reshape(-1, 3)[:, :2]
    elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(dim=2)
    tri_conn = None
    for i, et in enumerate(elemTypes):
        if et == 2:
            tri_conn = elemNodeTags[i].reshape(-1, 3)
            break
    node_map = {tag: i for i, tag in enumerate(nodeTags)}
    conn = np.array([[node_map[tri_conn[i, j]] for j in range(3)]
                     for i in range(tri_conn.shape[0])], dtype=np.int64)
    gmsh.finalize()

    areas = _signed_areas(nodes, conn)
    bad = areas < 0
    if np.any(bad):
        conn[bad, 1], conn[bad, 2] = conn[bad, 2].copy(), conn[bad, 1].copy()
    areas = np.abs(areas)
    print(f"  OK: {nodes.shape[0]:,} nodes, {conn.shape[0]:,} triangles (lc/h={lc / h_f:.1f})")
    return nodes, conn, areas


def boundary_node_sets(nodes):
    tol = 1.0e-6 * conf.H
    bottom = np.where(np.abs(nodes[:, 1]) < tol)[0]
    top = np.where(np.abs(nodes[:, 1] - conf.H) < tol)[0]
    return np.sort(bottom).astype(np.int64), np.sort(top).astype(np.int64)


# ==========================================
# 4. P1 assembly (shared by harmonic lift and damage solver)
# ==========================================
class P1Assembler:
    def __init__(self, nodes, conn, areas):
        n = nodes.shape[0]
        self.n_nodes = n
        self.conn = conn
        areas = areas.astype(np.float64)
        x = nodes[:, 0].astype(np.float64)
        y = nodes[:, 1].astype(np.float64)
        i0, i1, i2 = conn[:, 0], conn[:, 1], conn[:, 2]
        b = np.stack([y[i1] - y[i2], y[i2] - y[i0], y[i0] - y[i1]], axis=1) / (2.0 * areas[:, None])
        c = np.stack([x[i2] - x[i1], x[i0] - x[i2], x[i1] - x[i0]], axis=1) / (2.0 * areas[:, None])
        self.Ge = areas[:, None, None] * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
        mloc = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0
        self.Me = areas[:, None, None] * mloc[None, :, :]
        self.rows = np.repeat(conn, 3, axis=1).ravel()
        self.cols = np.tile(conn, (1, 3)).ravel()
        used = np.zeros(n, dtype=bool)
        used[conn.ravel()] = True
        self.orphans = np.where(~used)[0]

    def assemble(self, coef_mass, coef_grad):
        ke = coef_mass[:, None, None] * self.Me + coef_grad[:, None, None] * self.Ge
        vals = ke.ravel()
        rows, cols = self.rows, self.cols
        if len(self.orphans) > 0:
            vals = np.concatenate([vals, np.ones(len(self.orphans))])
            rows = np.concatenate([rows, self.orphans])
            cols = np.concatenate([cols, self.orphans])
        return coo_matrix((vals, (rows, cols)), shape=(self.n_nodes, self.n_nodes)).tocsr()

    def mass_rhs(self, coef_elem):
        fe = coef_elem[:, None, None] * self.Me
        rhs = np.zeros(self.n_nodes)
        np.add.at(rhs, self.conn.ravel(), fe.sum(axis=2).ravel())
        return rhs


def solve_harmonic_lift(assembler, bottom, top):
    n = assembler.n_nodes
    K = assembler.assemble(np.zeros(len(assembler.Ge)), np.ones(len(assembler.Ge))).tolil()
    rhs = np.zeros(n)
    for i in np.concatenate([bottom, top]):
        K.rows[i] = [i]
        K.data[i] = [1.0]
    rhs[top] = 1.0
    phi = spsolve(K.tocsr(), rhs)
    phi = np.clip(phi, 0.0, 1.0)
    phi[bottom] = 0.0
    phi[top] = 1.0
    return phi


class MeshDamageSolver:
    def __init__(self, assembler, gf_int, lc_int):
        self.asm = assembler
        self.gf = gf_int
        self.lc = lc_int
        self.nel = len(assembler.Ge)

    def solve(self, h_elem, alpha_committed):
        coef_m = 2.0 * h_elem + self.gf / self.lc
        coef_g = np.full(self.nel, self.gf * self.lc)
        K = self.asm.assemble(coef_m, coef_g)
        rhs = self.asm.mass_rhs(2.0 * h_elem)
        alpha = spsolve(K, rhs)
        return np.clip(np.maximum(alpha, alpha_committed), 0.0, 1.0)


# ==========================================
# 5. ChebyKAN / GNN
# ==========================================
class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=3):
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
    def __init__(self, input_size, hidden, output_size, num_hidden_layers, degree):
        super().__init__()
        sizes = [input_size] + [hidden] * num_hidden_layers
        if output_size:
            sizes.append(output_size)
        self.layers = nn.ModuleList(
            [ChebyKANLayer(sizes[i], sizes[i + 1], degree) for i in range(len(sizes) - 1)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


def make_mlp(in_size, hidden, out_size, n_hidden):
    return ChebyKANMLP(in_size, hidden, out_size, n_hidden, conf.cheby_degree)


def scatter_add(src, index, dim_size):
    out = torch.zeros((dim_size,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    view = (index.shape[0],) + (1,) * (src.dim() - 1)
    out.scatter_add_(0, index.view(view).expand_as(src), src)
    return out


def scatter_mean(src, index, dim_size):
    out = scatter_add(src, index, dim_size)
    cnt = torch.zeros((dim_size,), dtype=src.dtype, device=src.device)
    cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp_min(1.0).view(-1, *([1] * (src.dim() - 1)))


class EdgeSet:
    def __init__(self, senders, receivers):
        self.senders = senders
        self.receivers = receivers


class MultiGraph:
    def __init__(self, node_features, edge_sets, T_conn, area_elem, edge_features=None):
        self.node_features = node_features
        self.edge_sets = edge_sets
        self.T_conn = T_conn
        self.area_elem = area_elem
        self.edge_features = edge_features


class ProcessorMean(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.message_processor = make_mlp(2 * latent + 3, latent, latent, 2)
        self.node_processor = make_mlp(latent, latent, latent, 2)

    def forward(self, graph):
        s, r = graph.edge_sets.senders, graph.edge_sets.receivers
        messages = self.message_processor(
            torch.cat([graph.node_features[s], graph.node_features[r], graph.edge_features], dim=-1))
        agg = scatter_mean(messages, r, graph.node_features.shape[0])
        node_latent = self.node_processor(agg) + graph.node_features
        return MultiGraph(node_latent, graph.edge_sets, graph.T_conn, graph.area_elem, graph.edge_features)


class GraphNeuralNetwork(nn.Module):
    def __init__(self, output_dimension=2):
        super().__init__()
        base_input_dim = 2 * 4 + 1
        self.encoder_nodes = make_mlp(base_input_dim, conf.neurons, conf.neurons, conf.hidden_layers)
        self.processors = nn.ModuleList([ProcessorMean(conf.neurons) for _ in range(conf.mpnn_layer)])
        self.decoder_node = make_mlp(conf.neurons, conf.neurons, output_dimension, conf.hidden_layers)
        total = sum(p.numel() for p in self.parameters())
        print(f"  GNN+ChebyKAN: neurons={conf.neurons}, mpnn={conf.mpnn_layer}, "
              f"degree={conf.cheby_degree}, params={total:,}")

    def forward(self, data):
        graph = MultiGraph(self.encoder_nodes(data.node_features), data.edge_sets,
                           data.T_conn, data.area_elem, data.edge_features)
        for proc in self.processors:
            graph = proc(graph)
        return self.decoder_node(graph.node_features)


def get_optimizer(params, kind):
    if kind == "LBFGS":
        return optim.LBFGS(params, lr=0.5, max_iter=20000, max_eval=20000000,
                           history_size=250, line_search_fn="strong_wolfe",
                           tolerance_change=np.finfo(float).eps,
                           tolerance_grad=np.finfo(float).eps)
    return optim.Rprop(params, lr=1.0e-5, step_sizes=(1.0e-10, 50))


# ==========================================
# 6. Strain / Rankine plasticity / energy
# ==========================================
def field_grads(inp, field, area_elem, T):
    grad_x = ((inp[T[:, 1], -1] - inp[T[:, 2], -1]) * field[T[:, 0]]
              + (inp[T[:, 2], -1] - inp[T[:, 0], -1]) * field[T[:, 1]]
              + (inp[T[:, 0], -1] - inp[T[:, 1], -1]) * field[T[:, 2]])
    grad_y = ((inp[T[:, 2], -2] - inp[T[:, 1], -2]) * field[T[:, 0]]
              + (inp[T[:, 0], -2] - inp[T[:, 2], -2]) * field[T[:, 1]]
              + (inp[T[:, 1], -2] - inp[T[:, 0], -2]) * field[T[:, 2]])
    return grad_x / area_elem / 2, grad_y / area_elem / 2


def strains(inp, u, v, area_elem, T_conn):
    gux, guy = field_grads(inp, u, area_elem, T_conn)
    gvx, gvy = field_grads(inp, v, area_elem, T_conn)
    return gux, gvy, 0.5 * (guy + gvx)


class PlasticState:
    def __init__(self, n_elem=None, device=None, dtype=None, _empty=False):
        if _empty:
            return
        z = lambda: torch.zeros(n_elem, device=device, dtype=dtype)
        self.ep11, self.ep22, self.ep12, self.lam = z(), z(), z(), z()

    def clone(self):
        new = PlasticState(_empty=True)
        new.ep11 = self.ep11.clone()
        new.ep22 = self.ep22.clone()
        new.ep12 = self.ep12.clone()
        new.lam = self.lam.clone()
        return new


class Mat:
    def __init__(self, device, dtype, sigy0_int):
        E, nu = 1.0, conf.mat_nu
        t = lambda x: torch.tensor(x, device=device, dtype=dtype)
        self.E = t(E)
        self.nu = t(nu)
        self.C11 = t(E / (1 - nu ** 2))
        self.C12 = t(nu * E / (1 - nu ** 2))
        self.mu = t(E / (2 * (1 + nu)))
        self.lam_ps = t(nu * E / (1 - nu ** 2))
        self.sigy0 = t(sigy0_int)
        self.hk = t(conf.hk_frac)


def rankine_core(s11, s22, s12, pstate, mat):
    ee11 = s11 - pstate.ep11
    ee22 = s22 - pstate.ep22
    ee12 = s12 - pstate.ep12

    t11 = mat.C11 * ee11 + mat.C12 * ee22
    t22 = mat.C12 * ee11 + mat.C11 * ee22
    t12 = 2.0 * mat.mu * ee12

    c = 0.5 * (t11 + t22)
    R = torch.sqrt(torch.clamp((0.5 * (t11 - t22)) ** 2 + t12 ** 2, min=1.0e-30))
    s1t = c + R
    s2t = c - R
    sy = mat.sigy0 + mat.hk * pstate.lam

    A = mat.C11 + mat.hk
    B = mat.C12
    f1 = torch.relu(s1t - sy)
    dl1_s = f1 / A
    s2n_s = s2t - B * dl1_s
    corner = s2n_s > sy + mat.hk * dl1_s
    f2 = torch.relu(s2t - sy)
    Bh = mat.C12 + mat.hk
    det = A * A - Bh * Bh
    dl1_c = torch.clamp((A * f1 - Bh * f2) / det, min=0.0)
    dl2_c = torch.clamp((A * f2 - Bh * f1) / det, min=0.0)
    dl1 = torch.where(corner, dl1_c, dl1_s)
    dl2 = torch.where(corner, dl2_c, torch.zeros_like(dl2_c))

    s1n = s1t - mat.C11 * dl1 - mat.C12 * dl2
    s2n = s2t - mat.C12 * dl1 - mat.C11 * dl2

    e1 = (s1n - mat.nu * s2n) / mat.E
    e2 = (s2n - mat.nu * s1n) / mat.E
    psi_e = 0.5 * (s1n * e1 + s2n * e2)
    dltot = dl1 + dl2
    lam_new = pstate.lam + dltot
    W_inc = psi_e + mat.sigy0 * dltot + 0.5 * mat.hk * (lam_new ** 2 - pstate.lam ** 2)

    trp = torch.relu(e1 + e2)
    psi_plus = 0.5 * mat.lam_ps * trp ** 2 + mat.mu * (torch.relu(e1) ** 2 + torch.relu(e2) ** 2)
    psi_p = mat.sigy0 * lam_new + 0.5 * mat.hk * lam_new ** 2
    driving = psi_plus + psi_p

    theta = 0.5 * torch.atan2(2.0 * t12, t11 - t22)
    ct, st = torch.cos(theta), torch.sin(theta)
    dep11 = dl1 * ct ** 2 + dl2 * st ** 2
    dep22 = dl1 * st ** 2 + dl2 * ct ** 2
    dep12 = (dl1 - dl2) * ct * st

    return {"W_inc": W_inc, "driving": driving, "lam_new": lam_new,
            "dep11": dep11, "dep22": dep22, "dep12": dep12}


def compute_energy(inp, u, v, alpha, mat, pstate, gf_int, lc_int):
    s11, s22, s12 = strains(inp.node_features[:, 0:2], u, v, inp.area_elem, inp.T_conn)
    alpha_elem = (alpha[inp.T_conn[:, 0]] + alpha[inp.T_conn[:, 1]] + alpha[inp.T_conn[:, 2]]) / 3.0
    g = (1.0 - alpha_elem) ** 2 + conf.kappa
    core = rankine_core(s11, s22, s12, pstate, mat)
    e_el = (inp.area_elem * g * core["W_inc"]).sum()
    dax, day = field_grads(inp.node_features[:, 0:2], alpha, inp.area_elem, inp.T_conn)
    e_d = (inp.area_elem * (gf_int / (2.0 * lc_int) * alpha_elem ** 2
                            + 0.5 * gf_int * lc_int * (dax ** 2 + day ** 2))).sum()
    return e_el, e_d


def commit_plastic_state(inp, u, v, mat, pstate_old):
    with torch.no_grad():
        s11, s22, s12 = strains(inp.node_features[:, 0:2], u, v, inp.area_elem, inp.T_conn)
        core = rankine_core(s11, s22, s12, pstate_old, mat)
        new = PlasticState(_empty=True)
        new.ep11 = (pstate_old.ep11 + core["dep11"]).detach()
        new.ep22 = (pstate_old.ep22 + core["dep22"]).detach()
        new.ep12 = (pstate_old.ep12 + core["dep12"]).detach()
        new.lam = core["lam_new"].detach()
    return new


def driving_from_fields(inp, u, v, mat, pstate):
    with torch.no_grad():
        s11, s22, s12 = strains(inp.node_features[:, 0:2], u, v, inp.area_elem, inp.T_conn)
        core = rankine_core(s11, s22, s12, pstate, mat)
        return core["driving"].detach()


def reaction_force(top_nodes, inp, u, v, alpha, mat, pstate, gf_int, lc_int, force_scale):
    with torch.enable_grad():
        u_req = u.detach().clone().requires_grad_(True)
        v_req = v.detach().clone().requires_grad_(True)
        e_el, _ = compute_energy(inp, u_req, v_req, alpha.detach(), mat, pstate, gf_int, lc_int)
        _, grad_v = torch.autograd.grad(e_el, (u_req, v_req))
        idx = torch.as_tensor(top_nodes, dtype=torch.long, device=grad_v.device)
        return float((grad_v[idx].sum() * force_scale).detach().cpu())


# ==========================================
# 7. Field computation (hard BCs via harmonic lift: bottom u=v=0, top u=0, v=vApp)
# ==========================================
class FieldComputation:
    """u = phi(1-phi)*out_u*dl + accumulated; v = [phi(1-phi)*out_v + phi]*dl + accumulated."""

    def __init__(self, net, phi, non_dim):
        self.net = net
        self.phi = phi
        self.bubble = phi * (1.0 - phi)
        self.non_dim = non_dim
        self.lmbda = torch.zeros((), device=phi.device)
        self.lmbda_delta = torch.zeros((), device=phi.device)

    def fieldCalculation(self, inp, world_pos, alpha_nodal):
        out = self.net(inp)
        out_u, out_v = out[:, 0], out[:, 1]
        dl = self.lmbda_delta
        x_ref = inp.node_features[:, 0]
        y_ref = inp.node_features[:, 1]
        u_inc = self.bubble * out_u * dl
        v_inc = (self.bubble * out_v + self.phi) * dl
        u = u_inc + (world_pos[:, 0] - x_ref) * self.non_dim
        v = v_inc + (world_pos[:, 1] - y_ref) * self.non_dim
        wx = x_ref + u / self.non_dim
        wy = y_ref + v / self.non_dim
        return u, v, alpha_nodal, wx, wy


def fit(inp, field_comp, world_pos, hist_alpha, mat, pstate, gf_int, lc_int,
        num_epochs, optimizer, kind, min_delta=None, label="",
        min_iters=0, window=20):
    loss_data = []
    t0 = time.time()

    def closure():
        optimizer.zero_grad()
        u, v, alpha, _, _ = field_comp.fieldCalculation(inp, world_pos, hist_alpha)
        e_el, e_d = compute_energy(inp, u, v, alpha, mat, pstate, gf_int, lc_int)
        total = e_el + e_d
        if (not torch.isfinite(total)) or total <= 0.0:
            if kind == "lbfgs":
                # Strong-Wolfe trial steps may give non-finite energy: return a large
                # penalty (zero gradient) so the line search backtracks
                return torch.full((), 1.0e6, device=u.device)
            raise FloatingPointError(f"invalid energy at {label}: {float(total):.3e}")
        loss = torch.log10(total)
        if conf.weight_decay != 0.0:
            reg = sum(torch.sum(p ** 2) for n, p in field_comp.net.named_parameters() if "weight" in n)
            loss = loss + conf.weight_decay * reg
        loss.backward()
        loss_data.append(loss.item())
        if conf.progress_every > 0 and len(loss_data) % conf.progress_every == 0:
            print(f"    [{label} {kind} it={len(loss_data):5d}] loss={loss.item():+.4e} "
                  f"t={time.time() - t0:.1f}s", flush=True)
        return loss

    if kind == "lbfgs":
        # Back up parameters each epoch; roll back and stop if they become non-finite
        backup = [p.detach().clone() for p in field_comp.net.parameters()]
        for _ in range(num_epochs):
            optimizer.step(closure)
            if any((not torch.isfinite(p).all()) for p in field_comp.net.parameters()):
                with torch.no_grad():
                    for p, b in zip(field_comp.net.parameters(), backup):
                        p.copy_(b)
                print(f"    [{label} lbfgs] non-finite params, rolled back; switch to Rprop")
                break
            backup = [p.detach().clone() for p in field_comp.net.parameters()]
        return loss_data
    eps = np.finfo(float).eps
    counter, loss_prev = 0, 0.0
    for it in range(1, num_epochs + 1):
        closure()
        optimizer.step()
        curr = loss_data[-1]
        delta = abs(curr - loss_prev) / (abs(loss_prev) + eps)
        if min_delta is not None and it >= min_iters and delta <= min_delta:
            counter += 1
            if counter >= window:
                break
        else:
            counter = 0
        loss_prev = curr
    return loss_data


# ==========================================
# 8. Graph data
# ==========================================
def mesh_edges_from_triangles(tris, num_nodes):
    edges = np.concatenate([tris[:, 0:2], tris[:, 1:3],
                            np.stack([tris[:, 2], tris[:, 0]], axis=1)], axis=0)
    reverse = np.stack([edges[:, 1], edges[:, 0]], axis=1)
    self_loops = np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=1)
    return np.unique(np.concatenate([edges, reverse, self_loops], axis=0), axis=0).astype(np.int64)


def build_graph_data(nodes_norm, conn, area_norm, edges, bottom, top, device):
    n = nodes_norm.shape[0]
    node_type = torch.zeros((n, 2), dtype=torch.float32, device=device)
    disp_node = torch.zeros((n, 2), dtype=torch.float32, device=device)
    b = torch.from_numpy(bottom).to(torch.long).to(device)
    t = torch.from_numpy(top).to(torch.long).to(device)
    node_type[b, 0] = 1.0
    node_type[t, 1] = 1.0
    disp_node[t, 1] = 1.0
    return {
        "mesh_pos": nodes_norm,
        "T_conn": torch.from_numpy(conn).to(torch.long).to(device),
        "area_T": torch.from_numpy(area_norm).to(torch.float32).to(device),
        "edges": torch.from_numpy(edges).to(torch.long).to(device),
        "node_type": node_type,
        "disp_node": disp_node,
    }


def build_graph(graph_data, world_pos, hist_alpha, field_comp):
    node_features = torch.cat([
        graph_data["mesh_pos"],
        world_pos,
        hist_alpha.reshape(-1, 1),
        graph_data["node_type"],
        field_comp.lmbda_delta * graph_data["disp_node"],
    ], dim=-1).to(torch.float32)
    edges = graph_data["edges"]
    senders, receivers = edges[:, 0], edges[:, 1]
    edge_vec = graph_data["mesh_pos"][receivers] - graph_data["mesh_pos"][senders]
    edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
    edge_features = torch.cat([edge_vec, edge_len], dim=-1).to(torch.float32)
    return MultiGraph(node_features, EdgeSet(senders, receivers),
                      graph_data["T_conn"], graph_data["area_T"], edge_features)


# ==========================================
# 9. Staggered solve
# ==========================================
def _run_staggered_solve(step, label, field_comp, graph_data, world_pos, hist_alpha, h_elem,
                         pstate, mat, net, opt_r, damage_solver, gf_int, lc_int,
                         top_nodes, force_scale):
    n_iters = 0
    fp_iters = 0
    alpha_committed = hist_alpha.detach().clone()
    alpha_committed_np = alpha_committed.cpu().numpy().astype(np.float64)
    alpha_prev_np = alpha_committed_np.copy()
    h_committed = h_elem.detach().clone()
    F_prev = None
    force_stall = 0
    stagger_residual = float("inf")
    stagger_converged = False
    device, dtype = hist_alpha.device, hist_alpha.dtype
    active = float(hist_alpha.max()) >= conf.damage_active_alpha
    max_st = conf.max_stagger_active if active else conf.max_stagger
    min_st = conf.min_stagger_active if active else 1
    tol = conf.stagger_abs_tol_active if active else conf.stagger_abs_tol

    for fp in range(1, max_st + 1):
        fp_iters = fp
        inp = build_graph(graph_data, world_pos, hist_alpha, field_comp)
        fp_label = f"{label} fp={fp:02d}"
        use_lbfgs = (step == 1 and fp == 1 and conf.n_epochs_lbfgs > 0) or \
                    (conf.n_epochs_lbfgs > 0 and float(hist_alpha.max()) < conf.lbfgs_alpha_max)
        if use_lbfgs:
            opt_l = get_optimizer(net.parameters(), "LBFGS")
            n_iters += len(fit(inp, field_comp, world_pos, hist_alpha, mat, pstate, gf_int, lc_int,
                               conf.n_epochs_lbfgs, opt_l, "lbfgs", label=fp_label))
        # fp=1 carries the load increment (full budget); fp>=2 is warm-started (reduced budget, looser stop)
        if fp == 1:
            n_ep = conf.n_epochs_rprop_active if active else conf.n_epochs_rprop
            rel_tol, win, mi = conf.optim_rel_tol, conf.optim_stable_window, conf.optim_min_iters
        else:
            n_ep = conf.n_epochs_rprop_later
            rel_tol, win, mi = (conf.optim_rel_tol_later, conf.optim_stable_window_later,
                                conf.optim_min_iters_later)
        n_iters += len(fit(inp, field_comp, world_pos, hist_alpha, mat, pstate, gf_int, lc_int,
                           n_ep, opt_r, "rprop", min_delta=rel_tol, label=fp_label,
                           min_iters=mi, window=win))

        with torch.no_grad():
            u_c, v_c, _, _, _ = field_comp.fieldCalculation(inp, world_pos, hist_alpha)
        h_trial = torch.maximum(
            h_committed, driving_from_fields(inp, u_c, v_c, mat, pstate))
        alpha_np = damage_solver.solve(
            h_trial.detach().cpu().numpy().astype(np.float64), alpha_committed_np)
        hist_alpha = torch.from_numpy(alpha_np).to(dtype).to(device)

        stagger_residual = float(np.max(np.abs(alpha_np - alpha_prev_np)))
        if fp >= min_st:
            if stagger_residual < tol:
                stagger_converged = True
                break
            # Reaction stagnation is diagnostic only; convergence uses the damage residual
            if stagger_residual < conf.stagger_alpha_slack * tol:
                F_now = reaction_force(top_nodes, inp, u_c, v_c, hist_alpha, mat, pstate,
                                       gf_int, lc_int, force_scale)
                if F_prev is not None and abs(F_now - F_prev) < conf.stagger_force_rtol * max(abs(F_now), 1e-12):
                    force_stall += 1
                else:
                    force_stall = 0
                F_prev = F_now
            else:
                F_prev = None
                force_stall = 0
        alpha_prev_np = alpha_np

    if not stagger_converged:
        print(f"    WARNING: {label} reached max_stagger={max_st} without convergence "
              f"(alpha_residual={stagger_residual:.3e}, tol={tol:.3e})", flush=True)

    # Polish u with damage fixed; if driving force changes, update trial damage and repeat
    alpha_consistency = float("inf")
    max_final_corrections = 6
    for correction in range(1, max_final_corrections + 1):
        inp = build_graph(graph_data, world_pos, hist_alpha, field_comp)
        if conf.n_epochs_polish > 0:
            n_iters += len(fit(inp, field_comp, world_pos, hist_alpha, mat, pstate, gf_int, lc_int,
                               conf.n_epochs_polish, opt_r, "rprop",
                               min_delta=conf.polish_rel_tol,
                               label=f"{label} polish={correction}",
                               min_iters=conf.optim_min_iters_later,
                               window=conf.optim_stable_window_later))
        u, v, alpha, wx, wy = field_comp.fieldCalculation(inp, world_pos, hist_alpha)
        h_final = torch.maximum(
            h_committed, driving_from_fields(inp, u.detach(), v.detach(), mat, pstate))
        alpha_check = damage_solver.solve(
            h_final.detach().cpu().numpy().astype(np.float64), alpha_committed_np)
        alpha_consistency = float(
            np.max(np.abs(alpha_check - hist_alpha.detach().cpu().numpy())))
        if alpha_consistency < tol:
            break
        if correction < max_final_corrections:
            hist_alpha = torch.from_numpy(alpha_check).to(dtype).to(device)
    else:
        stagger_converged = False
        print(f"    WARNING: {label} final alpha consistency residual "
              f"{alpha_consistency:.3e} >= {tol:.3e}", flush=True)
    stagger_residual = max(stagger_residual, alpha_consistency)
    reaction = reaction_force(top_nodes, inp, u, v, alpha, mat, pstate, gf_int, lc_int, force_scale)
    with torch.no_grad():
        world_pos = torch.stack([wx, wy], dim=1).detach()
        alpha_max = float(alpha.max())
    h_elem = h_final.detach()
    pstate = commit_plastic_state(inp, u.detach(), v.detach(), mat, pstate)
    return (reaction, alpha_max, u, v, alpha, fp_iters, n_iters,
            world_pos, hist_alpha, h_elem, pstate, stagger_residual,
            stagger_converged)


# ==========================================
# 10. Single case
# ==========================================
def run_single_case(offset, lc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    sigy0 = calibrate_sigy0(lc)
    print(f"\n{'=' * 74}")
    print(f"  Fang2020 DENT (DEM hybrid ChebyKAN) | offset={offset} mm, lc={lc} mm (lc*={lc_star():.2f})")
    print(f"  Device: {conf.device}")
    print(f"  E={conf.mat_E} MPa, nu={conf.mat_nu}, gf={conf.gf} N/mm, sigt={conf.sigt} MPa")
    print(f"  sigy0={sigy0:.4f} MPa ({sigy0 / conf.sigt:.3f} sigt), Rankine perfect plasticity, AT2, plane stress")
    print(f"  Loading: u_max={conf.u_max} mm, {conf.n_steps} steps -> {output_dir}")
    print(f"{'=' * 74}")

    nodes_np, conn, areas = generate_dent_mesh(output_dir, offset, lc)
    device = torch.device(conf.device)
    dtype = torch.float32

    Ls = conf.H                                    # 120 mm
    non_dim = math.sqrt(conf.mat_E * lc / conf.gf)
    stress_scale = conf.mat_E / non_dim
    lc_int = lc / Ls
    gf_int = conf.gf * non_dim / (stress_scale * Ls)
    assert abs(gf_int - lc_int) < 1e-10 * lc_int
    sigy0_int = sigy0 / stress_scale
    print(f"  non_dim={non_dim:.4g}, stress_scale={stress_scale:.4g} MPa, "
          f"gf_int={gf_int:.4g}, lc_int={lc_int:.4g}, sigy0_int={sigy0_int:.4g}")

    nodes_norm_np = nodes_np / Ls
    area_norm = areas / Ls ** 2
    edges = mesh_edges_from_triangles(conn, len(nodes_np))
    bottom, top = boundary_node_sets(nodes_np)
    print(f"  BC: bottom u=v=0 ({len(bottom)} nodes), top u=0 v=vApp ({len(top)} nodes)")

    assembler = P1Assembler(nodes_norm_np, conn, area_norm)
    phi_np = solve_harmonic_lift(assembler, bottom, top)
    damage_solver = MeshDamageSolver(assembler, gf_int, lc_int)
    print(f"  harmonic lift: phi in [{phi_np.min():.3f}, {phi_np.max():.3f}], "
          f"damage solver ready ({conn.shape[0]:,} elems)")

    mat = Mat(device, dtype, sigy0_int)
    net = GraphNeuralNetwork(output_dimension=2).to(device)
    if conf.compile_net and hasattr(torch, "compile"):
        try:
            net = torch.compile(net)
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile unavailable ({e}); running eager")
    phi = torch.from_numpy(phi_np).to(dtype).to(device)
    field_comp = FieldComputation(net, phi, non_dim)

    nodes_norm = torch.from_numpy(nodes_norm_np).to(dtype).to(device)
    graph_data = build_graph_data(nodes_norm, conn, area_norm, edges, bottom, top, device)
    world_pos = graph_data["mesh_pos"].clone().detach()
    hist_alpha = torch.zeros(len(nodes_np), dtype=dtype, device=device)
    h_elem = torch.zeros(conn.shape[0], dtype=dtype, device=device)
    pstate = PlasticState(conn.shape[0], device, dtype)

    force_scale = stress_scale * Ls

    physical_disp = np.linspace(0.0, conf.u_max, conf.n_steps + 1)[1:]
    internal_all = np.concatenate(([0.0], physical_disp / Ls * non_dim))
    internal_deltas = np.diff(internal_all)

    triang = tri.Triangulation(nodes_np[:, 0], nodes_np[:, 1], conn)
    data_log = []
    opt_r = get_optimizer(net.parameters(), "RPROP")
    total_start = time.time()
    peak_kN = 0.0
    stop_after = None

    print("\n>>> Main loop (hybrid: net solves u, mesh solves d)...")
    print("  step | u(mm)  | F(kN)   | d_max  | lam_max  | fp | iters | t(s)")

    for step, (u_top, du_int) in enumerate(zip(physical_disp, internal_deltas), 1):
        t_step = time.time()
        field_comp.lmbda = torch.tensor(float(internal_all[step - 1]), dtype=dtype, device=device)
        field_comp.lmbda_delta = torch.tensor(float(du_int), dtype=dtype, device=device)
        label = f"step={step:3d}"

        (reaction, alpha_max, u, v, alpha, fp_iters, n_iters,
         world_pos, hist_alpha, h_elem, pstate, stagger_residual,
         stagger_converged) = _run_staggered_solve(
            step, label, field_comp, graph_data, world_pos, hist_alpha, h_elem,
            pstate, mat, net, opt_r, damage_solver, gf_int, lc_int, top, force_scale)

        F_kN = reaction * conf.thickness / 1000.0
        lam_max = float(pstate.lam.max()) / non_dim
        elapsed = time.time() - t_step
        data_log.append([step, u_top, F_kN, alpha_max, lam_max, fp_iters, n_iters,
                         stagger_residual, int(stagger_converged)])
        print(f"  {step:4d} | {u_top:.4f} | {F_kN:7.4f} | {alpha_max:.4f} | "
              f"{lam_max:.4e} | {fp_iters:2d} | {n_iters:5d} | {elapsed:.1f}")

        peak_kN = max(peak_kN, F_kN)
        softened = (peak_kN > 0 and F_kN < conf.early_stop_force_frac * peak_kN
                    and alpha_max >= conf.early_stop_min_alpha)
        if stop_after is None and softened:
            stop_after = step + 3
            print(f"    early stop armed at step {step} (F={F_kN:.3f} < "
                  f"{conf.early_stop_force_frac:.0%} peak {peak_kN:.3f})")
        is_last = (step == conf.n_steps) or (stop_after is not None and step >= stop_after)

        stride = conf.save_every_high if alpha_max >= conf.damage_high_threshold else conf.save_every_low
        if ((step - 1) % stride == 0) or is_last:
            np.savez(os.path.join(output_dir, f"step_{step:03d}.npz"),
                     u=u.detach().cpu().numpy(), v=v.detach().cpu().numpy(),
                     alpha=hist_alpha.cpu().numpy(),
                     lam=(pstate.lam / non_dim).cpu().numpy())
            fig, ax = plt.subplots(figsize=(5, 9))
            tcf = ax.tripcolor(triang, hist_alpha.cpu().numpy(), shading="gouraud",
                               cmap="jet", vmin=0, vmax=1)
            plt.colorbar(tcf, ax=ax, label="damage d")
            ax.set_aspect("equal")
            ax.set_title(f"off={offset} lc={lc}: step {step}, u={u_top:.4f}mm, "
                         f"F={F_kN:.3f}kN, d_max={alpha_max:.3f}")
            plt.savefig(os.path.join(output_dir, f"frame_{step:03d}.png"), dpi=110, bbox_inches="tight")
            plt.close()

        if is_last and step < conf.n_steps:
            print(f"  Early stop at step {step}.")
            break

    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        csv.writer(f).writerows(
            [["step", "u_mm", "F_kN", "d_max", "lam_max", "fp", "iters",
              "stagger_residual", "stagger_converged"]] + data_log)
    torch.save(net.state_dict(), os.path.join(output_dir, "model.pth"))

    us = [r[1] for r in data_log]
    fs = [r[2] for r in data_log]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(us, fs, "-", lw=2)
    ax.set_xlabel("u (mm)")
    ax.set_ylabel("F (kN)")
    ax.set_title(f"DENT DEM off={offset} lc={lc} (peak {max(fs):.3f} kN)")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(output_dir, "F_u_curve.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(output_dir, "DONE"), "w") as f:
        f.write(f"offset={offset} lc={lc} peak_kN={max(fs):.4f} steps={len(data_log)}\n")
    print(f"\n  Done off={offset} lc={lc}: peak={max(fs):.3f} kN, total={time.time() - total_start:.0f}s")


# ==========================================
# 11. Sweep driver
# ==========================================
def run():
    if "--device" in sys.argv:
        conf.device = sys.argv[sys.argv.index("--device") + 1]
        print(f"  Device override: {conf.device}")
    if "--compile" in sys.argv:
        conf.compile_net = True
    os.makedirs(conf.output_base, exist_ok=True)

    for offset, lc in conf.cases:
        od = os.path.join(conf.output_base, f"off{offset:g}_lc{lc:g}")
        if os.path.exists(os.path.join(od, "DONE")):
            print(f"[skip] offset={offset:g} lc={lc:g} done -> {od}")
            continue
        run_single_case(offset, lc, od)

    # Combined plot 1: one figure per offset, comparing lc
    for offset in sorted({off for off, _ in conf.cases}):
        fig, ax = plt.subplots(figsize=(8, 6))
        found = False
        for off, lc in conf.cases:
            if off != offset:
                continue
            p = os.path.join(conf.output_base, f"off{off:g}_lc{lc:g}", "results.csv")
            if not os.path.exists(p):
                continue
            rows = list(csv.reader(open(p)))[1:]
            us = [float(r[1]) for r in rows]
            fs = [float(r[2]) for r in rows]
            ax.plot(us, fs, "-", lw=2, label=f"lc = {lc:g} mm")
            found = True
        if found:
            ax.set_xlabel("u (mm)")
            ax.set_ylabel("F (kN)")
            ax.set_title(f"Fang2020 DENT (DEM) offset={offset:g} mm")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.savefig(os.path.join(conf.output_base, f"F_u_offset{offset:g}.png"),
                        dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Combined plot 2: offset sweep at lc=3
    fig, ax = plt.subplots(figsize=(8, 6))
    found = False
    for off, lc in sorted(conf.cases):
        if lc != 3.0:
            continue
        p = os.path.join(conf.output_base, f"off{off:g}_lc{lc:g}", "results.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.reader(open(p)))[1:]
        us = [float(r[1]) for r in rows]
        fs = [float(r[2]) for r in rows]
        ax.plot(us, fs, "-", lw=2, label=f"offset = {off:g} mm")
        found = True
    if found:
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("F (kN)")
        ax.set_title("Fang2020 DENT (DEM) offset scan, lc = 3 mm")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(conf.output_base, "F_u_offset_scan_lc3.png"),
                    dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined plots -> {conf.output_base}/")


if __name__ == "__main__":
    run()

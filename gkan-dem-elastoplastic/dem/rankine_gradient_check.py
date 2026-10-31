"""Branch-wise check of the Rankine incremental-potential gradient against an independent return mapping."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from dem_fang2020_dent import Mat, PlasticState, rankine_core


DTYPE = torch.float64
SIGY = 0.1
N_RANDOM = 100
SEED = 20260904


def rotate_principal(e1: float, e2: float, theta: float) -> tuple[float, float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return e1 * c * c + e2 * s * s, e1 * s * s + e2 * c * c, (e1 - e2) * c * s


def strain_from_principal_stress(s1: float, s2: float, nu: float) -> tuple[float, float]:
    return s1 - nu * s2, s2 - nu * s1


def independent_return(e: np.ndarray, nu: float, hk: float) -> tuple[np.ndarray, str]:
    c11 = 1.0 / (1.0 - nu * nu)
    c12 = nu * c11
    mu = 1.0 / (2.0 * (1.0 + nu))
    trial = np.array(
        [[c11 * e[0] + c12 * e[1], 2.0 * mu * e[2]],
         [2.0 * mu * e[2], c12 * e[0] + c11 * e[1]]],
        dtype=float,
    )
    vals, vecs = np.linalg.eigh(trial)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    s1t, s2t = vals
    if s1t <= SIGY:
        returned = vals
        branch = "elastic"
    else:
        a = c11 + hk
        dl1_s = (s1t - SIGY) / a
        s2n_s = s2t - c12 * dl1_s
        if s2n_s <= SIGY + hk * dl1_s:
            returned = np.array([s1t - c11 * dl1_s, s2n_s])
            branch = "single"
        else:
            bh = c12 + hk
            det = a * a - bh * bh
            f = np.array([s1t - SIGY, s2t - SIGY])
            dl = np.array([(a * f[0] - bh * f[1]) / det,
                           (a * f[1] - bh * f[0]) / det])
            dl = np.maximum(dl, 0.0)
            returned = np.array(
                [s1t - c11 * dl[0] - c12 * dl[1],
                 s2t - c12 * dl[0] - c11 * dl[1]]
            )
            branch = "corner"
    stress = vecs @ np.diag(returned) @ vecs.T
    return np.array([stress[0, 0], stress[1, 1], 2.0 * stress[0, 1]]), branch


def build_samples(nu: float, hk: float) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    samples: list[tuple[str, np.ndarray]] = []
    for _ in range(N_RANDOM):
        th = rng.uniform(-math.pi, math.pi)
        samples.append(("elastic", np.array(rotate_principal(
            rng.uniform(-0.04, 0.06), rng.uniform(-0.04, 0.04), th))))
        samples.append(("single", np.array(rotate_principal(
            rng.uniform(0.13, 0.25), rng.uniform(-0.05, 0.03), th))))
        samples.append(("corner", np.array(rotate_principal(
            rng.uniform(0.15, 0.25), rng.uniform(0.13, 0.22), th))))

    for sign in (-1.0, 1.0):
        for delta in np.logspace(-10, -4, 20):
            e1, e2 = strain_from_principal_stress(SIGY * (1.0 + sign * delta), 0.0, nu)
            samples.append(("near_elastic_single", np.array(
                rotate_principal(e1, e2, rng.uniform(-math.pi, math.pi)))))

            s1 = 1.35 * SIGY
            dl1 = (s1 - SIGY) / (1.0 / (1.0 - nu * nu) + hk)
            boundary_s2 = SIGY + (nu / (1.0 - nu * nu) + hk) * dl1
            e1, e2 = strain_from_principal_stress(
                s1, boundary_s2 + sign * delta * SIGY, nu)
            samples.append(("near_single_corner", np.array(
                rotate_principal(e1, e2, rng.uniform(-math.pi, math.pi)))))

    for value in (-0.02, 0.05, 0.10, 0.20):
        samples.append(("repeated_eigenvalue", np.array([value, value, 0.0])))
    return samples


def main() -> None:
    out_root = Path(os.environ.get("DEM_REVISION_OUTPUT_ROOT", "outputs"))
    out_dir = out_root / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = Mat("cpu", DTYPE, SIGY)
    nu, hk = float(mat.nu), float(mat.hk)
    samples = build_samples(nu, hk)
    e = torch.tensor(np.stack([x[1] for x in samples]), dtype=DTYPE, requires_grad=True)
    state = PlasticState(len(samples), "cpu", DTYPE)
    core = rankine_core(e[:, 0], e[:, 1], e[:, 2], state, mat)
    grad = torch.autograd.grad(core["W_inc"].sum(), e)[0].detach().numpy()

    rows = []
    for i, ((requested, strain), ad) in enumerate(zip(samples, grad)):
        ref, actual = independent_return(strain, nu, hk)
        abs_err = float(np.max(np.abs(ad - ref)))
        rel_err = abs_err / max(float(np.max(np.abs(ref))), 1.0e-14)
        rows.append({
            "sample": i,
            "requested_group": requested,
            "actual_branch": actual,
            "e11": strain[0],
            "e22": strain[1],
            "e12": strain[2],
            "ad_s11": ad[0],
            "ad_s22": ad[1],
            "ad_2s12": ad[2],
            "ref_s11": ref[0],
            "ref_s22": ref[1],
            "ref_2s12": ref[2],
            "max_abs_error": abs_err,
            "max_rel_error": rel_err,
        })

    csv_path = out_dir / "rankine_gradient_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": len(rows),
        "dtype": "float64",
        "max_abs_error": max(r["max_abs_error"] for r in rows),
        "max_rel_error": max(r["max_rel_error"] for r in rows),
        "rms_abs_error": float(np.sqrt(np.mean(
            [r["max_abs_error"] ** 2 for r in rows]))),
        "actual_branch_counts": {
            branch: sum(r["actual_branch"] == branch for r in rows)
            for branch in ("elastic", "single", "corner")
        },
        "groups": sorted({r["requested_group"] for r in rows}),
        "passed": all(np.isfinite(r["max_abs_error"]) for r in rows)
                  and max(r["max_abs_error"] for r in rows) < 1.0e-10,
    }
    summary_path = out_dir / "rankine_gradient_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("Rankine gradient verification failed")


if __name__ == "__main__":
    main()

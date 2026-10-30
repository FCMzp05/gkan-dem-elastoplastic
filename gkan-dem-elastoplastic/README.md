# gkan-dem-elastoplastic

Code for the paper

> P. Zhang, K. Tang. *A deep energy method for elasto-plastic phase-field fracture in quasi-brittle solids*. **Theoretical and Applied Fracture Mechanics** 149, 106004 (2027). https://doi.org/10.1016/j.tafmec.2026.106004

Length-scale-insensitive elasto-plastic phase-field model (AT2 with Rankine perfect plasticity in effective stress space). The displacement field is represented by a graph Kolmogorov-Arnold network (PyTorch); the damage field is solved on the mesh. Finite element reference solvers (Julia/Gridap) are provided for every benchmark.

## Contents

| Benchmark | GKAN-DEM (`dem/`) | Finite element reference (`fem/`) |
| --- | --- | --- |
| 1D homogeneous bar | `dem_fang2020_bar.py` | `bar_1d.jl` |
| 1D reduced-section bar | `dem_fang2020_bar_gamma.py` | `bar_gamma.jl` |
| Double-edge-notched tension | `dem_fang2020_dent.py` | `dent.jl` |
| L-shaped panel | `dem_fang2020_lplate.py` | `lplate.jl` |
| Cyclic three-point bending | `dem_fang2020_cyclic.py` | `cyclic_bending.jl` |

`dem/study_fang2020_dent_network.py`: network architecture, field-split and mesh studies (DENT).
`dem/rankine_gradient_check.py`: automatic-differentiation check of the Rankine incremental potential.

## Requirements

- Python 3.10+: `dem/requirements.txt`.
- Julia: `fem/Project.toml` and `fem/Manifest.toml`.

Physical and numerical parameters are set in each script (constants, command-line options or environment variables).

## Citation

```bibtex
@article{Zhang2027GKANElastoplastic,
  author  = {Zhang, Peng and Tang, Keke},
  title   = {A deep energy method for elasto-plastic phase-field fracture in quasi-brittle solids},
  journal = {Theoretical and Applied Fracture Mechanics},
  volume  = {149},
  pages   = {106004},
  year    = {2027},
  doi     = {10.1016/j.tafmec.2026.106004}
}
```

## License

MIT, see `LICENSE`.

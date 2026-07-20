# gkan-dem-elastoplastic

Code accompanying the paper *A Deep Energy Method for Elasto-Plastic Phase-Field Fracture in Quasi-Brittle Solids*.

The framework couples a length-scale-insensitive elasto-plastic phase-field model (AT2 + Rankine perfect plasticity in effective stress space, with analytical yield-stress calibration) with a hybrid graph Kolmogorov-Arnold network deep energy method (GKAN-DEM): the displacement field is represented by a graph neural network, while the damage field is solved on the mesh.

## Repository layout

| Folder | Content |
|---|---|
| `fem/` | Reference finite element solver (Julia, Gridap.jl) for all benchmarks |
| `dem/` | GKAN-DEM deep-learning solver (PyTorch) — to be released upon publication |

## FEM reference solver

Standalone Julia scripts, one per benchmark. Each script meshes the geometry with Gmsh, runs the staggered elasto-plastic phase-field solve over a sweep of regularization lengths, and writes force-displacement curves (CSV), damage snapshots (PNG) and full field states (VTK) to `outputs/`.

| Script | Benchmark |
|---|---|
| `fem/bar_1d.jl` | 1D homogeneous softening bar (strength recovery, length-scale insensitivity) |
| `fem/bar_gamma.jl` | 1D reduced-section bar (Gamma-convergence of the crack surface) |
| `fem/dent.jl` | Double-edge-notched tension specimen (notch offset and length-scale sweep) |
| `fem/lplate.jl` | L-shaped panel |
| `fem/cyclic_bending.jl` | Notched beam under cyclic three-point bending (half model) |

### Requirements

Julia >= 1.9 with [Gridap.jl](https://github.com/gridap/Gridap.jl), GridapGmsh.jl, Gmsh.jl and Plots.jl:

```julia
using Pkg
Pkg.activate("fem")
Pkg.instantiate()
```

### Running

```bash
cd fem
julia --project=. dent.jl
```

The 2D scripts sweep all configurations of the paper and support restart: completed cases (marked by a `DONE` file) are skipped on re-run.

## DEM solver

The GKAN-DEM solver (PyTorch) will be released in `dem/` upon publication of the article.

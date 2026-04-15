# Legacy code

Frozen code from earlier iterations of ISOKANN. **Not loaded by the package** and not maintained — kept as a reference corpus for ideas, lost APIs, and experimental variants. Expect bitrot.

If something here is useful, fish it out into `src/` and revive it; don't `include` from `legacy/`.

## Top-level files

| File | Topic |
|---|---|
| [iso1.jl](iso1.jl) | Original `IsoRun` — three-nested-loop ISOKANN with adaptive stratified χ-subsampling. Superseded by [`src/iso.jl`](../src/iso.jl) `Iso`. |
| [isosimple.jl](isosimple.jl) | A pared-down rewrite of the top-level API over `koopman`/`shiftscale`/`train_step!`. |
| [extrapolate.jl](extrapolate.jl) | `addextrapolates!` — sample new starting points by extrapolating χ beyond current extrema. |
| [ghostdata.jl](ghostdata.jl) | `data_from_trajectory_ghosted` — trajectory-aligned Koopman data with self-references on boundary points. |
| [loggers.jl](loggers.jl) | Older training-loss / plotting / autosave loggers (`TrainlossLogger7`, etc.). |
| [benchmarks.jl](benchmarks.jl) | `isobenchmarks` — parameter sweeps over `IsoRun`. Depends on `iso1.jl`. |
| [dataloader.jl](dataloader.jl) | Adapter letting old ISOKANN run with `MLUtils.DataLoader` (needs `(d, nk, n)` permutation). |
| [mmaped.jl](mmaped.jl) | `MMapped` array wrapper for on-disk training data. |
| [fluxtiming.jl](fluxtiming.jl) | Micro-benchmark of Flux/Optimisers/Zygote training step timings. |
| [testloss.jl](testloss.jl) | `data_vs_loss` — plot loss as a function of accumulated training data. |
| [vectorneurons.jl](vectorneurons.jl) | Equivariance/grad test stub for a `VecNeur` layer. |
| [xinet.jl](xinet.jl) | Mock-up of the ξ-net loss (double integral over χ level sets via KDE). |
| [tda.jl](tda.jl) | Kernel two-sample test + topological-data-analysis sketches for χ analysis. |
| [precompile.jl](precompile.jl) | `SnoopPrecompile` block targeting `IsoRun` — obsolete since `IsoRun` moved here. |

## Subfolders

### [`IsoMu/`](IsoMu/) — membership-function ISOKANN

Old module for the μ-opioid receptor reaction-path paper. Ingests DCD/PDB trajectory pairs via a `DataLink` and trains a χ map. Referenced by `scripts/papersuru.jl`.

- [IsoMu.jl](IsoMu/IsoMu.jl) — module entry point
- [datalink.jl](IsoMu/datalink.jl) — `DataLink` directory wrapper
- [dataloader.jl](IsoMu/dataloader.jl) — lazy/streaming Chemfiles trajectory loading
- [imu.jl](IsoMu/imu.jl) — `IMu` = ISOKANN + DataLink, reactive-path computation

### [`forced/`](forced/) — controlled ISOKANN

Early prototype combining ISOKANN with optimal-control biasing (Sikorski/Ribera-Borrell line of work that later matured into the Girsanov importance-sampling path). Self-contained `IsoForce` module.

- [IsoForce.jl](forced/IsoForce.jl) — module entry
- [isokann.jl](forced/isokann.jl) — `isokann(...)` entry point with `usecontrol`/`resample=:humboldt`
- [control.jl](forced/control.jl) — controlled drift, cost functionals
- [humboldtsample.jl](forced/humboldtsample.jl) — equi-χ adaptive sampling
- [utils.jl](forced/utils.jl) — Flux/Lux neural-network helpers

### [`vgv/`](vgv/) — VGVAPG hexapeptide study

- [vgv.jl](vgv/vgv.jl) — scripted analysis of VGVAPG trajectories. Partially superseded by `scripts/vgvapg.jl`.

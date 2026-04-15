# Scripts

Unmaintained collection of example scripts, collaborator workspaces, and figure-producing snippets. **Expect bitrot** — paths are often absolute to a specific machine, APIs may have moved, and some scripts reference deprecated functions (`runadaptive!`, `trajectorydata_bursts`). Treat this folder as a *reference corpus* (read for ideas and patterns), not as runnable examples.

Filenames prefixed with a date (`YYMMDD`) mark the day the experiment was started. Per-collaborator folders bundle a Project.toml and data.

## Top-level scripts

### Systems & molecular examples
| File | Topic |
|---|---|
| [250930 multidim ala.jl](250930%20multidim%20ala.jl) | Multidim χ on alanine dipeptide with `TransformPinvHist`. |
| [250320 multidim villin.jl](250320%20multidim%20villin.jl) | Multidim χ on villin from pre-computed long trajectories. |
| [villin.jl](villin.jl) / [villin250122.jl](villin250122.jl) | Villin adaptive-sampling runs (superseded by the 250122 version). |
| [trpcage.jl](trpcage.jl) | Trp-cage `OpenMMSimulation` + adaptive run — minimal one-liner. |
| [vgvapg.jl](vgvapg.jl) | VGVAPG hexapeptide, implicit solvent, adaptive sampling. |
| [nfgail.jl](nfgail.jl) | NFGAIL with `trajectorydata_bursts` (**deprecated API**). |
| [trajectory multitraj.jl](multitraj.jl) | Multi-trajectory loading + localized pair distances for a large system. |
| [openmmtrajectory.jl](openmmtrajectory.jl) | End-to-end OpenMM trajectory → `SimulationData` → `Iso` demo. |
| [250415 amyl.jl](250415%20amyl.jl) | Amyloid-β setup — 9 lines, a stub. |
| [250723 maryam.jl](250723%20maryam.jl) | Custom `IsoSimulation` subtype for a gene-regulatory-network (SSA) example. |
| [worm.jl](worm.jl) | Protein–ligand "worm" system with chain-id handling. |
| [atefe.jl](atefe.jl) | μ-opioid receptor benchmark run. |
| [mor.jl](mor.jl) | μ-opioid receptor, low-level OpenMM system construction. |

### Methods & ideas
| File | Topic |
|---|---|
| [250327 lrpinv.jl](250327%20lrpinv.jl) | Left/right pseudoinverse transform experiments (paired with [lrpinv.ipynb](lrpinv.ipynb)). |
| [250402 testxy.jl](250402%20testxy.jl) | 2D Mueller–Brown eigenvector / SqRA playground. |
| [241101 gompertz.jl](241101%20gompertz.jl) | Gompertz example, feature engineering on 73 interacting atoms. |
| [papersuru.jl](papersuru.jl) | Membership-function paper reruns; uses the legacy `IsoMu` module. |
| [251106 psample.jl](251106%20psample.jl) | Iterative `picking` to maintain a spread-out walker ensemble. |
| [251020 triplewell.jl](251020%20triplewell.jl) | Triplewell hot/cold exploration comparison. |
| [specconvergence.jl](specconvergence.jl) / [specconv2.jl](specconv2.jl) | Spectral-convergence studies on Mueller–Brown via SqRA. |
| [mor.jl](mor.jl) | Hand-rolled OpenMM simulation building. |

### Girsanov & importance sampling
| File | Topic |
|---|---|
| [girsanov_default_system.jl](girsanov_default_system.jl) | Minimal Girsanov-biased simulation template. |
| [test_girsanov.jl](test_girsanov.jl) | Protein–ligand Girsanov test (6O0K receptor + ligand SDF). |
| [241112 benchmark girsanov.jl](241112%20benchmark%20girsanov.jl) | Timing `laggedtrajectory` with and without Girsanov bias. |

### Interop & plotting
| File | Topic |
|---|---|
| [250127 makie.jl](250127%20makie.jl) | Interactive χ scatter using Bonito + WGLMakie. |
| [250317 taifun.jl](250317%20taifun.jl) | Minimal walk-through: build a sim, load trajectory, inspect coords. |

### Utility / misc
| File | Topic |
|---|---|
| [brokenforces.py](brokenforces.py) | OpenMM Python snippet investigating force reporter inconsistency. |
| [center_of_mass.py](center_of_mass.py) | COM-related Python helper. |

## Subfolders

### [`241128 convergence/`](241128%20convergence/)
`subspace.jl` — test whether multiple ISOKANN runs span the same dominant subspace (SVD of stacked χ). Short probe, ~30 lines.

### [`mfht/`](mfht/)
**Mean first-hitting-time** paper figures. Uses SqRA eigenfunctions and ODE integration on Mueller–Brown to compare MFHT vs χ-derived estimates. Has its own `Project.toml` and pre-rendered PNGs (`fig1`…`fig6`).

### [`waterimpexp/`](waterimpexp/)
Alanine dipeptide **implicit vs explicit water** comparison, with Ramachandran PNGs for three runs of each.

### [`251126_carsten/`](251126_carsten/)
Committor-ISOKANN work with Carsten (self-contained, has a `Project.toml`).
- [`main.jl`](251126_carsten/main.jl) — driver using `dchidx`, `isotarget`, `phi`/`psi`.
- [`committor.jl`](251126_carsten/committor.jl) — committor linear system via preconditioned GMRES.
- [`muellerbrown.jl`](251126_carsten/muellerbrown.jl) — Mueller–Brown committor on a grid with core sets.
- [`gridded.jl`](251126_carsten/gridded.jl), [`api_adp_grid.jl`](251126_carsten/api_adp_grid.jl) — grid-based studies.
- `bugs.md` — running notes.

### [`260217_lenard/`](260217_lenard/)
Aβ amyloid classification study with Lenard (DrWatson layout, k-fold CV).
- [`run_Abeta.jl`](260217_lenard/run_Abeta.jl) — main run script.
- [`batch.jl`](260217_lenard/batch.jl) — batched parameter sweep.
- Large `data/` + `out/` directories are **gitignored**.

### [`251024 effdynprojectwluca/`](251024%20effdynprojectwluca/)
Effective-dynamics project with Luca — 3D toy potential (`LucasPotential1` in `luca.jl`), SqRA diffusion comparisons, PCCA plots, and a large parameter sweep under `runs/` (gitignored).

### [`archive/`](archive/)
Older still — adaptive-sampling prototypes, OpenMM↔Molly comparisons, a custom `@autokw` macro, SLURM batch runs, data-convergence study. Predates the dated-filename convention.

### [`data/`](data/)
Input PDBs / JLD2s for the scripts. **Gitignored** — populate locally as needed.

## Assets at root

`test*.pdb`, `aladip4.png`, `tmp.png`, `lrpinv.ipynb` are one-off outputs kept alongside their originating scripts. The Jupyter notebook pairs with `250327 lrpinv.jl`.

## Housekeeping notes

- `scripts/` is listed in [`.gitignore`](../.gitignore), so new files need `git add -f` to be tracked. The currently tracked 40+ files predate that rule.
- Several large subdirs (`data/`, `260217_lenard/`, `251024 effdynprojectwluca/`) are explicitly ignored on top of that.
- `.history.jl_` is a VS Code history dump; not worth reading.

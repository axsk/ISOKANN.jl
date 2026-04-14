# ISOKANN Release Notes

## v2.0 (Major Release — since v1.3, Oct 15 2025)

### 🔴 BREAKING CHANGES

#### Python Interop: PyCall → PythonCall
- Switched from PyCall.jl to PythonCall.jl for OpenMM bindings
- `CondaPkg.jl` now manages Python dependencies automatically
- OpenMM must be installed separately by user (no longer auto-managed)
- Python wrapper `mopenmm.py` refactored and cwd-independent

#### API Renames
- `Iso.transform` field → `Iso.target` (affects serialized models)
- `runadaptive!()` → `run_kde!()` (old name deprecated with warning)
- Default featurizer now outputs Float32 (was mixed Float32/Float64)

#### Dependency Management
- User is now responsible for managing Python dependencies
- `openmmforcefields` is optional (custom forcefields can be supplied)
- Reduces installation complexity and version conflicts

---

### ✨ MAJOR FEATURES

#### Multidimensional ISOKANN (new in v2.0)
- Support for learning **multiple slow reaction coordinates** (multidim χ)
- **New target transforms**:
  - `TransformISA` — Inner Simplex Algorithm (with optional whitening)
  - `TransformPseudoInv` — pseudoinverse-based approach
  - Experimental: `TransformSVD`, `TransformGramSchmidt`, `TransformLeftRight`, `TransformCross`
- Loss is **variance-weighted** across output dimensions
- Permutation stabilization via `fixperm()` and `Stabilize` wrapper
- Allows learning coupled slow modes in complex systems

#### Metadynamics Simulator
- New `MetadynamicsSimulation` type for enhanced sampling
- `deposit!()` function for bias deposition
- `run_bridges!()` and `adaptive_metadynamics()` workflows
- Well-tempered metadynamics support with empirically validated results
- GPU-compatible

#### Bridge Sampling & Effective Dynamics
- New `GuidedLangevinBridge` simulator for biased dynamics toward target RC values
- Time-dependent forces with Zygote autodiff
- `bridge_simplex` for simplicial bridge construction
- Enables evaluation of RC quality and effective dynamics
- GPU-compatible

#### New Analysis Tools
- `rates(iso)` — compute exit rates from learned RC (improved over old `exit_rates`)
- "Adaptive Dynamic Path Sampling on Grid" — efficient grid-based RC exploration
- `plot_potential()` — energy landscape visualization
- Enhanced molecular utilities with unified dihedral computation (phi, psi, theta, omega)

---

### 📈 IMPROVEMENTS & FIXES

#### Correctness & Stability
- **Training**: Fixed incomplete final minibatch causing high variance in loss
- **OpenMM**: Fixed sigma computation discrepancy (underdamped vs Brownian dynamics)
- **OpenMM**: `langevin_girsanov!()` no longer mutates input `x0`
- **Numerics**: `pairdists` clipping prevents numerical instability
- **Numerics**: `TransformISA` uses Float64 for matrix inverse (stability)

#### Performance
- Float32 default for all features (better GPU performance)
- `data_from_trajectories` now preallocates output arrays
- CUDA kernel for `sqpairdists` with autodiff support
- GPU support throughout (alignment, shortest-path, training, bridges, metadynamics)

#### Infrastructure
- New logging system with `FunctionLogger` for custom callbacks
- Fixed logger skipping on offset runs (resume training now works)
- `run_kde!()` improved resampling: density difference with periodic boundaries
- Molecular I/O: `load_trajectory`, `save_trajectory` now use PythonCall (PyCall-independent)
- `langevin_force()` accepts `reclaim` kwarg for memory management
- Brownian integrator support in OpenMM

#### API Improvements
- `langevin_girsanov!()` accepts `should_stop` callback
- `validationloss` fixed to use all data for shift-scale transform
- `rates()` extended to work with arbitrary `(x, y)` data
- Network architecture: `pairnet` now dispatches cleanly to `densenet` with auto-sizing
  - Featurizer layer support removed (simplifies the API)

---

### 📚 EARLIER FEATURES (from prior commits)

New features:
- support for particle-weighted alignment/rmsd
- `data_from_trajectory` now supports an additional `lag` parameter (on top of `stride`)
- `dchidfeat` returns derivative of the model wrt. the features
- `mutual_information(iso)` computes the mutual information between chi and features

Improvements and fixes:
- Make the `WGLMakie` dependency optional
- refactor utility code to `utils`
- simplified handling of external `SimulationData` through implicit use of `ExternalSimulation`
- more robust dispatch for `exit_rates`
- `ExternalSimulation` now stores parameters in a `Dict`
- `FeaturesPairs` now supports `mdtraj` selection syntax
- `propagate` has inbuild retry mechanic
- `reactive_path` now allows to distinguish between `coords` for the distance computations and `fullcoords` for the output, as well as supporting weights
- `shortestchain` now masks the distance computations, improving `reactive_path` compute time for tight `minjump`/`maxjump` bounds

---

## Known Issues / TODOs for v2.1+

- Feature dtype: `Float32` is intended but not consistently enforced everywhere (can silently produce Float64)
- `TransformISA` permutation: uses brute-force search, breaks for large `nout` (TODO: Hungarian algorithm)
- `simulationtime` doesn't work correctly with `cutoff` parameter
- Bridge sampling: `angdiff` (periodic RC distance) is always applied; should be conditional
- `isotarget.jl` line 337: `@show vals` debug output left in code
- Test suite: `runtests.jl` and `workflow.jl` have some stale function call references
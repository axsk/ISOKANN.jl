# Introduction

ISOKANN.jl learns slow reaction coordinates (χ, "chi") as eigenfunctions of the
Koopman operator from short trajectory data. This page covers the data model,
the training loop, and the handful of types you will use most.

## The data model

ISOKANN needs two kinds of samples:

- `xs`: a matrix of shape `(d, n)` — `n` starting points in a `d`-dimensional
  state or feature space.
- `ys`: a 3-D array of shape `(d, nk, n)` — for each starting point in `xs`,
  `nk` Koopman replicas obtained by propagating the dynamics with a common
  lag time.

These are used to estimate the action of the Koopman operator via a Monte-Carlo
approximation,

```math
[K\chi](x) = \mathbb{E}_{X_0 = x} [\chi(X_t)] \approx \frac{1}{N} \sum_{i=1}^{N} \chi(y_i).
```

The starting points can be sampled arbitrarily (they do not need to come from
the stationary distribution), but the learned χ can only represent the part of
the dynamics covered by `xs`. The `ys` must be propagated from the corresponding
`xs` under the system's dynamics with a common lag time.

## Minimal example

For raw `(xs, ys)` data you can construct an [`Iso`](@ref) directly:

```julia
using ISOKANN

xs = rand(3, 100)
ys = rand(3, 10, 100)
iso = Iso((xs, ys))
run!(iso, 100)   # 100 ISOKANN iterations
chis(iso)        # learned χ values
```

## Using a simulation

For adaptive sampling and for bookkeeping the link between coordinates and
network features, wrap the dynamics in a [`SimulationData`](@ref):

```julia
using ISOKANN

sim  = Doublewell()                        # 1D analytical potential
data = SimulationData(sim, 100, 10)        # 100 starting points, 10 Koopman samples each
iso  = Iso(data)
run!(iso, 100)
chis(iso)
```

Built-in simulations include [`Doublewell`](@ref), [`Triplewell`](@ref),
[`MuellerBrown`](@ref), [`Diffusion`](@ref) (general Langevin dynamics) and
[`OpenMMSimulation`](@ref) (molecular dynamics via OpenMM). You can also plug in
your own dynamics by implementing the `IsoSimulation` interface (at minimum
[`propagate`](@ref)) or provide externally-generated trajectories via
`ExternalSimulation` / `data_from_trajectory`.

## OpenMM quick start

```julia
using ISOKANN

sim  = OpenMMSimulation()                  # bundled alanine dipeptide
data = SimulationData(sim, 100, 5)
iso  = Iso(data)

run!(iso, 100)                             # or: run_kde!(iso; generations=5, iter=100)

plot_training(iso)
scatter_ramachandran(iso)
rates(iso)                                 # macro-state transition rates
save_reactive_path(iso, out="path.pdb")
```

## Classical ISOKANN on precomputed trajectories

If you already have trajectory data and just want to run ISOKANN on it (no new
sampling), turn your trajectories into `(xs, ys)` pairs with
[`data_from_trajectory`](@ref) (single trajectory) or
[`data_from_trajectories`](@ref) (many). Both support `stride`, `lag`, and
`reverse` (treat the trajectory as time-reversible, which doubles the data):

```julia
# single trajectory of shape (d, nframes)
data = SimulationData(sim, data_from_trajectory(xs; lag=10, stride=1))

# many trajectories of varying length
data = SimulationData(sim, data_from_trajectories([xs1, xs2, ...]; lag=10))

# or, if you have no `sim` and only raw arrays
data = SimulationData(xs, ys)
```

For OpenMM trajectories on disk, [`load_trajectory`](@ref) / `readchemfile`
return a `(d, nframes)` coordinate matrix ready to pass into
`data_from_trajectory`.

## What the pieces do

- [`Iso`](@ref) is the main training object. It holds the neural network, the
  optimizer, training data, the target transform, and loss/log history.
- [`SimulationData`](@ref) wraps an `IsoSimulation` together with its sampled
  `coords` and the derived `features` that are actually fed into the network.
  The mapping `coords → features` is defined by the simulation's featurizer
  (for `OpenMMSimulation` the default is pairwise distances, selectable between
  all atoms, locally close atoms, or only C-α atoms; see its docstring).
- Training is driven by [`run!`](@ref). Each iteration computes a target from
  `E[χ(ys)]` via a configurable transform (`TransformShiftscale` for 1-D χ,
  `TransformISA` / `TransformPseudoInv` for multi-dim χ) and takes a batch of
  gradient steps against `‖model(xs) − target‖²`.
- Adaptive sampling is available via [`run_kde!`](@ref), which interleaves
  KDE-based resampling along the current χ with training. (The older
  `runadaptive!` is a deprecated alias.)
- Results: [`chis`](@ref) returns the learned χ values, [`rates`](@ref) returns
  the macro-state transition rate matrix, [`save_reactive_path`](@ref) writes
  a PDB of the extracted reaction path.

## Accessing data

`SimulationData` lets you reach either view of the samples:

- [`coords`](@ref) / [`propcoords`](@ref) — raw `xs` / `ys` coordinates.
- [`features`](@ref) / [`propfeatures`](@ref) — featurized `xs` / `ys`
  actually passed to the network.

## Source layout

Core:

- `src/iso.jl` — the `Iso` type and the training loop (`run!`, `run_kde!`)
- `src/simulation.jl` — `SimulationData` and the `IsoSimulation` interface
- `src/data.jl` — low-level operations on the `(xs, ys)` tuple
- `src/isotarget.jl` — ISOKANN target transforms (1-D and multi-dim)
- `src/models.jl` — default networks and optimizers

Simulators:

- `src/simulators/langevin.jl` — analytical potentials (Doublewell, Triplewell,
  MuellerBrown) integrated via StochasticDiffEq
- `src/simulators/openmm.jl` — OpenMM wrapper (through PythonCall)
- `src/simulators/bridge.jl` — guided Langevin bridges toward target χ values
- `src/simulators/metadynamics.jl` — draft metadynamics implementation

Utilities:

- `src/molutils.jl` — alignment, dihedrals, and other molecular helpers
- `src/pairdists.jl` — pairwise-distance features
- `src/plots.jl` — training / χ plots
- `src/subsample.jl` — stratified and KDE-based subsampling
- `src/reactionpath.jl`, `src/reactionpath2.jl` — reaction-path extraction

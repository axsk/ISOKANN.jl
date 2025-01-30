# Introduction

This package provides the core ISOKANN algorithm as well as some wrappers and convenience routines to work with different kind of simulations and data.

The core ISOKANN algorithm is accessed by the `Iso` type,
which holds the neural network, optimizer, ISOKANN parameters and training data.

You can construct it by passing a tuple of `(xs, ys)` of arrays as input data. Here `xs` is a matrix where the columns are starting points of trajectories and `ys` is a 3 dimensional array where `ys[d,k,n]` is the `d`-th coordinate of the `k`-th Koopman-replica of the `n`-th trajectory.

To start training the neural network simply call the `run!` function passing the `Iso` object and the number of ISOKANN iterations.
The resulting \chi values can be obtained via the `chis` method

```julia
iso=Iso((rand(3,100), rand(3,10,100)))
run!(iso)
chis(iso)
```

For more advanced use, such as with the adaptive sampling algorithms we pass a `SimulationData` object instead of the data tuple to the `Iso` constructor.

The `SimulationData` itself is composed of a `Simulation`, its simulated trajectory data as well as the features fed into the neural network for training.
We supply some basic simulations which can generate the data, e.g. [`Doublewell`](@ref), [`MuellerBrown`](@ref), [`Diffusion`](@ref), [`MollySimulation`](@ref) and [`OpenMMSimulation`](@ref).
Of course you can write your own `Simulation` which in its most basic form needs to supply only the [`propagate`](@ref) method.

```julia
sim = Doublewell()
data = isodata(sim, 100, 20)
iso = Iso(data)
```

We also provide different type of wrappers to load simulations [`vgv`] or generate data from trajectories [`IsoMu`].

For an advanced example take a look at the `scripts/vgvadapt.jl` file.


# Components

The `OpenMMSimulation` is a good example for an `Simulation` object. It parametrises a system by specifying a molecular simulation by reading the molecular structure from a .pdb file but also the system temperature, the simulation lag time and other simulation parameters.

The `SimulationData` in turn links such a simulation `Simulation` to actual simulation data which is used by ISOKANN for training.
Through the specification of a `featurizer` the neural network does not need to digest the simulation coordinates but can use optimized features which for example guarantee invariance under rigid transformations.
By default the `featurizer` is inhereted from the default `featurizer` of the `Simulation`. For the `OpenMMSimulation` we have pre-implemented pairwise distances between all atoms, locally close atoms and/or the c-Alpha atoms (c.f. the `OpenMMSimulation` docstring).

The `Iso` object then brings together the `SimulationData` with a neural network `model` and an `optimizer`.
Its main use is together with the training routine `run!()` which computes the ISOKANN iteration via `isotarget` and updates the networks weights with `train_batch`.
The `logger` field allows to ammend other operations such as the default `autoplot()` which displays the progress during training.
The default model is the `pairnet` which constructs a fully connected network of a given number of layers of descreasing width and the default optimizer is Adam with weight decay.

Adaptive sampling is facilitated either by the `runadaptive!` method, or the individual `adddata!`, `resample_kde!`, `addextrapolates!` used in a custom training routine.


The learned chi values can be accessed via `chis(::Iso)` and the reaction rates via `exit_rates(::Iso)`


# Contents of the source files

Core:

- `simulation.jl`: handling of `SimulationData` which mainly dispatches to other lower-level functions
- `data.jl`: low level functions for accessing and manipulating the data tuple
- `iso2.jl`: main training routine
- `isotarget.jl`:  different ISOKANN iteration targets  for 1D and higher dimensional chi functions
- `models.jl`: convencience functions for the contruction/manipulation of `model` and `optimiser`

Simulators:

- `simulators/langevin.jl`: A simulator for the Langevin equation
- `simulators/openmm.jl`: Wrapper around OpenMM for molecular dynamics simulations

Utility:
- `molutils.jl`: different utilities to work with molecules and molecular data, such as alignment, dihedrals etc.
- `pairdists.jl`: different methods to compute pairwise distance features
- `plots.jl`: plotting functionality
- `subsample.jl`: stratified or KDE based uniform subsampling along a given reaction coordinate

Experimental:

- `extrapolate.jl`: generation of new sampling points by extrapolating on the neural network (dreaming)
- `bonito.jl` and `makie.jl`: live visualizations via Makie.jl/WebGL and the dashboard using the Bonito.jl webserver
- `reactionpath.jl`: reaction paths by integration on the neural network
- `reactionpath2.jl`: reactive path extraction from sampled data by solving shortest paths problems

# On the representation of the data

In order to estimate the chi functions ISOKANN requires two kinds of samples, starting points `xs` and propagated points `ys`.
These are used to estimate the action of the Koopman operator through a Monte-Carlo approximation, i.e.
`` [K\chi](x) = \mathbb{E}_{X_0 = x} [\chi(X_t)] \approx \frac{1}{N} \sum_{i=1}^N \chi(y_i).``

The starting points can be sampled without any restriction. In particular they do not need to be sampled from the stationary distribution.
Note however, that the learned chi function can only represent the part of the dynamics covered by `xs`. Furthermore, regions with more samples have higher contribution to the loss, i.e. will be resolved more precisely.

The `ys` samples however have to be propagated from their respective `xs` according to the processes dynamics and with a common lagtime amongst all samples.

Internally ISOKANN.jl handles the `xs` and `ys` as 2-dimensional resp. 3-dimensional Arrays.
In particular `xs` has the size `(ndim, nsamples)` and `ys` has size `(ndim, nkoop, nsamples)`, where `ndim` is the size of the state- or feature-space, `nkoop` the number of Koopman-/Batch-samples per starting point and `nsamples` is the number of total samples.

You can create an `Iso` object directly from raw data of this type, by passing in a tuple of `(xs, ys)` as data.

Alternatively, you can use the `SimulationData` object, whose main task is to keep together the original samples' coordinates as well as their corresponding features which are passed as arguments to the neural network representing `chi`.
You can access its coordinates or features (i.e. the above `xs`) through the methods `coords` or `features`, and the corresponding Koopman samples (the `ys`) through `propcoords` and `propfeatures`.
`SimulationData` futhermore allows to link the data to a `IsoSimulation` and provides futher methods to conveniently augment the data with new samples, merge data sets, sample adaptively etc.

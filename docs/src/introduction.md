# Introduction

This package provides the core ISOKANN algorithm as well as some wrappers and convenience routines to work with different kind of simulations and data.

The core ISOKANN algorithm is accessed by the `Iso2` type,
which holds the neural network, optimizer, ISOKANN parameters and training data.

You can construct it by passing a tuple of `(xs, ys)` of arrays as input data. Here `xs` is a matrix where the columns are starting points of trajectories and `ys` is a 3 dimensional array where `ys[d,k,n]` is the `d`-th coordinate of the `k`-th Koopman-replica of the `n`-th trajectory.

To start training the neural network simply call the `run!` function passing the `Iso2` object and the number of ISOKANN iterations.
The resulting \chi values can be obtained via the `chis` method

```julia
iso=Iso2((rand(3,100), rand(3,10,100)))
run!(iso)
chis(iso)
```

We also supply some basic simulations which can generate the data, e.g. [`Doublewell`](@ref), [`MuellerBrown`], [`Diffusion`], [`MollySimulation`] and [`OpenMMSimulation`].
You can use the [`isodata`] function to sample data for ISOKANN.

```julia
sim = Doublewell()
data = isodata(sim, 100, 20)
iso = Iso2(data)
```

We also provide different type of wrappers to load simulations [`vgv`] or generate data from trajectories [`IsoMu`].

Experimental support for adaptive sampling is provided by [`run(iso, sim; ny)`].



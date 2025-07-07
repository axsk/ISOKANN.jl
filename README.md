# ISOKANN

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://axsk.github.io/ISOKANN.jl/dev)

The ISOKANN.jl package implements the ISOKANN algorithm for the identification of macro-states of molecular systems. Its main features comprise of:
- A flexible implementation of the ISOKANN core (`Iso`) (supporting 1D and N-D ISOKANN, customizable neural networks on a broad set of `SimulationData`)
- A battery-included interfaces to OpenMM for automated adaptive sampling of molecular dynamics
- Different adaptive sampling strategies (extapolation, kde and stratified sampling)
- A posteriori analysis tools (plots, reaction path extraction and reaction rate estimation)

See the [documentation](https://axsk.github.io/ISOKANN.jl/dev) for details.

## Quick start

Install the package via `julia> ]add https://github.com/axsk/ISOKANN.jl`.

If you want to use Julia's built Conda.jl to automatically install OpenMM, you shoud build the package after setting the environment variable
`PYTHON=""`, e.g. through `ENV["PYTHON"]=""; using Pkg; Pkg.build()`.

The usual pipeline consists of the creation of system simulation, generation of training data, training ISOKANN and a posteriori analysis of the results.

```julia

using ISOKANN

# Define an OpenMMSimulation. The default molecule is the Alanine-Dipeptide.
sim = OpenMMSimulation()

# Sample the initial data for training of ISOKANN with 100 initial points and 5 koopman samples per point.
data = SimulationData(sim, 100, 5)

# create the ISOKANN training object
iso = Iso(data)

# train for 100 episodes
run!(iso, 100)

# plot the training losses and chi values
plot_training(iso)

# scatter plot of all initial points colored in corresponding chi value
scatter_ramachandran(iso)

# estimate the exit rates, i.e. the metastability
exit_rates(iso)

# extract the reactive path
save_reactive_path(iso, out="path.pdb")
```

More comprehensive usecase examples can be found in
- [`scripts/villin.jl`](scripts/villin.jl): simulating the folding of the chicken villin
- [`scripts/vgvapg.jl`](scripts/vgvapg.jl)
- [`scripts/trpcaeg.jl](scripts/trpcage.jl)
- [`scripts/multitraj.jl](scripts/multitraj.jl): Extraction of reaction paths from multiple long trajectories.

For further information on specific functions use Julias built-in help/docstring functionality, e.g. `?Iso`.

## References

- [Rabben, Ray, Weber (2018) - ISOKANN: Invariant subspaces of Koopman operators learned by a neural network.](https://doi.org/10.1063/5.0015132)
- [Sikorski, Ribera Borrell, Weber (2024) - Learning Koopman eigenfunctions of stochastic diffusions with optimal importance sampling and ISOKANN](http://dx.doi.org/10.1063/5.0140764)
- [Sikorski, Rabben, Chewle, Weber (2024) - Capturing the Macroscopic Behaviour of Molecular Dynamics with Membership Functions](http://arxiv.org/abs/2404.10523)

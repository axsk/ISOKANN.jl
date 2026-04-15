# ISOKANN.jl

Documentation for ISOKANN.jl.

Start with the [Introduction](introduction.md) for an overview of the data model
and training loop. The [Installation](installation.md) page walks through
getting Julia and OpenMM set up, and [Tips](tips.md) covers practical choices
like optimizer and regularization.

```@meta
CurrentModule = ISOKANN
```

## Main entry points

```@docs
Iso
SimulationData
OpenMMSimulation
propagate
run!
run_kde!
chis
rates
plot_training
scatter_ramachandran
save_reactive_path
```

## Data construction and manipulation

```@docs
data_from_trajectory
data_from_trajectories
mergedata
addcoords!
resample_kde!
laggedtrajectory
```

## Models and Optimizers
```@docs
pairnet
densenet
AdamRegularized
NesterovRegularized
```

## Public API

```@autodocs
Modules = [ISOKANN, ISOKANN.OpenMM]
Private = false
```

## Internal API

```@autodocs
Modules = [ISOKANN, ISOKANN.OpenMM]
Public = false
```

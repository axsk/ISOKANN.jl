# ISOKANN.jl

Documentation for ISOKANN.jl

```@meta
CurrentModule = ISOKANN
```

## Main entry points

```@docs
OpenMMSimulation
SimulationData
Iso
propagate
isodata
run!
plot_training
```

## Models and Optimizers
```@docs
defaultmodel
pairnet
densenet
smallnet
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

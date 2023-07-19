#__precompile__(false)

module ISOKANN

#using Startup           # precompiles most used packages

include("forced/IsoForce.jl")

using LinearAlgebra: norm
include("humboldtsample.jl")  # adaptive sampling

import ChainRulesCore
include("pairdists.jl")  # pair distances

import Flux
export pairnet, pairnetn
include("models.jl")     # the neural network models/architectures

using Molly: Molly, System
using Unitful
using StaticArrays
export PDB_5XER, PDB_6MRR, PDB_ACEMD
include("molly.jl")      # interface to work with Molly Systems

using Molly
export MollyLangevin, MollySDE, propagate, solve
include("simulation.jl") # Langeving dynamic simulator (MollySystem+Integrator)

using LinearAlgebra
using StatsBase: mean
include("molutils.jl")   # molecular utilities: dihedrals, rotation

import StatsBase, Zygote, Optimisers, Flux, JLD2
using ProgressMeter
using LsqFit
export IsoRun, run!, Adam, AdamRegularized
include("isomolly.jl")   # ISOKANN for Molly systems

using Plots
export plot_learning, scatter_ramachandran
include("plots.jl")      # visualizations

using JLD2
using StatsBase: sample
include("data.jl")       # tools for handling the data (sampling, slicing, ...)
include("performance.jl") # performance metric loggers
include("benchmarks.jl") # benchmark runs, deprecated by scripts/*

using CUDA
using Flux
using NNlib, NNlibCUDA
include("cuda.jl")       # fixes for cuda

using MLUtils
include("dataloader.jl")

using Zygote
using OrdinaryDiffEq
using SpecialFunctions: erf
export reactionpath
include("reactionpath.jl")

#include("precompile.jl") # precompile for faster ttx

end

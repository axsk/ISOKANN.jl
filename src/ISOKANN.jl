#__precompile__(false)

module ISOKANN

#using Startup           # precompiles most used packages

#include("forced/IsoForce.jl")

import Random

using LinearAlgebra: norm, dot, cross, diag, svd
using StatsBase: mean, sample, mean_and_std
using Molly: Molly, System
using StaticArrays: SVector
using StatsBase: sample, quantile
using CUDA: CuArray, CuMatrix, cu, CUDA
using NNlib: batched_adjoint, batched_mul
using Unitful: @u_str, unit
using SpecialFunctions: erf
using Plots: plot, plot!, scatter, scatter!
using MLUtils: numobs, getobs, shuffleobs, unsqueeze
using Flux: cpu, gpu


import Chemfiles
import ProgressMeter
import ChainRulesCore
import Flux
import StatsBase, Zygote, Optimisers, Flux, JLD2
import LsqFit
import JLD2
import Flux
import NNlib
import MLUtils
import Zygote
import OrdinaryDiffEq
import Graphs
import Optimisers

import MLUtils: numobs
import Flux: cpu, gpu

export pairnet#, pairnetn
export PDB_ACEMD, PDB_1UAO, PDB_diala_water
export MollyLangevin, propagate, solve#, MollySDE
export IsoRun, run!
export AdamRegularized, pairnet#, Adam
export plot_training, scatter_ramachandran
export reactionpath
export cpu, gpu
export iso2
export Iso2
export Doublewell, Triplewell, MuellerBrown




include("subsample.jl")  # adaptive sampling
include("pairdists.jl")       # pair distances
include("simulators/simulation.jl")      # Langevin dynamic simulator (MollySystem+Integrator)
include("models.jl")          # the neural network models/architectures
include("simulators/molly.jl")           # interface to work with Molly Systems
include("molutils.jl")        # molecular utilities: dihedrals, rotation
include("data.jl")            # tools for handling the data (sampling, slicing, ...)
include("iso1.jl")        # ISOKANN for Molly systems
include("plots.jl")           # visualizations
#include("loggers.jl")     # performance metric loggers
#include("benchmarks.jl")      # benchmark runs, deprecated by scripts/*
include("reactionpath.jl")

include("simulators/langevin.jl")  # for the simulators

include("isosimple.jl")
include("isotarget.jl")
include("iso2.jl")
include("simulators/potentials.jl")

include("simulators/openmm.jl")

import .OpenMM.OpenMMSimulation
export OpenMMSimulation

include("cuda.jl")            # fixes for cuda

#include("dataloader.jl")

#include("precompile.jl") # precompile for faster ttx

include("reactionpath2.jl")

include("IsoMu/IsoMu.jl")
include("vgv/vgv.jl")

end

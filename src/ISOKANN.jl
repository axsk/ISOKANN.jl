#__precompile__(false)

module ISOKANN

#using Startup           # precompiles most used packages
#include("forced/IsoForce.jl")

import StochasticDiffEq, Flux, CUDA, PCCAPlus, Plots

using ProgressMeter
using Plots

using LinearAlgebra: norm, dot, cross, diag, svd, pinv, I, schur, qr
using StatsBase: mean, sample, mean_and_std
using StaticArrays: SVector
using StatsBase: sample, quantile
using CUDA: CuArray, CuMatrix, cu, CUDA
using NNlib: batched_adjoint, batched_mul
using Unitful: @u_str, unit
using SpecialFunctions: erf
using Plots: plot, plot!, scatter, scatter!
using MLUtils: numobs, getobs, shuffleobs, unsqueeze
using StaticArrays: @SVector
using StochasticDiffEq: StochasticDiffEq
using PyCall: @py_str, pyimport_conda, PyReverseDims, PyArray
using SimpleWeightedGraphs: SimpleWeightedDiGraph
using SparseArrays: sparse
using Functors: @functor, fmap

import Distances
import Distributions
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
import Optim
import PyCall
import Random
import KernelDensity
import ForwardDiff
import StatsBase
import Flux
import PCCAPlus
import LinearAlgebra

import MLUtils: numobs
import Flux: cpu, gpu

export OpenMM

export pairnet
#export PDB_ACEMD, PDB_1UAO, PDB_diala_water
#export MollyLangevin, propagate, solve#, MollySDE

export propagate
export laggedtrajectory

export run!, runadaptive!
export AdamRegularized, NesterovRegularized
export plot_training, scatter_ramachandran
export reactive_path, save_reactive_path
export cpu, gpu
export Iso
export Doublewell, Triplewell, MuellerBrown
export chis
export SimulationData
export addcoords, addcoords!, resample_kde!, resample_kde
export getxs, getys
export exit_rates
export load_trajectory, save_trajectory
export savecoords
export atom_indices
export localpdistinds, pdists, restricted_localpdistinds
export data_from_trajectory, mergedata
export trajectorydata_bursts, trajectorydata_linear
export reactionpath_minimum, reactionpath_ode
export chicoords
export ca_rmsd


include("subsample.jl")  # adaptive sampling
include("pairdists.jl")       # pair distances
include("simulation.jl")      # Interface for simulations
include("models.jl")          # the neural network models/architectures
#include("simulators/molly.jl")           # interface to work with Molly Systems
include("molutils.jl")        # molecular utilities: dihedrals, rotation
include("data.jl")            # tools for handling the data (sampling, slicing, ...)
#include("iso1.jl")        # ISOKANN - first implementation with adaptive sampling
#include("loggers.jl")     # performance metric loggers
#include("benchmarks.jl")      # benchmark runs, deprecated by scripts/*

include("simulators/langevin.jl")  # for the simulators

#include("isosimple.jl")
include("isotarget.jl")
include("iso.jl")
include("plots.jl")           # visualizations

include("simulators/openmm.jl")

import .OpenMM.OpenMMSimulation
export OpenMMSimulation

#include("dataloader.jl")

#include("precompile.jl") # precompile for faster ttx

include("extrapolate.jl")

include("minimumpath.jl")
include("reactionpath.jl")

#include("IsoMu/IsoMu.jl")
#include("vgv/vgv.jl")

include("makie.jl")
include("bonito.jl")

end

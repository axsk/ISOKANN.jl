#__precompile__(false)

module ISOKANN

#using Startup           # precompiles most used packages

module IsoForce

include("forced/utils.jl")     # neural network convenience

include("forced/langevin.jl")  # langevin process
include("forced/control.jl")   # opt control

include("humboldtsample.jl")  # adaptive sampling
include("forced/isokann.jl")   # new implementation of isokann

end


include("humboldtsample.jl")  # adaptive sampling

include("pairdists.jl")  # pair distances
include("models.jl")     # the neural network models/architectures

include("molly.jl")      # interface to work with Molly Systems
include("simulation.jl") # Langeving dynamic simulator (MollySystem+Integrator)
include("molutils.jl")   # molecular utilities: dihedrals, rotation
include("isomolly.jl")   # ISOKANN for Molly systems
include("plots.jl")      # visualizations

include("data.jl")       # tools for handling the data (sampling, slicing, ...)
include("performance.jl") # performance metric loggers
include("benchmarks.jl") # benchmark runs, deprecated by scripts/*

include("cuda.jl")       # fixes for cuda
include("dataloader.jl")


include("reactionpath.jl")

#include("precompile.jl") # precompile for faster ttx

end

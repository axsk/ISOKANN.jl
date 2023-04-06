#__precompile__(false)

module ISOKANN

using Startup           # precompiles most used packages


include("forced/utils.jl")     # neural network convenience

include("forced/langevin.jl")  # langevin process
include("forced/control.jl")   # opt control

include("humboldtsample.jl")  # adaptive sampling
include("forced/isokann.jl")   # new implementation of isokann

export isokann


include("pairdists.jl")  # pair distances
include("molly.jl")      # mainly Molly-SDE interface
include("molutils.jl")   # dihedrals, rotation
include("isomolly.jl")   # ISOKANN for Molly systems
include("plots.jl")      # visualizations

include("data.jl")       # tools for handling the data (sampling, slicing, ...)
include("performance.jl") # performance metric loggers
include("benchmarks.jl") # benchmark runs, deprecated by scripts/*

include("cuda.jl")       # fixes for cuda

include("precompile.jl") # precompile for faster ttx
end

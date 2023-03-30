#__precompile__(false)

module ISOKANN

using Startup


include("utils.jl")     # neural network convenience

include("langevin.jl")  # langevin process
include("control.jl")   # opt control

include("humboldtsample.jl")  # adaptive sampling
include("isokann.jl")   # new implementation of isokann

export isokann


include("pairdists.jl")  # pair distances
include("molly.jl")      # mainly Molly-SDE interface
include("molutils.jl")   # dihedrals, rotation
include("isomolly.jl")   # ISOKANN for Molly systems
include("plots.jl")      # visualizations

include("data.jl")
include("performance.jl")
include("benchmarks.jl")

include("cuda.jl") # fixes for cuda

end

module ISOKANN

include("utils.jl")
include("utils.jl")     # neural network convenience

include("langevin.jl")  # langevin process
include("control.jl")   # opt control

include("humboldtsample.jl")  # adaptive sampling
include("isokann.jl")   # new implementation of isokann

export isokann

end

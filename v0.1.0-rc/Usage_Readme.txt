Be in the IsoMu.jl directory

start julia 
#package activation
] activate .

#backspace to exit package mode

julia> include("src/main.jl")

#provide data using DataLink, one can provide extra arguments such as startpos for starting frame, stride for every 'n'th step of the trajectory and radius for pairwise distances threshold   
julia> data = DataLink("path to struct.pdb and traj.dcd", startpos=2)

mu = IMu(data)

#run the neural network, extra arguement can be iterations such as run!(iso,1000)
run!(mu)

#save path: one can use full path or quantile(transition path only) & sigma values determine the number of samples gathered (low number, fewer samples)
save_reactive_path(mu, method=:full/quantile sigma=0.1)

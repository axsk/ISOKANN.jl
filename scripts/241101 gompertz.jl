# 1.11.24 - Implementation of Gombartz. With Taifun, Jakob and Marcus

using Revise
using ISOKANN
using PyCall
using Measures
using Flux
path_data = "/data/numerik/people/jkresse/bachelor"
#######################Molecular dynamics##################################
intpairs = Vector{Tuple{Int,Int}}()
for i in 1:73
    for j in i+1:73
        push!(intpairs, (i, j))
    end
end
sim = ISOKANN.OpenMM.OpenMMSimulation(;
    pdb="/home/numerik/jkresse/code/chidirect/vgvapg_unfolded_processed.pdb",
    forcefields=["amber14-all.xml", "implicit/obc2.xml"],
    step=0.002, # picoseconds
    steps=100, # steps per simulation
    temp=310, # Kelvin
    friction=1,
    nthreads=1,
    features=intpairs
)

#30 ns
data = trajectorydata_bursts(sim, 80000, 1)
iso = Iso(data, opt=NesterovRegularized(), minibatch=10000)
ISOKANN.save(path_data * "/iso_vgvapg_new.jld2", iso)

###################Training Chi###########################################
iso = ISOKANN.load(path_data * "/iso_vgvapg_new.jld2")
iso = gpu(iso)
#train iso
@time run!(iso, 1000)

#################Extracting the minimum path################################
simstart = ISOKANN.OpenMMSimulation(; pdb=path_script * "/vgvapg_unfolded_processed.pdb", gpu=false)#workaround because sim is overwritten in iso
x = coords(simstart)
xs = reactionpath_minimum(iso, x; steps=100)
ISOKANN.savecoords(path_data * "/minimum_path_vgvapg.pdb", iso, xs)

xx = hcat(x, x)#workaround to be able to save just one frame
ISOKANN.savecoords(path_data * "/start_point_vgvapg.pdb", iso, xx)

################1Dimensional representation###############################
"""
function Gompertz(a0,a1,a2,a3,a4,x)
  return a0 + a1*exp(-a2*exp(-a3*(x-a4)))
end
"""


@kwdef struct Gompertz4{T}
    a0::T = [1]
    a1::T = [1]
    a2::T = [1]
    a3::T = [1]
    a4::T = [1]
    layers::Tuple = ()
end

Gompertz = Gompertz4

Flux.@layer Gompertz trainable = (a0, a1, a2, a3, a4)

(m::Gompertz)(x) = @. m.a0 + m.a1 * exp(-m.a2 * exp(-m.a3 * (x - m.a4)))

# Use Flux.@layer to allow for training



# Step 2: Create an instance of the Gompertz model with initial parameters
model = Gompertz(ones(4), ())


# Define outputdim for GompertzWrapperModel in the ISOKANN namespace
ISOKANN.outputdim(::Gompertz) = 1  # Assuming it outputs a scalar
ISOKANN.inputdim(::Gompertz) = 1
# Instantiate the model

featurizer = OpenMM.FeaturesPairs([(1, 73)])
sim1d = ISOKANN.OpenMM.OpenMMSimulation(;
    pdb="/home/numerik/jkresse/code/chidirect/vgvapg_unfolded_processed.pdb",
    forcefields=["amber14-all.xml", "implicit/obc2.xml"],
    step=0.002, # picoseconds
    steps=100, # steps per simulation
    temp=310, # Kelvin
    friction=1,
    nthreads=1
)
data1d = SimulationData(sim1d, iso.data.coords, featurizer=featurizer)
iso1d = Iso(data1d, model=model)
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/ISOKANN.jl")
end
#include("../src/ISOKANN.jl")

using ISOKANN
using JLD2

#data = JLD2.load("data/aladipep-implicit-10000x1000.jld2")["data"] #  long trajectory, not showing the ->right  clusters
x0 = JLD2.load("../data/aladi-implicit-picked.jld2")["x"]
sim = OpenMMSimulation(forcefields=OpenMM.FORCE_AMBER_IMPLICIT, steps=1000)

data = SimulationData(sim, x0, 10)

#900s
@time is = [(iso = Iso(model=pairnet(data, nout=5), data=data, opt=NesterovRegularized(1e-3, r), 
    transform=ISOKANN.TransformCross1(npoints=length(data), maxcols=15), loggers=[]); run!(iso, 500)) 
    for i in 1:5, r in [1e-2,1e-3, 1e-4, 1e-5,1e-6]]

using StatsBase: mean

r1 = map(is) do i
    lambda = i.opt.layers[2].diag.scale.rule.opts[1].lambda
    loss = i.losses[end]
    isotarget(i)
    meanres = ISOKANN.ret.res[1:3] |> mean

    (;lambda, loss, meanres)
end

@time is2 = [(iso = Iso(model=pairnet(data, nout=5), data=data, opt=NesterovRegularized(1e-3, r), 
    transform=ISOKANN.TransformCross1(npoints=length(data), maxcols=5), loggers=[]); run!(iso, 500)) 
    for i in 1:5, r in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]]
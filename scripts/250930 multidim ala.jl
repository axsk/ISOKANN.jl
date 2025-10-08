if isdefined(@__MODULE__, :LanguageServer)
    include("../src/ISOKANN.jl")
end
#include("../src/ISOKANN.jl")

using ISOKANN
using JLD2


using StatsBase: mean

function getdata()
    x0 = JLD2.load("../data/aladi-implicit-picked.jld2")["x"]
    sim = OpenMMSimulation(forcefields=OpenMM.FORCE_AMBER_IMPLICIT, steps=10000)

    data = SimulationData(sim, x0, 1)
    #data = JLD2.load("data/aladipep-implicit-10000x1000.jld2")["data"] #  long trajectory, not showing the ->right  clusters


    #900s
    @time is = [(iso = Iso(model=pairnet(data, nout=5), data=data, opt=NesterovRegularized(1e-3, r),
        transform=ISOKANN.TransformCross1(npoints=length(data), maxcols=15), loggers=[]);
    run!(iso, 500))
                for i in 1:5, r in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]]

    @time is2 = [(iso = Iso(model=pairnet(data, nout=5), data=data, opt=NesterovRegularized(1e-3, r),
        transform=ISOKANN.TransformCross1(npoints=length(data), maxcols=5), loggers=[]);
    run!(iso, 500))
                 for i in 1:5, r in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]]

    # more inner training
    @time is3 = [(iso = Iso(model=pairnet(data, nout=5), data=data, opt=NesterovRegularized(1e-3, r),
        transform=ISOKANN.TransformCross1(npoints=length(data), maxcols=5), loggers=[]);
    run!(iso, 500, 10))
                 for i in 1:3, r in [1e-3, 1e-4]]

    # even more
    @time is4 = [(iso = Iso(model=pairnet(data, nout=5), data=data, opt=NesterovRegularized(1e-3, r),
        transform=ISOKANN.TransformCross1(npoints=length(data), maxcols=5), loggers=[]);
    run!(iso, 500, 50))
                 for i in 1:3, r in [1e-3, 1e-4]]
end

#=
r1 = map(is) do i
lambda = i.opt.layers[2].diag.scale.rule.opts[1].lambda
loss = i.losses[end]
ISOKANN.isotarget(i)
    meanres = ISOKANN.ret.res[1:3] |> mean

    (;lambda, loss, meanres)
end

=#

function stats(iso)
    i=iso
    lambda = i.opt.layers[2].diag.scale.rule.opts[1].lambda
    loss = i.losses[end]
    ISOKANN.isotarget(i)
    meanres = ISOKANN.ret.res[1:3] |> mean

    (;lambda, loss, meanres)
end

function validation_nd(iso, data)
    c,k = ISOKANN.chi_kchi(iso.model, data)
    # componentwise relative L_2 error from eigenvalue reconstruction
    norm.(eachrow(Diagonal(k / c) * c - k)) ./ norm.(eachrow(k))
end
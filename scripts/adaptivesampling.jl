
using Flux

if false # workaround vscode "missing reference bug"
    include("../src/ISOKANN.jl")
    using .ISOKANN: randx0, propagate, TransformShiftscale, TransformISA, adddata, isosteps, defaultmodel, OpenMMSimulation
end

using ISOKANN: randx0, propagate, TransformShiftscale, TransformISA, adddata, isosteps, defaultmodel, OpenMMSimulation

function adapt_setup(;
    steps=100,
    sim=OpenMMSimulation(pdb="data/alanine-dipeptide-nowater av.pdb", steps=steps, features=1:22),
    nx0=10,
    nmc=5,
    nd=1,
    lr=1e-3,
    decay=1e-5,
    nlayers=4,
    activation=Flux.relu,
    transform=(nd == 1 ? TransformShiftscale() : TransformISA())
)
    xs = randx0(sim, nx0)
    ys = propagate(sim, xs, nmc)

    model = defaultmodel(sim; nout=nd, activation, layers=nlayers)
    opt = Flux.setup(Flux.AdamW(lr, (0.9, 0.999), decay), model)

    losses = Float64[]
    targets = Matrix{Float64}[]

    return (; xs, ys, model, opt, sim, transform, losses, targets)
end

function adapt_run(; iso,
    epochs=1, nresample=0, nkoop=1, nupdate=1)

    (; xs, ys, model, opt, sim, transform, losses, targets) = iso

    for e in 1:epochs
        @time "resampling" xs, ys = adddata((xs, ys), model, sim, nresample,)
        @time "training" l, t = isosteps(model, opt, (xs, ys), nkoop, nupdate; transform, losses, targets)
    end

    return (; xs, ys, model, opt, sim, transform, losses, targets)
end

function test_adapt()
    iso = adapt_setup()
    iso = adapt_run(; iso, nresample=1)
end

function gpu(iso::NamedTuple)

    (; xs, ys, model, opt, sim, transform, losses, targets) = iso

    model = Flux.gpu(model)
    opt = Flux.gpu(opt)
    xs = Flux.gpu(xs)
    ys = Flux.gpu(ys)

    iso = (; xs, ys, model, opt, sim, transform, losses, targets)

end
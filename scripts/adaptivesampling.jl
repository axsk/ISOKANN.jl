
using Flux

if false # workaround vscode "missing reference bug"
    include("../src/ISOKANN.jl")
    using .ISOKANN: randx0, propagate, TransformShiftscale, TransformISA, adddata, isosteps, defaultmodel, OpenMMSimulation
end

using ISOKANN: ISOKANN, randx0, propagate, TransformShiftscale, TransformISA, adddata, isosteps, defaultmodel, OpenMMSimulation, scatter_ramachandran, scatter_chifix, plot_chi

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

@kwdef mutable struct ISO2
    sim
    transform
    model
    opt

    nepochs::Int
    nresample::Int
    npower::Int
    nupdate::Int

    xs
    ys
    losses
    targets
end


# The new IsoRun()
# missing features:
# live visualization / loggers
# minibatch

function IsoRun2(;
    sim=OpenMMSimulation(),
    nchi=1,
    transform=(nchi == 1 ? TransformShiftscale() : TransformISA()),
    nlayers=4,
    activation=Flux.relu,
    model=defaultmodel(sim; nout=nchi, activation, layers=nlayers),
    lr=1e-4,
    decay=1e-5,
    opt=Flux.setup(Flux.AdamW(lr, (0.9, 0.999), decay), model),
    nx0=10,
    nmc=10,
    nepochs=1,
    nresample=0,
    npower=1,
    nupdate=1,
    xs=randx0(sim, nx0),
    ys=propagate(sim, xs, nmc),
    losses=Float64[],
    targets=Matrix{Float64}[],
)
    ISO2(; sim, transform, model, opt,
        nepochs, nresample, npower, nupdate,
        xs, ys, losses, targets)
end

function ISOKANN.run!(iso::ISO2)
    (; sim, transform, model, opt,
        nepochs, nresample, npower, nupdate,
        losses, targets) = iso

    for _ in 1:nepochs
        @time "resampling" iso.xs, iso.ys = adddata((iso.xs, iso.ys), model, sim, nresample,)
        @time "training" isosteps(model, opt, (iso.xs, iso.ys), npower, nupdate; transform, losses, targets)
    end

    return iso
end

using Plots

function plot_training(iso::ISO2)
    (; losses, xs, model) = iso
    p1 = plot(losses, yaxis=:log, title="loss", label="trainloss", xlabel="iter")
    p2 = plot_chi(xs, vec(model(xs)))
    p3 = scatter_chifix((iso.xs, iso.ys), model)
    ps = [p1, p2, p3]
    plot(ps..., layout=(length(ps), 1), size=(400, 300 * length(ps)))
end



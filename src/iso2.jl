# test implementation of ISOKANN 2.0

# for a simple demonstration try `test_dw()` and `test_tw()`

import StatsBase
import Flux
import PCCAPlus
import ISOKANN
using LinearAlgebra: pinv, norm, I, schur
using Plots

@kwdef mutable struct Iso2
    model
    opt
    data
    transform
    losses = Float64[]
    loggers = [autoplot(1)]
    minibatch = 0
end

function Iso2(data; opt=AdamRegularized(), model=pairnet(data), gpu=false, kwargs...)
    opt = Flux.setup(opt, model)
    transform = outputdim(model) == 1 ? TransformShiftscale() : TransformISA()

    iso = Iso2(; model, opt, data, transform, kwargs...)
    gpu && (iso = ISOKANN.gpu(iso))
    return iso
end

Iso2(iso::IsoRun) = Iso2(iso.model, iso.opt, iso.data, TransformShiftscale(), iso.losses, iso.loggers, iso.minibatch)

function run!(iso::Iso2, n=1, epochs=1)
    p = ProgressMeter.Progress(n)
    iso.opt isa Optimisers.AbstractRule && (iso.opt = Optimisers.setup(iso.opt, iso.model))

    for _ in 1:n
        xs, ys = getobs(iso.data)
        target = isotarget(iso.model, xs, ys, iso.transform)
        for i in 1:epochs
            loss = train_batch2!(iso.model, iso.data[1], target, iso.opt, iso.minibatch)
            push!(iso.losses, loss)
        end

        for logger in iso.loggers
            log(logger; iso, subdata=nothing)
        end

        ProgressMeter.next!(p; showvalues=() -> [(:loss, iso.losses[end]), (:n, length(iso.losses)), (:data, size(ys))])
    end
    return iso
end

function train_batch2!(model, xs::AbstractMatrix, ys::AbstractMatrix, opt, minibatch; shuffle=true)
    batchsize = minibatch == 0 ? size(ys, 2) : minibatch
    data = Flux.DataLoader((xs, ys); batchsize, shuffle)
    ls = 0.0
    Flux.train!(model, data, opt) do m, x, y
        l = sum(abs2, m(x) .- y)
        ls += l
        l / numobs(x)
    end
    return ls / numobs(xs)
end

chis(iso::Iso2) = iso.model(iso.data[1])
isotarget(iso::Iso2) = isotarget(iso.model, iso.data..., iso.transform)

Optimisers.adjust!(iso::Iso2; kwargs...) = Optimisers.adjust!(iso.opt; kwargs...)
Optimisers.setup(iso::Iso2) = (iso.opt = Optimisers.setup(iso.opt, iso.model))
Flux.gpu(iso::Iso2) = Iso2(Flux.gpu(iso.model), Flux.gpu(iso.opt), Flux.gpu(iso.data), iso.transform, iso.losses, iso.loggers, iso.minibatch)
Flux.cpu(iso::Iso2) = Iso2(Flux.cpu(iso.model), Flux.cpu(iso.opt), Flux.cpu(iso.data), iso.transform, iso.losses, iso.loggers, iso.minibatch)

function Base.show(io::IO, mime::MIME"text/plain", iso::Iso2)
    println(io, typeof(iso), ":")
    println(io, " model: $(iso.model.layers)")
    println(io, " tranform: $(iso.transform)")
    println(io, " opt: $(optimizerstring(iso.opt))")
    println(io, " minibatch: $(iso.minibatch)")
    println(io, " loggers: $(length(iso.loggers))")
    println(io, " data: $(size.(iso.data)), $(typeof(iso.data))")
    length(iso.losses) > 0 && println(io, " loss: $(iso.losses[end]) (length: $(length(iso.losses)))")
end

# TODO: rewrite for Iso2
# with adaptive sampling
function run!(iso::NamedTuple; epochs=1, ny=10, nkoop=1000, nupdate=10)
    (; data, model, sim, opt, losses) = iso
    local target
    for _ in 1:epochs
        l, targets = isosteps(model, opt, data, nkoop, nupdate)
        data = adddata(data, model, sim, ny)
        append!(losses, l)
        target = targets[end]
        vis_training(; model, data, target, losses) |> display
    end

    (; data, model, sim, opt, losses, target)
end


# constructor for simulation
# TODO: dispatch on AbstractSimulation / IsoSimulation / Simulation
function IsoSimulation(sim=Doublewell(); nx=100, ny=10, nd=2, kwargs...)
    xs = randx0(sim, nx)
    ys = propagate(sim, xs, ny)
    data = (xs, ys)

    model = defaultmodel(sim; nout=nd)

    return Iso2(data; model; kwargs...)
end

# TODO: update for Iso2
### Examples with different simulations

function isodata(diffusion, nx, ny)
    d = diffusion
    xs = randx0(d, nx)
    ys = propagate(d, xs, ny)
    return xs, ys
end

function test_dw(; kwargs...)
    i = iso2(nd=2, sim=Doublewell(); kwargs...)
    vismodel(i.model)
end

function test_tw(; kwargs...)
    i = iso2(nd=3, sim=Triplewell(); kwargs...)
    vismodel(i.model)
end

# obtain data from 1d ISOKANN
function diala2data()
    iso = ISOKANN.IsoRun(nd=3000, loggers=[])
    ISOKANN.run!(iso)
    iso
end

""" ISOKANN 2 dialanine experiment
generates data from 1d ISOKANN and trains a 3d ISOKANN on it
"""
function diala2(; data=nothing, nd=3000)
    if isnothing(data)
        iso = run!(IsoRun(nd=nd, loggers=[]))
        data = iso.data
    end

    sim = MollyLangevin(sys=PDB_ACEMD())
    model = ISOKANN.pairnet(484, features=flatpairdists, nout=3)
    opt = Flux.setup(Flux.AdamW(1e-3, (0.9, 0.999), 1e-4), model)
    losses = Float64[]

    return (; data, model, sim, opt, losses)
end

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

function Iso2(sim::IsoSimulation; nx=100, nk=10, nd=1, kwargs...)
    data = isodata(sim, nx, nk)
    model = defaultmodel(sim; nout=nd)
    return Iso2(data; model, kwargs...)
end

Iso2(iso::IsoRun) = Iso2(iso.model, iso.opt, iso.data, TransformShiftscale(), iso.losses, iso.loggers, iso.minibatch)

function run!(iso::Iso2, n=1, epochs=1)
    p = ProgressMeter.Progress(n)
    iso.opt isa Optimisers.AbstractRule && (iso.opt = Optimisers.setup(iso.opt, iso.model))

    for _ in 1:n
        xs, ys = getobs(iso.data)
        target = isotarget(iso.model, xs, ys, iso.transform)
        for i in 1:epochs
            loss = train_batch2!(iso.model, xs, target, iso.opt, iso.minibatch)
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

chis(iso::Iso2) = iso.model(getxs(iso.data))
isotarget(iso::Iso2) = isotarget(iso.model, getobs(iso.data)..., iso.transform)

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
    #println(io, " data: $(size.(iso.data)), $(typeof(iso.data))")
    length(iso.losses) > 0 && println(io, " loss: $(iso.losses[end]) (length: $(length(iso.losses)))")
end

# TODO: rewrite for Iso2
# with adaptive sampling
function run!(iso::Iso2, sim::IsoSimulation, generates=1, iter=1, epochs=1; ny)
    for _ in 1:generates
        iso.data = adddata(iso.data, iso.model, sim, ny)
        run!(iso, iter, epochs)
    end
    return iso
end


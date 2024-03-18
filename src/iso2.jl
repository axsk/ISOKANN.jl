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


"""
    Iso2(data; opt=AdamRegularized(), model=pairnet(data), gpu=false, kwargs...)

"""
function Iso2(data; opt=AdamRegularized(), model=pairnet(data), gpu=false, kwargs...)
    opt = Flux.setup(opt, model)
    transform = outputdim(model) == 1 ? TransformShiftscale() : TransformISA()

    iso = Iso2(; model, opt, data, transform, kwargs...)
    gpu && (iso = ISOKANN.gpu(iso))
    return iso
end

"""
    Iso2(sim::IsoSimulation; nx=100, nk=10, nd=1, kwargs...)

Convenience constructor which generates the `SimulationData` from the simulation `sim`
and constructs the Iso2 object. See also Iso2(data; kwargs...)

## Arguments
- `sim::IsoSimulation`: The `IsoSimulation` object.
- `nx::Int`: The number of starting points.
- `nk::Int`: The number of koopman samples.
- `nd::Int`: Dimension of the χ function.
"""
function Iso2(sim::IsoSimulation; nx=100, nk=10, nd=1, kwargs...)
    data = SimulationData(sim; nx, nk)
    model = pairnet(data; nout=nd)  # maybe defaultmodel(data) makes sense here?
    return Iso2(data; model, kwargs...)
end

#Iso2(iso::IsoRun) = Iso2(iso.model, iso.opt, iso.data, TransformShiftscale(), iso.losses, iso.loggers, iso.minibatch)

function run!(iso::Iso2, n=1, epochs=1)
    p = ProgressMeter.Progress(n)
    iso.opt isa Optimisers.AbstractRule && (iso.opt = Optimisers.setup(iso.opt, iso.model))

    for _ in 1:n
        xs, ys = getobs(iso.data)
        target = isotarget(iso.model, xs, ys, iso.transform)
        for i in 1:epochs
            loss = train_batch!(iso.model, xs, target, iso.opt, iso.minibatch)
            push!(iso.losses, loss)
        end

        for logger in iso.loggers
            log(logger; iso, subdata=nothing)
        end

        ProgressMeter.next!(p; showvalues=() -> [(:loss, iso.losses[end]), (:n, length(iso.losses)), (:data, size(ys))])
    end
    return iso
end

function train_batch!(model, xs::AbstractMatrix, ys::AbstractMatrix, opt, minibatch; shuffle=true)
    batchsize = minibatch == 0 || size(xs, 2) < minibatch ? size(ys, 2) : minibatch
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
chicoords(iso::Iso2, xs) = iso.model(iso.data.featurizer(xs))
isotarget(iso::Iso2) = isotarget(iso.model, getobs(iso.data)..., iso.transform)

Optimisers.adjust!(iso::Iso2; kwargs...) = Optimisers.adjust!(iso.opt; kwargs...)
Optimisers.setup(iso::Iso2) = (iso.opt = Optimisers.setup(iso.opt, iso.model))
gpu(iso::Iso2) = Iso2(Flux.gpu(iso.model), Flux.gpu(iso.opt), Flux.gpu(iso.data), iso.transform, iso.losses, iso.loggers, iso.minibatch)
cpu(iso::Iso2) = Iso2(Flux.cpu(iso.model), Flux.cpu(iso.opt), Flux.cpu(iso.data), iso.transform, iso.losses, iso.loggers, iso.minibatch)

function Base.show(io::IO, mime::MIME"text/plain", iso::Iso2)
    println(io, typeof(iso), ":")
    println(io, " model: $(iso.model.layers)")
    println(io, " tranform: $(iso.transform)")
    println(io, " opt: $(optimizerstring(iso.opt))")
    println(io, " minibatch: $(iso.minibatch)")
    println(io, " loggers: $(length(iso.loggers))")
    println(io, " data: $(size.(getobs(iso.data))), $(typeof(getobs(iso.data)))")
    length(iso.losses) > 0 && println(io, " loss: $(iso.losses[end]) (length: $(length(iso.losses)))")
end

"""
    runadaptive!(iso; generations=100, nx=10, iter=100, cutoff=1000)

Train iso with adaptive sampling. Sample `nx` new data points followed by `iter` isokann iterations and repeat this `generations` times.
`cutoff` specifies the maximal data size, after which new data overwrites the oldest data.
"""
function runadaptive!(iso; generations=100, nx=10, iter=100, cutoff=1000)
    for _ in 1:generations
        @time adddata!(iso, nx)
        @time run!(iso, iter)
        #@show exit_rates(iso)

        if length(iso.data) > cutoff
            iso.data = iso.data[end-cutoff+1:end]
        end

        CUDA.reclaim()
    end
    iso
end

function adddata!(iso::Iso2, nx)
    iso.data = ISOKANN.adddata(iso.data, iso.model, nx)
end



log(f::Function; kwargs...) = f(; kwargs...)
log(logger::NamedTuple; kwargs...) = :call in keys(logger) && logger.call(; kwargs...)

""" evluation of koopman by shiftscale(mean(model(data))) on the data """
function koopman(model, ys)
    #ys = Array(ys)
    cs = model(ys)::AbstractArray{<:Number,3}
    #ks = vec(StatsBase.mean(cs[1, :, :], dims=2))::AbstractVector
    ks = dropdims(StatsBase.mean(cs, dims=2), dims=2)
    return ks
end

""" empirical shift-scale operation """
shiftscale(ks) =
    let (a, b) = extrema(ks)
        (ks .- a) ./ (b - a)
    end

""" compute the chi exit rate as per Ernst, Weber (2017), chap. 3.3 """
function chi_exit_rate(x, Kx, tau)
    @. shiftscale(x, p) = p[1] * x + p[2]
    γ1, γ2 = LsqFit.coef(LsqFit.curve_fit(shiftscale, vec(x), vec(Kx), [1, 0.5]))
    α = -1 / tau * Base.log(γ1)
    β = α * γ2 / (γ1 - 1)
    return α + β
end

chi_exit_rate(iso::Iso2, tau) = chi_exit_rate(iso.model(getxs(iso.data)), koopman(iso.model, getys(iso.data)), tau)


function exit_rates(x, kx, tau)
    o = ones(length(x))
    x = vec(x)
    kx = vec(kx)
    P = [x o .- x] \ [kx o .- kx]
    return -1 / tau .* Base.log.(diag(P))
end

koopman(iso::Iso2) = koopman(iso.model, getys(iso.data))

exit_rates(iso::Iso2) = exit_rates(cpu(chis(iso)), cpu(koopman(iso)), iso.data.sim.step * iso.data.sim.steps)
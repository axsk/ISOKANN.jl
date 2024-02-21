# test implementation of ISOKANN 2.0

# for a simple demonstration try `test_dw()` and `test_tw()`

import StatsBase
import Flux
import PCCAPlus
import ISOKANN
using LinearAlgebra: pinv, norm, I, schur
using Plots
#include("isosimple.jl")  # for the learnstep! function

"""
    iso2(; n=1000, nx=100, ny=10, nd=2, sys=Doublewell(), lr=1e-2, decay=1e-5)

ISOKANN 2.0 under construction.

## Arguments
- `n`: Number of iterations (default: 1000)
- `nx`: Number of start points (default: 100)
- `ny`: Number of end points per start (default: 10)
- `nd`: Number of dimensions of the koopman function (default: 2)
- `sys`: System object (needs to support randx0, dim, propagate) (default: Doublewell())
- `lr`: Learning rate
- `decay`: Decay rate
"""
function iso2(; n=1000, nx=100, ny=10, nd=2, sim=Doublewell(), lr=1e-3, decay=1e-5, transform=TransformISA(), kwargs...)
    s = sim
    xs = randx0(sim, nx)
    ys = propagate(sim, xs, ny)

    model, opt = model_with_opt(defaultmodel(sim; nout=nd), lr, decay)

    losses, target = isosteps(model, opt, (xs, ys), n; transform, kwargs...)

    return (; sim, xs, ys, model, opt, losses, target, transform, kwargs...)
end


""" isostep(model, opt, (xs, ys), nkoop=1, nupdate=1; transform = TransformISA())

train the model on a given batch of trajectory data `(xs, ys)` with
- nkoop: outer iterations, i.e. reevaluating Koopman
- nupdate: inner iterations, i.e. updating the neural network on fixed data
"""
function isosteps(model, opt, (xs, ys), nkoop=1, nupdate=1;
    transform=TransformISA(),
    losses=Float64[],
    targets=[])

    local target = nothing
    for i in 1:nkoop
        target = isotarget(model, xs, ys, transform)
        push!(targets, target)
        # note that nupdate is classically called epochs
        # TODO: should we use minibatches here?
        for j in 1:nupdate
            loss = ISOKANN.learnstep!(model, xs, target, opt)  # Neural Network update
            push!(losses, loss)
        end
    end
    (; losses, targets)
end


isosteps(iso::NamedTuple; kwargs...) = isosteps(; iso..., kwargs...)
function isosteps(; nkoop=1, nupdate=1, exp...)
    exp = NamedTuple(exp)
    (; xs, ys, model, opt, losses, transform) = exp
    l2, target = isosteps(model, opt, (xs, ys), nkoop, nupdate; transform)
    (; exp..., target, losses=vcat(losses, l2), nkoop, nupdate)
end

### ISOKANN target transformations

"""
    TransformPseudoInv(normalize, direct, eigenvecs, permute)

Compute the target by approximately inverting the action of K with the Moore-Penrose pseudoinverse.

If `direct==true` solve `chi * pinv(K(chi))`, otherwise `inv(K(chi) * pinv(chi)))`.
`eigenvecs` specifies whether to use the eigenvectors of the schur matrix.
`normalize` specifies whether to renormalize the resulting target vectors.
`permute` specifies whether to permute the target for stability.
"""
@kwdef struct TransformPseudoInv
    normalize::Bool = true
    direct::Bool = true
    eigenvecs::Bool = true
    permute::Bool = true
end

function isotarget(model, xs, ys, t::TransformPseudoInv)
    (; normalize, direct, eigenvecs, permute) = t
    chi = model(xs)
    size(chi, 1) > 1 || error("TransformPseudoInv does not work with one dimensional chi functions")

    cs = model(ys)::AbstractArray{<:Number,3}
    kchi = StatsBase.mean(cs[:, :, :], dims=2)[:, 1, :]

    if direct
        Kinv = chi * pinv(kchi)
        s = schur(Kinv)
        T = eigenvecs ? inv(s.vectors) : I
        target = T * Kinv * kchi
    else
        K = kchi * pinv(chi)
        s = schur(K)
        T = eigenvecs ? inv(s.vectors) : I
        target = T * inv(K) * kchi
    end

    normalize && (target = target ./ norm.(eachrow(target), 1) .* size(target, 2))
    permute && (target = fixperm(target, chi))

    return target
end


""" TransformISA(permute)

Compute the target via the inner simplex algorithm (without feasiblization routine).
`permute` specifies whether to apply the stabilizing permutation """
@kwdef struct TransformISA
    permute::Bool = true
end

# we cannot use the PCCAPAlus inner simplex algorithm because it uses feasiblize!,
# which in turn assumes that the first column is equal to one.
function myisa(X)
    inv(X[PCCAPlus.indexmap(X), :])
end

function isotarget(model, xs, ys, t::TransformISA)
    chi = model(xs)
    size(chi, 1) > 1 || error("TransformISA does not work with one dimensional chi functions")
    cs = model(ys)
    ks = StatsBase.mean(cs[:, :, :], dims=2)[:, 1, :]
    target = myisa(ks')' * ks
    t.permute && (target = fixperm(target, chi))
    return target
end

""" TransformShiftscale()

Classical 1D shift-scale (ISOKANN 1) """
struct TransformShiftscale end

function isotarget(model, xs, ys, t::TransformShiftscale)
    cs = model(ys)
    size(cs, 1) == 1 || error("TransformShiftscale only works with one dimensional chi functions")
    ks = StatsBase.mean(cs[:, :, :], dims=2)[:, 1, :]
    target = (ks .- minimum(ks)) ./ (maximum(ks) - minimum(ks))
    return target
end


### Permutation - stabilization
import Combinatorics

"""
    fixperm(new, old)

Permutes the rows of `new` such as to minimize L1 distance to `old`.

# Arguments
- `new`: The data to match to the reference data.
- `old`: The reference data.
"""
function fixperm(new, old)
    # TODO: use the hungarian algorithm for larger systems
    n = size(new, 1)
    p = argmin(Combinatorics.permutations(1:n)) do p
        norm(new[p, :] - old, 1)
    end
    if p != collect(1:length(p))
        @show p
    end
    new[p, :]
end

using Random: shuffle
function test_fixperm(n=3)
    old = rand(n, n)
    @show old
    new = old[shuffle(1:n), :]
    new = fixperm(new, old)
    @show new
    norm(new - old) < 1e-9
end


### Examples with the Double and Triplewell

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


### Visualization

function vismodel(model, grd=-2:0.03:2; xs=nothing, ys=nothing, float=0.01, kwargs...)
    defargs = (; markeralpha=0.1, markersize=0.5, markerstrokewidth=0)
    dim = ISOKANN.inputdim(model)
    if dim == 1
        plot(grd, model(collect(grd)')')
    elseif dim == 2
        p = plot()
        g = makegrid(grd, grd)
        y = model(g)
        for i in 1:ISOKANN.outputdim(model)
            yy = reshape(y[i, :], length(grd), length(grd))
            surface!(p, grd, grd, yy, clims=(0, 1); kwargs...)
        end
        if !isnothing(ys)
            yy = reshape(ys, 2, :)
            scatter!(eachrow(yy)..., maximum(model(yy), dims=1) .+ float |> vec; markercolor=:blue, defargs..., kwargs...)
        end
        !isnothing(xs) && scatter!(eachrow(xs)..., maximum(model(xs), dims=1) .+ float |> vec; markercolor=:red, kwargs...)
        plot!(; kwargs...)
    end
end

function makegrid(x, y)
    A = zeros(Float32, 2, length(x) * length(y))
    i = 1
    for x in x
        for y in y
            A[:, i] .= (x, y)
            i += 1
        end
    end
    A
end


###  DIALANINE

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

function vis_training(; model, data, target, losses, others...)
    p1 = visualize_diala(model, data[1],)
    p2 = scatter(eachrow(target)..., markersize=1)
    #p3 = plot(losses, yaxis=:log)
    plot(p1, p2)#, p3)
end

function visualize_diala(mm, xs; kwargs...)
    p1, p2 = ISOKANN.phi(xs), ISOKANN.psi(xs)
    plot()
    for chi in eachrow(mm(xs))
        markersize = max.(chi .* 3, 0.01)
        scatter!(p1, p2, chi; kwargs..., markersize, markerstrokewidth=0)
    end
    plot!()
end


### Simplex plotting - should be in PCCAPlus.jl
using Plots

function euclidean_coords_simplex()
    s1 = [0, 0, 0]
    s2 = [1, 0, 0]
    s3 = [0.5, sqrt(3) / 2, 0]
    s4 = [0.5, sqrt(3) / 4, sqrt(3) / 2]
    hcat(s1, s2, s3, s4)'
end

function plot_simplex(; n=2, kwargs...)
    c = euclidean_coords_simplex()
    c = c[1:(n+1), 1:n]
    for i in 1:(n+1), j in i+1:(n+1)
        plot!(eachcol(c[[i, j], :])...; kwargs...)
    end
    plot!()
end

function bary_to_euclidean(x::AbstractMatrix)
    n = size(x, 2)
    x * euclidean_coords_simplex()[1:n, 1:(n-1)]
end

function scatter_chi!(chi; kwargs...)
    c = bary_to_euclidean(chi)
    scatter!(eachcol(c)...; kwargs...)
end

scatter_chi(chi; kwargs...) = (plot(); scatter_chi!(chi; kwargs...))

function plot_path(chi, path; kwargs...)
    plot!(eachcol(bary_to_euclidean(chi[path, :]))...; kwargs...)
end


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
    if gpu
        model = gpu(model)
        opt = gpu(opt)
        data = gpu(data)
    end
    Iso2(; model, opt, data, transform, kwargs...)
end

function run!(iso::Iso2, n=1, epochs=1)
    p = ProgressMeter.Progress(n)

    for _ in 1:n
        xs, ys = getobs(iso.data)
        target = isotarget(iso.model, xs, ys, iso.transform)
        for i in 1:epochs
            ls = learnbatch!(iso.model, xs, target, iso.opt, iso.minibatch)
            push!(iso.losses, ls)
        end

        for logger in iso.loggers
            log(logger; iso, subdata=nothing)
        end

        ProgressMeter.next!(p; showvalues=() -> [(:loss, iso.losses[end]), (:n, length(iso.losses)), (:data, size(ys))])
    end
    return iso
end

Flux.adjust!(iso::Iso2; kwargs...) = Flux.adjust!(iso.opt; kwargs...)
Flux.gpu(iso::Iso2) = Iso2(Flux.gpu(iso.model), Flux.gpu(iso.opt), Flux.gpu(iso.data), iso.transform, iso.losses, iso.loggers, iso.minibatch)
function Base.show(io::IO, mime::MIME"text/plain", iso::Iso2)
    println(io, typeof(iso), ":")
    println(io, " model: $(iso.model.layers)")
    println(io, " opt: $(optimizerstring(iso.opt))")
    println(io, " minibatch: $(iso.minibatch)")
    println(io, " loggers: $(length(iso.loggers))")
    println(io, " data: $(size.(iso.data))")
    length(iso.losses) > 0 && println(io, " loss: $(iso.losses[end]) (length: $(length(iso.losses)))")
end


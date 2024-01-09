# test implementation of ISOKANN 2.0

# for a simple demonstration try `test_dw()` and `test_tw()`

import StatsBase
import Flux
import PCCAPlus
import ISOKANN
using LinearAlgebra: pinv, norm, I, schur
using Plots
#include("isosimple.jl")  # for the learnstep! function
include("forced/langevin.jl")  # for the simulators

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
function iso2(; n=1000, nx=100, ny=10, nd=2, sys=Doublewell(), lr=1e-2, decay=1e-5)
    global s, model, xs, ys, opt, loss
    s = sys
    xs = randx0(sys, nx)
    ys = propagate(sys, xs, ny)

    nl = Flux.sigmoid
    model = Flux.Chain(
        Flux.Dense(dim(sys), 5, nl),
        Flux.Dense(5, 10, nl),
        Flux.Dense(10, 5, nl),
        Flux.Dense(5, nd))

    opt = Flux.setup(Flux.AdamW(lr, (0.9, 0.999), decay), model)
    loss = isosteps(model, opt, (xs, ys), n)
end

""" isostep(model, opt, (xs, ys), nkoop=1, nupdate=1)

train the model on a given batch of trajectory data `(xs, ys)` with
- nkoop: outer iterations, i.e. reevaluating Koopman
- nupdate: inner iterations, i.e. updating the neural network on fixed data
"""
function isosteps(model, opt, (xs, ys), nkoop=1, nupdate=1; targetargs=(;))
    losses = Float64[]
    local target
    for i in 1:nkoop
        target = isotarget(model, xs, ys; targetargs...)
        for j in 1:nupdate
            loss = ISOKANN.learnstep!(model, xs, target, opt)  # Neural Network update
            push!(losses, loss)
        end
    end
    (; losses, target)
end

ISO2_METHOD = :isa
ISO2_NORMALIZE = true
ISO2_PERMUTE = true
ISO2_DIRECT = false
ISO2_EIGENVECS = true

""" isotarget(model, xs, ys, method=:inv, kwargs...)
compute the isokann target function for the given model and data.
optional arguments:
- `method`: `:inv` or `:isa` for the pseudoinverse or inner simplex algorithm
- `normalize`: normalize the target function to have unit norm (:inv only)
- `permute`: permute the target function to match the chi function
- `direct`: use the direct method instead of the inverse method (:inv only)
- `eigenvecs`: use the eigenvecs of the schur decomposition instead of the identity matrix (:inv only)
"""
function isotarget(model, xs, ys; method=:isa, kwargs...)
    method == :inv && return isotarget_pseudoinv(model, xs, ys; kwargs...)
    method == :isa && return isotarget_isa(model, xs, ys; kwargs...)
    error("unknown method $method")
end

function isotarget_pseudoinv(model, xs, ys;
    normalize=ISO2_NORMALIZE,
    permute=ISO2_PERMUTE,
    direct=ISO2_DIRECT,
    eigenvecs=ISO2_EIGENVECS
)
    chi = model(xs)

    cs = model(ys)::AbstractArray{<:Number,3}
    kchi = StatsBase.mean(cs[:, :, :], dims=3)[:, :, 1]

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

# we cannot use the PCCAPAlus inner simplex algorithm because it uses feasiblize!,
# which in turn assumes that the first column is equal to one.
myisa(X) = inv(X[PCCAPlus.indexmap(X), :])

function isotarget_isa(model, xs, ys; permute=ISO2_PERMUTE)
    chi = model(xs)
    cs = model(ys)
    ks = StatsBase.mean(cs[:, :, :], dims=3)[:, :, 1]
    target = myisa(ks')' * ks
    permute && (target = fixperm(target, chi))
    return target
end

### Permutation - stabilization
using Combinatorics

#

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
    p = argmin(permutations(1:n)) do p
        norm(new[p, :] - old, 1)
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
    iso2(nd=2, sys=Doublewell(); kwargs...)
    vismodel(model)
end

function test_tw(; kwargs...)
    iso2(nd=3, sys=Triplewell(); kwargs...)
    vismodel(model)
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
The kwargs are passed on to `isotarget`
"""
function diala2(; data=nothing, nd=3000, kwargs...)
    if isnothing(data)
        iso = run!(IsoRun(nd=nd, loggers=[]))
        data = iso.data
    end

    sim = MollyLangevin(sys=PDB_ACEMD())
    model = ISOKANN.pairnet(66, nout=3)
    opt = Flux.setup(Flux.AdamW(1e-3, (0.9, 0.999), 1e-4), model)
    losses = Float64[]

    return (; data, model, sim, opt, losses)
end

function run!(iso::NamedTuple; epochs=1, ny=10, nkoop=1000, kwargs...)
    (; data, model, sim, opt, losses) = iso
    local target
    for _ in 1:epochs
        l, target = isosteps(model, opt, data, nkoop, targetargs=kwargs)
        data = adddata(data, model, sim, ny)
        append!(losses, l)
        vis_training(; model, data, target, losses) |> display
    end

    (; data, model, sim, opt, losses, target)
end

function vis_training(; model, data, target, losses, others...)
    p1 = visualize_diala(model, data[1],)
    p2 = scatter(eachrow(target)..., markersize=0.1)
    p3 = plot(losses, yaxis=:log)
    plot(p1, p2, p3)
end


function visualize_diala(mm, xs; markersize, kwargs...)
    p1, p2 = ISOKANN.phi(xs), ISOKANN.psi(xs)
    plot()
    for chi in eachrow(mm(xs))
        @show markersize = max.(chi .* 3, 0.01)
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
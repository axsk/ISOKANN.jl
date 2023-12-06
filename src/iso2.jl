# test implementation of ISOKANN 2.0

# for a simple demonstration try `test_dw()` and `test_tw()`

import StatsBase
import Flux
using ISOKANN: inputdim, outputdim
using LinearAlgebra: pinv, eigen, norm, diag, I
using Plots
include("isosimple.jl")  # for the learnstep! function
include("forced/langevin.jl")  # for the simulators

""" iso2(; kwargs...)
ISOKANN 2.0 under construction. 
Uses the pseudoinverse, normalization, and permutation stabilization
- `n`: number of steps
- `nx`: number of sampling points
- `ny`: number of koopman samples
- `nd`: number of dimensions of the chi function / number of metastabilities
- `sys`: system to simulate (needs to support randx0, dim, propagate)
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
- nkoop outer iterations, i.e. reevaluating Koopman
- nupdate inner iterations, i.e. updating the neural network on fixed data
"""
function isosteps(model, opt, (xs, ys), nkoop=1, nupdate=1)
    losses = Float64[]
    global target
    for i in 1:nkoop
        target = isotarget(model, xs, ys)
        for j in 1:nupdate
            loss = learnstep!(model, xs, target, opt)  # Neural Network update
            push!(losses, loss)
        end
    end
    losses
end

""" isotarget(model, xs, ys)
compute the isokann target function for the given model and data"""
#isotarget(model, xs, ys) = isotarget_pseudoinv(model, xs, ys)
isotarget(model, xs, ys) = isotarget_isa(model, xs, ys)

function isotarget_pseudoinv(model, xs, ys; normalize=true, permute=true, direct=true, eigenvecs=true, debug=false)
    chi = model(xs)

    cs = model(ys)::AbstractArray{<:Number,3}
    ks = StatsBase.mean(cs[:, :, :], dims=3)[:, :, 1]

    target = Kinv(chi, ks; direct, eigenvecs)

    if normalize
        target = target ./ norm.(eachrow(target), 1) .* size(target, 2)
        target = real.(target)
    end

    # stabilization of permutationn
    if permute
        A = chi * target'  # matrix of the inner products
        P = stableperm(A)  # permutation matrix to maximize diagonal
        target = P * target
    end

    # TODO: this does not belong here
    rand() > (1 - debug) && plot_training(xs, chi, target) |> display

    return target
end

function isotarget_isa(model, xs, ys; permute=true)
    cs = model(ys)
    ks = StatsBase.mean(cs[:, :, :], dims=3)[:, :, 1]
    target = K_isa(ks)

    if permute
        A = model(xs) * target'  # matrix of the inner products
        P = stableperm(A)  # permutation matrix to maximize diagonal
        target = P * target
    end

    return target
end

# there are more stable versions using QR or SVD for the application of the pseudoinv
function Kinv(chi::Matrix, kchi::Matrix; direct=true, eigenvecs=true)
    if direct
        Kinv = chi * pinv(kchi)
        e = eigen(Kinv)
        @show 1 ./ e.values
        T = eigenvecs ? inv(e.vectors) : I
        return T * Kinv * kchi
    else
        K = kchi * pinv(chi)
        e = eigen(K)
        T = eigenvecs ? inv(e.vectors) : I
        return T * inv(K) * kchi
    end
end

### Permutation - stabilization
using Combinatorics

# return the permutation matrix that maximizes the diagonal including possible sign changes
function stableperm(A)
    n = size(A, 1)
    p = argmax(permutations(1:n, n)) do p
        sum(abs.(diag(A[p, :])))
    end
    P = collect(Int, I(n)[p, :])
    for i in 1:n
        P[p[i], i] *= sign(A[i, p[i]])
    end
    P
end

### Inner simplex approch

function K_isa(ks)
    A = innersimplexalgorithm(ks')'
    #A = stableA(A)
    A * ks
end

innersimplexalgorithm(X) = inv(X[indexmap(X), :])

function indexmap(X)
    @assert size(X, 1) > size(X, 2)
    # get indices of rows of X to span the largest simplex
    rnorm(x) = sqrt.(sum(abs2.(x), dims=2)) |> vec
    ind = zeros(Int, size(X, 2))
    for j in 1:length(ind)
        rownorm = rnorm(X)
        # store largest row index
        ind[j] = argmax(rownorm)
        if j == 1
            # translate to origin
            X = X .- X[ind[1], :]'
        else
            # remove subspace
            X = X / rownorm[ind[j]]
            vt = X[ind[j], :]'
            X = X - X * vt' * vt
        end
    end
    return ind
end

### Examples

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

function plot_training(xs, chi, target)
    if size(xs, 1) == 1
        xs = vec(xs)
        s = sortperm(xs)
        scatter(xs[s], target[:, s]')
        plot!(xs[s], chi[:, s]')
    elseif size(xs, 1) == 2
        vis2d_scatter(xs, target)
    end
    plot!()
end

function vismodel(model, grd=-2:0.03:2; xs=nothing, ys=nothing, markeralpha=0.1, markersize=1, float=0.01, kwargs...)
    dim = inputdim(model)
    if dim == 1
        plot(grd, model(collect(grd)')')
    elseif dim == 2
        p = plot()
        for i in 1:outputdim(model)
            surface!(p, grd, grd, (x, y) -> model([x, y])[i])
        end
        if !isnothing(ys)
            yy = reshape(ys, 2, :)
            scatter!(eachrow(yy)..., maximum(model(yy), dims=1) .+ float |> vec; markeralpha, markersize, markercolor=:blue, kwargs...)
        end
        !isnothing(xs) && scatter!(eachrow(xs)..., maximum(model(xs), dims=1) .+ float |> vec; markeralpha, markersize, markercolor=:red, kwargs...)
        plot!()
    end
end

function vis2d_scatter(xs, fxs)
    plot()
    for (i, f) in enumerate(eachrow(fxs))
        scatter!(eachrow(xs)..., marker_z=f, markersize=10 - 2 * i, markerstrokewidth=0)
    end
    plot!()
end


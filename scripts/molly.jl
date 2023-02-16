using ISOKANN:
    PDB_6MRR,
    defaultmodel,
    randx0,
    generatedata,
    isokann,
    plotlossdata,
    stratified_x0,
    bootstrapx0

using ISOKANN
using Optimisers
using Plots
import Flux
using JLD2
import StatsBase, Zygote, Optimisers, ISOKANN


# Adhoc constructor for the experiment, stored in a named tuple

function setup(;
    nd = 100, # number of outer iterations
    nx = 100, # size of data subsamples
    np = 1, # number of power iterates
    nl = 20, # number of learning steps  #merely >1 for performance reasons

    nres = Inf, # how often to resample
    ny = 8, # number of new x samples to generate
    nk = 16, # number of koop samples

    batchsize = Inf, # minibatchsize for sgd

    sys = PDB_ACEMD(),

    model = defaultmodel(sys),
    opt = Adam(1e-3),

    data = nothing,
    losses = Float64[])

    isa(opt, Optimisers.AbstractRule) && (opt = Optimisers.setup(opt, model))

    if isnothing(data)
        x0 = bootstrapx0(sys, ny)
        @time data = generatedata(sys, nk, x0)
    end

    return (; nd, nx, ny, nk, np, nl, sys, model, opt, data, losses, batchsize, nres)
end


## Reimplementation of ISOKANN


runloop(setup::NamedTuple = setup(); kwargs...) = runloop(;setup..., kwargs...)

function runloop(; nd, nx, ny, nk, np, nl, sys, model, opt, data, losses, batchsize, nres, plotevery=10, kwargs...)
    isa(opt, Optimisers.AbstractRule) && (opt = Optimisers.setup(opt, model))
    datastats(data)

    local sdata
    plt = Flux.throttle(plotevery) do
        plot_learning(losses, data, model) |> display
        savefig("out/lastplot.png")
    end

    for j in 1:nd
        sdata = datasubsample(model, data, nx)
        for i in 1:np
            # train model(xs) = target
            xs, target = koopman(sdata, model)
            for i in 1:nl
                l = learnbatch!(model, xs, target, opt, batchsize)
                push!(losses, l)
            end
        end

        plt()

        if j%nres == 0
            @time data = adddata(data, model, sys, ny)
            extractdata(data[1], model, sys.sys)
        end
    end

    return (; nd, nx, ny, nk, np, nl, sys, model, opt, data, losses, batchsize,nres)
end


function koopman(data, model)
    xs, ys = data
    cs = model(ys) :: Array{<:Number, 3}
    ks = vec(StatsBase.mean(cs[1,:,:], dims=2)) :: Vector
    target = ((ks .- minimum(ks)) ./ (maximum(ks) - minimum(ks))) :: Vector
    return xs, target
end

""" batched supervised learning for a given batchsize """
function learnbatch!(model, xs::Matrix, target::Vector, opt, batchsize)
    ndata = length(target)
    if ndata <= batchsize
        return learnstep!(model, xs, target, opt)
    end

    nbatches = ceil(Int, ndata / batchsize)
    l = 0.
    for batch in 1:nbatches
        ind = ((batch-1)*batchsize+1):min(batch*batchsize, ndata)
        l += learnstep!(model, xs[:, ind], target[ind], opt)
    end
    return l
end

""" single supervised learning step """
function learnstep!(model, xs, target, opt)
    l, grad = let xs=xs  # `let` allows xs to not be boxed
        Zygote.withgradient(model) do model
            sum(abs2, (model(xs)|>vec) .- target) / length(target)
        end
    end
    Optimisers.update!(opt, model, grad[1])
    return l
end


## DATA MANGLING

function datasubsample(model, data, nx)
    # chi stratified subsampling
    xs, ys = data
    ks = ISOKANN.shiftscale(model(xs) |> vec)
    ix = ISOKANN.subsample_uniformgrid(ks, nx)
    xs = xs[:,ix]
    ys = ys[:,ix,:]

    return xs, ys
end


function adddata(data, model, sys, ny, lastonly = false)
    _, ys = data
    nk = size(ys, 3)
    if lastonly
        x0 = stratified_x0(model, ys[:, end-ny+1:end, :], ny)
    else
        x0 = stratified_x0(model, ys, ny)
    end
    ndata = generatedata(sys, nk, x0)
    data = hcat.(data, ndata)

    datastats(data)
    return data
end


function datastats(data)
    xs, ys = data
    ext = extrema(xs[1,:])
    uni = length(unique(xs[1,:]))
    _, n, ks = size(ys)
    println("Dataset has $n entries ($uni unique) with $ks koop's. Extrema: $ext")
end


""" save data into a pdb file sorted by model evaluation """
function extractdata(data::AbstractArray, model, sys, path="out/data.pdb")
    dd = data
    dd = reshape(dd, size(dd, 1), :)
    ks = model(dd)
    i = sortperm(vec(ks))
    dd = dd[:, i]
    i = uniqueidx(dd[1,:] |> vec)
    dd = dd[:, i]
    ISOKANN.exportdata(sys, path, dd)
    dd
end

uniqueidx(v) = unique(i -> v[i], eachindex(v))


## Plotting

function plot_learning(losses, data, model)
    p1 = plot(losses, yaxis=:log, title="loss")
    p2 = plot(); plotatoms!(data..., model)
    p3 = scatter_ramachandran(data[1], model)
    plot(p1, p2, p3, layout=(3,1), size=(600,1000))
end

function scatter_ramachandran(x::Matrix, model=nothing)
    z = nothing
    !isnothing(model) && (z = model(x) |> vec)
    ph = phi(x)
    ps = psi(x)
    scatter(ph, ps, marker_z=z, xlabel="\\phi", ylabel="\\psi", title="Ramachandran")
end


function psi(x::AbstractVector)  # dihedral of the oxygens
    x = reshape(x, 3, :)
    @views ISOKANN.dihedral(x[:, [7,9,15,17]])
end

function phi(x::AbstractVector)
    x = reshape(x, 3, :)
    @views ISOKANN.dihedral(x[:, [5,7,9,15]])
end

phi(x::Matrix) = mapslices(phi, x, dims=1) |> vec  # is this a candidate for flatmap?
psi(x::Matrix) = mapslices(psi, x, dims=1) |> vec

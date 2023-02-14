using ISOKANN:
    PDB_6MRR,
    defaultmodel,
    randx0,
    generatedata,
    isokann,
    plotlossdata,
    stratified_x0,
    bootstrapx0

using Optimisers: Adam
using Plots
import Flux
using JLD2

function setup(;
    nd = 200, # number of outer iterations
    nx = 100, # size of data subsamples
    np = 2, # number of power iterates
    nl = 2, # number of learning steps  #merely >1 for performance reasons

    nres = 40, # how often to resample
    ny = 8, # number of new x samples to generate
    nk = 8, # number of koop samples

    batchsize = Inf, # minibatchsize for sgd

    sys = PDB_6MRR(),
    dt = 2e-5,

    model = defaultmodel(sys),
    opt = Adam(1e-4),

    data = nothing,
    losses = Float64[])

    isa(opt, Optimisers.AbstractRule) && (opt = Optimisers.setup(opt, model))

    if isnothing(data)
        x0 = bootstrapx0(sys, ny; dt)
        @time data = generatedata(sys, nk; dt, x0)
    end


    return (; nd, nx, ny, nk, np, nl, dt, sys, model, opt, data, losses, batchsize, nres)
end

runloop(setup::NamedTuple = setup(); kwargs...) = runloop(;setup..., kwargs...)

function runloop(; nd, nx, ny, nk, np, nl, dt, sys, model, opt, data, losses, batchsize, nres)
    isa(opt, Optimisers.AbstractRule) && (opt = Optimisers.setup(opt, model))
    datastats(data)

    local sdata
    plt = Flux.throttle(1) do
        plotlossdata(losses, sdata, model) |> display
        xs, ys = sdata
        #println("Extrema: $(model(xs) |> extrema), $(model(ys)|>extrema)")
        #savefig("lastplot.png")
    end

    @time for j in 1:nd
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
            data = @time adddata(data, model, sys, ny, dt)
        end
    end

    return (; nd, nx, ny, nk, np, nl, dt, sys, model, opt, data, losses, batchsize,nres)
end


import StatsBase, Zygote, Optimisers, ISOKANN

function datasubsample(model, data, nx)
    # chi stratified subsampling
    xs, ys = data
    ks = ISOKANN.shiftscale(model(xs) |> vec)
    ix = ISOKANN.subsample_uniformgrid(ks, nx)
    xs = xs[:,ix]
    ys = ys[:,ix,:]

    return xs, ys
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


""" subsample data=(x,y) uniformly in χ(x) """
function retainolddata(data, model, nx)
    xs, ys = data
    ks = ISOKANN.shiftscale(model(xs) |> vec)
    ix = ISOKANN.subsample_uniformgrid(ks, nx)

    return xs[:,ix], ys[:,ix,:]
end


function adddata(data, model, sys, ny, dt, lastonly = false)
    _, ys = data
    nk = size(ys, 3)
    if lastonly
        x0 = stratified_x0(model, ys[:, end-ny+1:end, :], ny)
    else
        x0 = stratified_x0(model, ys, ny)
    end
    ndata = generatedata(sys, nk; dt, x0)
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

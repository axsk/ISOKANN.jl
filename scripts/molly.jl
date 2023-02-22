using ISOKANN:
    PDB_6MRR,
    defaultmodel,
    randx0,
    generatedata,
    isokann,
    plotlossdata,
    stratified_x0,
    bootstrapx0,
    pairnet,
    threadpairdists

using ISOKANN
using Optimisers
using Plots
using Parameters
import Flux
using JLD2
import StatsBase, Zygote, Optimisers, ISOKANN



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
            # sum(abs2, (model(threadpairdists(xs))|>vec) .- target) / length(target)
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
    cs = model(xs) |> vec
    ks = ISOKANN.shiftscale(cs)
    ix = ISOKANN.subsample_uniformgrid(ks, nx)
    xs = xs[:,ix]
    ys = ys[:,ix,:]

    return xs, ys
end


function adddata(data, model, sys, ny, lastn = Inf)
    _, ys = data
    nk = size(ys, 3)
    firstind = max(size(ys, 2) - lastn + 1, 1)
    x0 = @views stratified_x0(model, ys[:, firstind:end, :], ny)
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

function plot_learning(losses, data, model; maxhist=100_000)
    i = max(1, length(losses)-maxhist)
    p1 = plot(losses[i:end], yaxis=:log, title="loss")
    #p2 = plot(); plotatoms!(data..., model)
    p3 = scatter_chifix(data, model)
    p2 = scatter_ramachandran(data[1], model)
    ps = [p1,p2,p3]
    plot(ps..., layout=(length(ps),1), size=(600,300*length(ps)))
end





## Reimplementation of ISOKANN

abstract type ISOSim end

ISOBench() = @time ISOSim1(nd=1, nx=100, np=1, nl=300, nres=Inf, ny=100)
ISOLong() = ISOSim1(nd=1_000_000, nx=200, np=10, nl=10, nres=200, ny=8, opt=Adam(1e-5))

function isobench()
    @time iso = ISOBench()
    @time run(iso)
end

function isosave()
    iso = ISOSim()
    run(iso)
    savefig("out/lastplot.png")
end

ISOSim(;kwargs...) = ISOSim1(;kwargs...)

@with_kw mutable struct ISOSim1  # takes 10 min
    nd = 1000
    nx = 100
    np = 1
    nl = 5

    nres = 10
    ny = 8
    nk = 8
    sys = PDB_ACEMD()
    model = pairnet(sys.sys)
    opt = OptimiserChain(Optimisers.WeightDecay(1e-4), Adam(1e-4))

    data = generatedata(sys, nk, bootstrapx0(sys, ny))
    losses = Float64[]
end

function run(iso::ISOSim1; plotevery = 5, batchsize=Inf)
    isa(iso.opt, Optimisers.AbstractRule) && (iso.opt = Optimisers.setup(iso.opt, iso.model))
    (; nd, nx, ny, nk, np, nl, sys, model, opt, data, losses, nres) = iso
    datastats(data)

    local sdata
    plt = Flux.throttle(plotevery) do
        plot_learning(losses, sdata,model) |> display
        #savefig("out/lastplot.png")
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
            @time data = adddata(data, model, sys, ny, ny)
            if size(data[1], 2) > 1000
                data = datasubsample(model, data, 500)
            end
            iso.data = data

            #extractdata(data[1], model, sys.sys)
        end
    end

    return iso
end


function scatter_chifix(data, model)
    xs, target = koopman(data, model)
    xs = model(xs)|>vec
    scatter(xs, target, markersize=2, xlabel = "model", ylabel="target")
    plot!([0, 1], [0,1], legend=false)
end

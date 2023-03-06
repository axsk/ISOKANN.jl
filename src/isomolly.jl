import StatsBase, Zygote, Optimisers, Flux, JLD2
export ISORun, run!

abstract type ISORun end

ISORun(;kwargs...) = ISO_ACEMD(;kwargs...)

Base.@kwdef mutable struct ISO_ACEMD <: ISORun # takes 10 min
    nd = 1000 # number of outer datasubsampling steps
    nx = 100  # size of subdata set
    np = 1    # number of poweriterations with the same subdata
    nl = 5    # number of weight updates  with the same poweriteration step

    nres = 10  # resample new data every n outer steps
    ny = 8     # number of new points to sample
    nk = 8     # number of koopman points to sample
    sim = PDB_ACEMD()
    model = pairnet(sim)
    opt = Optimisers.OptimiserChain(Optimisers.WeightDecay(1e-4), Optimisers.Adam(1e-4))

    data = bootstrap(sim, ny, nk)
    losses = Float64[]
end

function run!(iso::ISORun; callback = Flux.throttle(plotcallback, 5), batchsize=Inf)
    isa(iso.opt, Optimisers.AbstractRule) && (iso.opt = Optimisers.setup(iso.opt, iso.model))
    (; nd, nx, ny, nk, np, nl, sim, model, opt, data, losses, nres) = iso
    datastats(data)

    local subdata

    @time for j in 1:nd
        subdata = datasubsample(model, data, nx)
        # train model(xs) = target
        for i in 1:np
            xs, ys = subdata
            target = shiftscale(koopman(model, ys))
            for i in 1:nl
                l = learnbatch!(model, xs, target, opt, batchsize)
                push!(losses, l)
            end
        end

        callback(;model, losses, subdata, data)

        if j%nres == 0
            @time data = adddata(data, model, sim, ny)
            if size(data[1], 2) > 3000
                data = datasubsample(model, data, 1000)
            end
            iso.data = data

            #extractdata(data[1], model, sim.sys)
        end
    end

    return iso
end

function plotcallback(;losses, subdata, model, kwargs...)
    p = plot_learning(losses, subdata,model)
    try display(p) catch e;
        @warn "could not print ($e)"
    end
end

""" evluation of koopman by shiftscale(mean(model(data))) on the data """
function koopman(model, ys)
    cs = model(ys) :: Array{<:Number, 3}
    ks = vec(StatsBase.mean(cs[1,:,:], dims=2)) :: Vector
    return ks
end

""" empirical shift-scale operation """
shiftscale(ks) = (ks .- minimum(ks)) ./ (maximum(ks) - minimum(ks))

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
# TODO: better interface

function generatedata(ms, x0, ny)
    ys = propagate(ms, x0, ny)
    return center(x0), center(ys)
end

""" compute initial data by propagating the molecules initial state
to obtain the xs and propagating them further for the ys """
function bootstrap(sim::IsoSimulation, nx, ny)
    x0 = reshape(getcoords(sim), :, 1)
    xs = reshape(propagate(sim, x0, nx), :, nx)
    ys = propagate(sim, xs, ny)
    center(xs), center(ys)
end

function datasubsample(model, data, nx)
    # chi stratified subsampling
    xs, ys = data
    cs = model(xs) |> vec
    ks = shiftscale(cs)
    ix = ISOKANN.subsample_uniformgrid(ks, nx)
    xs = xs[:,ix]
    ys = ys[:,ix,:]

    return xs, ys
end

function adddata(data, model, sim::IsoSimulation, ny, lastn = 1_000_000)
    _, ys = data
    nk = size(ys, 3)
    firstind = max(size(ys, 2) - lastn + 1, 1)
    x0 = @views stratified_x0(model, ys[:, firstind:end, :], ny)
    ys = propagate(sim, x0, nk)
    ndata = center(x0), center(ys)
    data = hcat.(data, ndata)

    datastats(data)
    return data
end

""" given an array of states, return a chi stratified subsample """
function stratified_x0(model, ys, n)
    ys = reshape(ys, size(ys,1), :)
    ks = shiftscale(model(ys) |> vec)

    i = subsample_uniformgrid(ks, n)
    xs = ys[:, i]
    return xs
end

function datastats(data)
    xs, ys = data
    ext = extrema(xs[1,:])
    uni = length(unique(xs[1,:]))
    _, n, ks = size(ys)
    println("Dataset has $n entries ($uni unique) with $ks koop's. Extrema: $ext")
end

""" save data into a pdb file sorted by model evaluation """
function extractdata(data::AbstractArray, model, sim, path="out/data.pdb")
    dd = data
    dd = reshape(dd, size(dd, 1), :)
    ks = model(dd)
    i = sortperm(vec(ks))
    dd = dd[:, i]
    i = uniqueidx(dd[1,:] |> vec)
    dd = dd[:, i]
    dd = standardform(dd)
    ISOKANN.exportdata(sim, path, dd)
    dd
end

uniqueidx(v) = unique(i -> v[i], eachindex(v))


# default setups

ISOBench() = @time ISORun1(nd=1, nx=100, np=1, nl=300, nres=Inf, ny=100)
ISOLong() = ISORun1(nd=1_000_000, nx=200, np=10, nl=10, nres=200, ny=8, opt=Adam(1e-5))

function isobench()
    @time iso = ISOBench()
    @time run(iso)
end

function isosave()
    iso = ISORun()
    run(iso)
    savefig("out/lastplot.png")
end

function save(iso::ISORun, pathlength=300)
    (; model, losses, data, sim) = iso
    xs, ys = data
    zs = standardform(stratified_x0(model, xs, pathlength))
    savecoords(sim, zs, path="out/latest/path.pdb")
    savefig(plot_learning(losses, data, model), "out/latest/learning.png")

    JLD2.save("out/latest/iso.jld2", "iso", iso)



end

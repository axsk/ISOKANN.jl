import StatsBase, Zygote, Optimisers, Flux, JLD2

using LsqFit
using ProgressMeter

export IsoRun, run!, Adam, AdamRegularized

abstract type IsoRun end

run(;kwargs...) = run!(IsoRun(;kwargs...))

IsoRun(;kwargs...) = ISO_ACEMD5(;kwargs...)

Base.@kwdef mutable struct ISO_ACEMD5 <: IsoRun # takes 10 min
    nd::Integer = 1000 # number of outer datasubsampling steps
    nx::Integer = 100  # size of subdata set
    np::Integer = 2    # number of poweriterations with the same subdata
    nl::Integer = 5    # number of weight updates  with the same poweriteration step

    nres::Integer = 50  # resample new data every n outer steps
    ny::Integer = 8     # number of new points to sample
    nk::Integer = 8     # number of koopman points to sample
    nxmax::Integer = 0  # maximal number of x data points
    sim = MollyLangevin(sys=PDB_ACEMD())
    model = pairnet(sim)
    opt = AdamRegularized()
    minibatch = Inf

    data::Tuple = bootstrap(sim, ny, nk)
    losses = Float64[]
    loggers::Vector = Any[plotcallback(10)]
end

Regularized(opt, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), opt)

AdamRegularized(adam=1e-3, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), Optimisers.Adam(adam))

optparms(iso::IsoRun) = optparms(iso.opt.layers[2].bias.rule)
optparms(o::Optimisers.OptimiserChain) = map(optparms, o.opts)
optparms(o::Optimisers.WeightDecay) = (;WeightDecay = o.gamma)
optparms(o::Optimisers.Adam) = (;Adam = o.eta)

function run!(iso::IsoRun)
    isa(iso.opt, Optimisers.AbstractRule) && (iso.opt = Optimisers.setup(iso.opt, iso.model))
    (; nd, nx, ny, nk, np, nl, sim, model, opt, data, losses, nres, minibatch, loggers, nxmax) = iso
    #datastats(data)

    local subdata

    p = Progress(nd, 1, offset=1)

    for j in 1:nd
        subdata = datasubsample(model, data, nx)
        # train model(xs) = target
        for i in 1:np
            xs, ys = subdata
            ks = koopman(model, ys)
            target = shiftscale(ks)
            # target = gettarget(xs, ys, model)
            for i in 1:nl
                ls = learnbatch!(model, xs, target, opt, minibatch)
                push!(losses, ls)
            end
        end

        for logger in loggers
            log(logger; model, losses, subdata, data, j, iso)
        end

        if nres > 0 && j%nres == 0 && (nxmax == 0 || size(data[1], 2) < nxmax)
            data = adddata(data, model, sim, ny)
            #if size(data[1], 2) > 3000
            #    data = datasubsample(model, data, 1000)
            #end
            iso.data = data

            #extractdata(data[1], model, sim.sys)
        end

        ProgressMeter.next!(p; showvalues = ()->[(:loss, losses[end]), (:n, length(losses)), (:data, size(data[2]))])
    end

    return iso
end

# note there is also plot_callback in isokann.jl
function plotcallback(secs=10)
    Flux.throttle(
        function plotcallback(;iso, subdata, kwargs...)
            p = plot_learning(iso; subdata)
            try display(p) catch e;
                @warn "could not print ($e)"
            end
        end
        , secs)
end

""" evluation of koopman by shiftscale(mean(model(data))) on the data """
function koopman(model, ys)
    #ys = Array(ys)
    cs = model(ys) :: AbstractArray{<:Number, 3}
    ks = vec(StatsBase.mean(cs[1,:,:], dims=2)) :: AbstractVector
    return ks
end

""" empirical shift-scale operation """
shiftscale(ks) = (ks .- minimum(ks)) ./ (maximum(ks) - minimum(ks))

""" batched supervised learning for a given batchsize """
function learnbatch!(model, xs::AbstractMatrix, target::AbstractVector, opt, batchsize)
    ndata = length(target)
    if ndata <= batchsize
        return learnstep!(model, xs, target, opt)
    end

    nbatches = ceil(Int, ndata / batchsize)
    l = 0.
    for batch in 1:nbatches
        ind = ((batch-1)*batchsize+1):min(batch*batchsize, ndata)
        l += @views learnstep!(model, xs[:, ind], target[ind], opt)
    end
    return l / nbatches  # this is only approx correct for uneven batches
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
    size(xs,2) <= nx || nx == 0 && return data
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
    x0 = stratified_x0(model, ys[:, firstind:end, :], ny)
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
    #println("\n Dataset has $n entries ($uni unique) with $ks koop's. Extrema: $ext")
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

IsoBench() = @time IsoRun1(nd=1, nx=100, np=1, nl=300, nres=Inf, ny=100)
IsoLong() = IsoRun1(nd=1_000_000, nx=200, np=10, nl=10, nres=200, ny=8, opt=Adam(1e-5))

function saveall(iso::IsoRun, pathlength=300)
    mkpath("out/latest")
    (; model, losses, data, sim) = iso
    xs, ys = data
    zs = standardform(stratified_x0(model, xs, pathlength))
    savecoords(sim, zs, path="out/latest/path.pdb")
    savefig(plot_learning(losses, data, model), "out/latest/learning.png")

    JLD2.save("out/latest/iso.jld2", "iso", iso)
end


function estimate_K(x, Kx)
    @. Kinv(Kx, p) = p[1]^-1 * (Kx .- (1-p[1]) * p[2])
    fit = curve_fit(Kinv, vec(x), vec(Kx), [.5, 1])
    lambda, a = coef(fit)
end

function gettarget(xs, ys, model)
    ks = koopman(model, ys)
    lambda, a = estimate_K(model(xs), ks)
    @show lambda, a
    target = (ks .- ((1-lambda)*a)) ./ lambda
end

function Base.show(io::IO, mime::MIME"text/plain", iso::IsoRun)
    println(io, typeof(iso), ":")
    show(io, mime, iso.sim)
    println(io, " nd=$(iso.nd), np=$(iso.np), nl=$(iso.nl), nres=$(iso.nres), minibatch=$(iso.minibatch)")
    println(io, " nx=$(iso.nx), ny=$(iso.ny), nk=$(iso.nk)")
    println(io, " model: $(iso.model.layers)")
    println(io, " opt: $(optimizerstring(iso.opt))")
    println(io, " loggers: $(length(iso.loggers))")
    println(io, " data: $(size.(iso.data))")
    length(iso.losses)>0 && println(io, " loss: $(iso.losses[end]) (length: $(length(iso.losses)))")
end

optimizerstring(opt) = typeof(opt)
optimizerstring(opt::NamedTuple) = opt.layers[end-1].weight.rule

function autotune!(iso::IsoRun, targets=[4,1,1,4])
    (; nd, nx, ny, nk, np, nl, sim, model, opt, data, losses, nres) = iso
    tdata = @elapsed adddata(data, model, sim, ny)
    tsubdata = @elapsed (subdata = datasubsample(model, data, nx))
    xs, ys = subdata
    ttarget = @elapsed target = shiftscale(koopman(model, ys))
    ttrain = @elapsed learnbatch!(model, xs, target, opt, Inf)


    nl = round(Int, ttarget/ttrain   * targets[4]/targets[3])
    np = max(1, round(Int, tsubdata/ttarget * targets[3]/targets[2]))
    nres = round(Int, tdata/(tsubdata + np * ttarget + np*nl*ttrain) * sum(targets[2:end])/targets[1])

    iso.nl = nl
    iso.np = np
    iso.nres = nres

    return (;tdata, tsubdata, ttarget, ttrain), (;nl, np, nres)
end

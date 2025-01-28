
"""
    struct IsoRun{T}

The `IsoRun` struct represents a configuration for running the ISOKANN algorithm with adaptive sampling.

The whole algorithm consists of three nested loop

1. `nd` iterations of the data loop where `nx` points are subsampled (via stratified χ-subsampling)  from the pool of all available data
2. `np` iterations of the power iteration where the training target is determined with the current model and subdata
3. `nl` iterations of the SGD updates to the neural network model to learn the current target

On initialization it samples `ny` starting positions with `nk` Koopman samples each.
Furthermore if `nres` > 0 it samples `ny` new data points adaptively starting from χ-sampled positions every `nres` steps in the data loop.

The `sim` field takes any simulation object that implements the data sampling interface (mainly the `propagate` method, see data.jl),
usually a `MollyLangevin` simulation.

`model` and `opt` store the neural network model and the optimizert (defaulting to a `pairnet` and `AdamRegularized`).

`data` contains the training data and is by default constructed using the `bootstrap` method.

The vector `losses` keeps track of the training loss and `loggers` allows to pass in logging functions which are executed in the power iteration loop.

To start the actual training call the `run!` method.

# Fields
- `nd::Int64`: Number of outer data subsampling steps.
- `nx::Int64`: Size of subdata set.
- `np::Int64`: Number of power iterations with the same subdata.
- `nl::Int64`: Number of weight updates with the same power iteration step.
- `nres::Int64`: Resample new data every n outer steps.
- `ny::Int64`: Number of new points to sample.
- `nk::Int64`: Number of Koopman points to sample.
- `nxmax::Int64`: Maximal number of x data points.
- `sim`: Simulation object.
- `model`: Model object.
- `opt`: Optimization algorithm.
- `data::T`: Data object.
- `losses`: Vector to store loss values.
- `loggers::Vector`: Vector of loggers.
- `minibatch::Int`: Size of the (shuffled) minibatches. Set to 0 to disable

"""
Base.@kwdef mutable struct IsoRun{T} # takes 10 min
    nd::Int64 = 1000 # number of outer datasubsampling steps
    nx::Int64 = 100  # size of subdata set
    np::Int64 = 2    # number of poweriterations with the same subdata
    nl::Int64 = 5    # number of weight updates  with the same poweriteration step

    nres::Int64 = 50  # resample new data every n outer steps
    ny::Int64 = 8     # number of new points to sample
    nk::Int64 = 8     # number of koopman points to sample
    nxmax::Int64 = 0  # maximal number of x data points
    sim = MollyLangevin(sys=PDB_ACEMD())
    model = pairnet(sim)
    opt = AdamRegularized()
    data::T = bootstrap(sim, ny, nk)
    losses = Float64[]
    loggers::Vector = Any[]
    minibatch::Int = 128
end

# TODO: should make these defaults for sim==nothing
IsoRunFixedData(; data, kwargs...) = ISOKANN.IsoRun(;
    data=data,
    model=ISOKANN.pairnet(data),
    nd=1,
    minibatch=0,
    nx=0, # no chi subsampling,
    nres=0, # no resampling,
    np=1, # power iterations,
    nl=1, # weight updates,
    nk=0,
    ny=0,
    sim=nothing, kwargs...)

optparms(iso::IsoRun) = optparms(iso.opt.layers[2].bias.rule)
optparms(o::Optimisers.OptimiserChain) = map(optparms, o.opts)
optparms(o::Optimisers.WeightDecay) = (; WeightDecay=o.gamma)
optparms(o::Optimisers.Adam) = (; Adam=o.eta)

function Optimisers.setup(iso::IsoRun)

    iso.opt = if isa(iso.opt, Optimisers.AbstractRule)
        Optimisers.setup(iso.opt, iso.model)
    else
        Optimisers.setup(iso.opt.layers[end].bias.rule, iso.model)
    end
end

Base.extrema(iso::IsoRun) = extrema(chis(iso))
chis(iso::IsoRun) = iso.model(features(iso.data))

""" run the given `IsoRun` object """
function run!(iso::IsoRun; showprogress=true)
    isa(iso.opt, Optimisers.AbstractRule) && (iso.opt = Optimisers.setup(iso.opt, iso.model))
    (; nd, nx, ny, nk, np, nl, sim, model, opt, data, losses, nres, loggers, nxmax, minibatch) = iso
    #datastats(data)

    local subdata

    p = ProgressMeter.Progress(nd)

    t_samp = t_koop = t_train = 0.0

    for j in 1:nd
        if 0 < nx < size(data[1], 2)
            t_samp += @elapsed subdata = subsample(model, data, nx)
        else
            subdata = data
        end
        # train model(xs) = target
        xs, ys = subdata
        local ks
        for i in 1:np
            t_koop += @elapsed ks = koopman(model, ys) |> vec
            target = shiftscale(ks)
            # target = gettarget(xs, ys, model)
            t_train += @elapsed for i in 1:nl
                ls = train_batch!(model, xs, target, opt, minibatch)
                push!(losses, ls)
            end
        end

        for logger in loggers
            log(logger; model, losses, subdata, data, j, iso)
        end

        if nres > 0 && j % nres == 0 && (nxmax == 0 || size(data[1], 2) < nxmax)
            t_samp += @elapsed data = adddata(data, model, sim, ny)
            #if size(data[1], 2) > 3000
            #    data = datasubsample(model, data, 1000)  # keep the data small
            #end
            iso.data = getobs(data)

            #extractdata(data[1], model, sim.sys)
        end

        showprogress && ProgressMeter.next!(p; showvalues=() -> [
            (:loss, losses[end]),
            (:n, length(losses)),
            (:data, size(data[2])),
            #(:rate, chi_exit_rate(cpu(model(xs)), cpu(ks), 1),
        ])
    end

    @show t_samp, t_koop, t_train
    return iso
end

isotarget(iso::IsoRun, ys=iso.data[2]) = shiftscale(koopman(iso.model, ys)) |> vec



function RateLogger()
    rates = Float64[]
    call = (; iso, kwargs...) -> push!(rates, chi_exit_rate(iso, 1))
    plot = () -> Plots.plot(rates, legend=false, title="exit rate")
    (; call, plot, rates)
end







"""
    scaleandshift(model, xs, ys)

Estimate the scale and shift parameters using linear regression
"""
function scaleandshift(model, xs, ys)
    a = model(xs) |> vec
    b = koopman(model, ys) |> vec
    o = ones(length(a))
    [a o] \ [b o][:, 1]
end

scaleandshift(iso::IsoRun) = scaleandshift(iso.model, iso.data...)



""" DEPRECATED - batched supervised learning for a given batchsize """
function train_batch!(model, xs::AbstractMatrix, target::AbstractArray, opt, batchsize)
    ndata = numobs(xs)

    (0 < batchsize < ndata) || return train_step!(model, xs, target, opt)

    l = sum(Flux.DataLoader((xs, target); batchsize, shuffle=true)) do (xs, target)
        train_step!(model, xs, target, opt) * numobs(xs)
    end
    return l / ndata
end

""" single supervised learning step """
function train_step!(model, xs::AbstractMatrix, target::AbstractMatrix, opt)
    n = numobs(xs)
    l, grad = let xs = xs  # `let` allows xs to not be boxed
        Zygote.withgradient(model) do model
            sum(abs2, model(xs) .- target) / n
        end
    end
    Optimisers.update!(opt, model, grad[1])
    return l
end

train_step!(model, xs, target::AbstractVector, opt) = train_step!(model, xs, target', opt)

## DATA MANGLING
# TODO: better interface

# default setups

IsoBench() = @time IsoRun(nd=1, nx=100, np=1, nl=300, nres=Inf, ny=100)
IsoLong() = IsoRun(nd=1_000_000, nx=200, np=10, nl=10, nres=200, ny=8, opt=AdamRegularized(1e-5))

function saveall(iso::IsoRun, pathlength=300)
    mkpath("out/latest")
    (; model, losses, data, sim) = iso
    xs, ys = data
    zs = standardform(subsample(model, xs, pathlength))
    savecoords("out/latest/path.pdb", sim, zs)
    savefig(plot_training(iso), "out/latest/learning.png")

    JLD2.save("out/latest/iso.jld2", "iso", iso)
end


# TODO: check if this is consistent with scaleandshift
function estimate_K(x, Kx)
    @. Kinv(Kx, p) = p[1]^-1 * (Kx .- (1 - p[1]) * p[2])  # define the parametric inverse of K
    fit = LsqFit.curve_fit(Kinv, vec(x), vec(Kx), [0.5, 1])     # lsq regression
    lambda, a = LsqFit.coef(fit)
end



# TODO: remove, not used anywhere
function gettarget(xs, ys, model)
    ks = koopman(model, ys)
    lambda, a = estimate_K(model(xs), ks)
    @show lambda, a
    target = (ks .- ((1 - lambda) * a)) ./ lambda
end

function Base.show(io::IO, mime::MIME"text/plain", iso::IsoRun)
    println(io, typeof(iso), ":")
    show(io, mime, iso.sim)
    println(io, " nd=$(iso.nd), np=$(iso.np), nl=$(iso.nl), nres=$(iso.nres)")
    println(io, " nx=$(iso.nx), ny=$(iso.ny), nk=$(iso.nk)")
    println(io, " model: $(iso.model.layers)")
    println(io, " opt: $(optimizerstring(iso.opt))")
    println(io, " minibatch: $(iso.minibatch)")
    println(io, " loggers: $(length(iso.loggers))")
    println(io, " data: $(size.(iso.data))")
    length(iso.losses) > 0 && println(io, " loss: $(iso.losses[end]) (length: $(length(iso.losses)))")
end



function autotune!(iso::IsoRun, targets=[4, 1, 1, 4])
    (; nd, nx, ny, nk, np, nl, sim, model, opt, data, losses, nres) = iso
    tdata = nres > 0 ? (@elapsed adddata(data, model, sim, ny)) : 0
    tsubdata = @elapsed (subdata = subsample(model, data, nx))
    xs, ys = subdata
    ttarget = @elapsed target = shiftscale(koopman(model, ys))
    ttrain = @elapsed train_batch!(model, xs, target, opt, Inf)


    nl = ceil(Int, ttarget / ttrain * targets[4] / targets[3])
    np = max(1, round(Int, tsubdata / ttarget * targets[3] / targets[2]))
    nres = round(Int, tdata / (tsubdata + np * ttarget + np * nl * ttrain) * sum(targets[2:end]) / targets[1])

    iso.nl = nl
    iso.np = np
    iso.nres = nres

    return (; tdata, tsubdata, ttarget, ttrain), (; nl, np, nres)
end

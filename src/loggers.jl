Base.@kwdef mutable struct TrainlossLogger7
    data
    every = 100
    losses = Float64[]
    xs = Int[]
end

TrainlossLogger = TrainlossLogger7

function log(l::TrainlossLogger; model, losses, j, kwargs...)
    if j % l.every == 0
        push!(l.xs, length(losses))
        push!(l.losses, loss(model, l.data))
    end
end


function add_validationloss!(iso::IsoRun, nx=100, ny=100, every=100)
    xs, _ = iso.data
    i = rand(1:size(xs, 2), nx)
    xs = xs[:, i]
    ys = propagate(iso.sim, xs, ny)
    logger = TrainlossLogger(data=(xs, ys), every=every, losses=Float64[])
    push!(iso.loggers, logger)
    logger
end


function l2diff(model1, model2, data::DataTuple)
    ks = model1(data) |> vec
    ref = model2(data) |> vec
    e = min(sum(abs2, (1 .- ks) .- ref), sum(abs2, ks .- ref)) / length(ks)
    return e
end

Base.@kwdef mutable struct EigenvalueLogger
    values = []
    every = 100
end

function log(l::EigenvalueLogger; j, model, data, kwargs...)
    j % l.every == 0 && push!(l.values, (;ev=eigenvalue(model, data), evr=eigenvalue_regression(model,data)))
end

""" estimate the eigenvalue of the chi function at the extrema of the data """
function eigenvalue(model, data)
    xs, ys = data
    cs = model(xs)
    ks = koopman(model, ys)
    λ = reduce(-, extrema(ks)) / reduce(-, extrema(cs))
    return λ
end

""" estimate the eigenvalue of the chi function by regression over all data """
function eigenvalue_regression(model, data)
    xs, ys = data
    cs = model(xs) |> vec
    ks = koopman(model, ys)
    csm = mean(cs)
    ksm = mean(ks)
    λ = sum((cs .- csm) .* (ks .- ksm)) / sum(abs2, cs .- csm)
    return λ
end


Base.@kwdef mutable struct ModelLogger
    models = []
    every = 100
end

function log(l::ModelLogger; model, j, kwargs...)
    j % l.every != 0 && push!(l.models, model)
end


function loss(model, data)
    xs, ys = data
    ks = koopman(model, ys)
    target = shiftscale(ks)
    l = mean(abs2, (model(xs) |> vec) .- target)
    return l
end


## not really used = graveyard
#=


function throttleiter(f, i)
    c = 0
    function throttled(args...;kwargs...)
        c += 1
        if c%i == 0
            c = 0
            f(args...;kwargs...)
        end
    end
    return throttled
end

function comparecallback(xs, refmodel)
    ref = refmodel(xs)
    err = Float64[]

    function compare(;model, kwargs...)
        ks = model(xs)
        e = min(sum(abs2, (1 .- ks) .- ref), sum(abs2, ks .- ref)) / length(ks)
        @show e
        push!(err, e )
    end

    return err, compare
end

function compareperformance(refmodel, refdata)
    iso = IsoRun()
    iso.nd = iso.nres

    ref = refmodel(refdata)

    err = []

    for i in 1:10
        run!(iso)
        test = iso.model(refdata)
        push!(err, sum(abs2, ref .- refdata))
    end
    return err
end

function plot_datacomparison(models, iso, refiso)
    l1=map(m->loss(m, iso.data), models)
    l2=map(m->loss(m, refiso.data), models)
    plot(l1, label="loss train")
    plot!(l2, label="loss control")
    plot!(iso.losses[1:div(length(iso.losses), length(models)):end], label="loss running")
end




function callback_eval(xs, every=100)
    ks = []
    last = 0
    function evalmodel(;model, kwargs...)
        if last % every == 0
            push!(ks, model(xs))
        end
        last+=1
    end

    return ks, evalmodel
end



=#

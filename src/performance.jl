function compareperformance(refmodel, refdata)
    iso = ISORun()
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

function comparecallback(xs, model)
    ref = model(xs)
    err = Float64[]

    function compare(;model, kwargs...)
        ks = model(xs)
        e = min(sum(abs2, (1 .- ks) .- ref), sum(abs2, ks .- ref)) / length(ks)
        @show e
        push!(err, e )
    end

    return err, compare
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

function l2diff(model1, model2, data)
    ks = model1(data)
    ref = model2(data)
    e = min(sum(abs2, (1 .- ks) .- ref), sum(abs2, ks .- ref)) / length(ks)
    return e
end

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

function multiplex(fs...)
    function multiplexed(args...; kwargs...)
        for f in fs
            f(args...; kwargs...)
        end
    end
    return multiplexed
end

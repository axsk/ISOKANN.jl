# cleaner and simpler reimplementation of ISOKANN (1)
import Flux
import StatsBase
import Optimisers
import Plots
using Plots: plot, plot!, scatter!, savefig
using Random

#@assert @elapsed isokann(Doublewell()) < 2

defaultmodel(dynamics::AbstractLangevin, layers=[5,5]) = fluxnet([dim(dynamics); layers; 1])

# 10-3 in about 30s
#isokann(Doublewell(), throttle=3, poweriter=100000, learniter=100, opt=Optimisers.Adam(0.001), dt=0.001, nx=10, nkoop=10, keepedges=true);

function isokann(;dynamics=Doublewell(), model=defaultmodel(dynamics),
                 nx::Int=10, nkoop::Int=10, poweriter::Int=100, learniter::Int=10, dt::Float64=0.01, alg=SROCK2(),
                 opt=Optimisers.Adam(0.01), keepedges::Bool=true,
                 throttle=1, callback = plot_callback,
                 usecontrol::Bool=true,
                 resample::Symbol=:humboldt
                 )

    callback_throttled = Flux.throttle(callback, throttle, leading=true, trailing=false)

    xs = randx0(dynamics, nx) :: Matrix
    sde = SDEProblem(dynamics, dt = dt, alg=alg)

    opt = Optimisers.setup(opt, model)
    stds = Float64[]
    ls = Float64[]
    local S, cde, target, std, cs
    control = nocontrol

    for i in 1:poweriter

        cde = GirsanovSDE(sde, control)

        # evaluate koopman
        ys, ws = girsanovbatch(cde, xs, nkoop) :: Tuple{Array{Float64, 3},  Array{Float64, 2}}
        cs = model(ys)
        ks, std = vec.(StatsBase.mean_and_std(cs[1,:,:].*ws, 2))

        # estimate shift scale
        S = Shiftscale(ks)
        target = invert(S, ks)
        std = std ./ exp(S.q) / sqrt(nkoop)

        # train network
        for _ in 1:learniter
            l, grad = let xs=xs  # this let allows xs to not be boxed
                Zygote.withgradient(model) do model
                    sum(abs2, (model(xs)|>vec) .- target) / length(target)
                end
            end
            Optimisers.update!(opt, model, grad[1])
            push!(ls, l)
            push!(stds, StatsBase.mean(std))
        end

        if i < poweriter
            callback_throttled(;losses=ls, model, xs, target, stds, std, cs)
        else
            plot_callback(;losses=ls, model, xs, target, stds, std, cs)
            break
        end

        # update controls
        if usecontrol
            control = optcontrol(statify(model), S, sde)
        end

        # resample xs uniformly along chi
        if resample == :humboldt
            xys = hcat(xs, reshape(ys, size(xs, 1), :))
            cs = model(xys) |> vec
            xs = humboldtsample(xys, cs, nx; keepedges)
        elseif resample == :rand
            xs = randx0(dynamics, nx)
        elseif resample == :nothing
        else
            error("resample choice ($resample) is not defined")
        end
    end
    return (;model, ls, S, sde, cde, xs, dynamics, target, stds, std, cs, opt)
end

function plot_callback(; kwargs...)
    (;losses, model, xs, target, std, stds) = NamedTuple(kwargs)

    p1 = plot_loss(losses, stds)
    p2 = plot_fit(model, xs, target, std)
    plot(p1, p2) |> display
    return p1,p2
end

function plot_loss(losses, stds)
    p=plot(yaxis=:log, title="loss", legend=:bottomleft)
    plot!(p, sqrt.(losses), label="RMSE")
    plot!(p, vec(stds), label="MSTD")
    return p
end

function plot_fit(model, xs, target, std)
    if size(xs, 1) == 1  # 1d case
        p=plot(ylims=(-.1,1.1), title="fit",  legend=:best)
        plot!(p, x->model([x])[1], -3:.1:3, label="χ")
        scatter!(p, vec(xs), vec(target), yerror=vec(std), label="SKχ")
    else
        p = contour(-2:.1:2, -2:.1:2, (x,y)->model( [x,y])[1], fill=true, alpha=.1)
        l = vec(mapslices(model, xs, dims=1)) - target
        scatter!(p, xs[:,1], xs[:,2], markersize=l.^2 * 100)
    end
    return p
end

function plot_mean_loss(rs)
    losses = StatsBase.mean(reduce(hcat, [r.ls  for r in rs]), dims=2)
    stds   = StatsBase.mean(reduce(hcat, [r.stds for r in rs]), dims=2)
    plot_loss(losses, stds)
end

function batch_analysis(;nbatch = 10, kwargs...)
    rs = [OptImpSampling.isokann(throttle=Inf, resample=:rand, poweriter=100, learniter=100, nx=10, nkoop=10, usecontrol=true) for i in 1:nbatch]
    plot_mean_loss(rs)
end

function paperplot(;seed=1, kwargs...)

    for controlled in [true, false]
        Random.seed!(seed)
        poweriter = 50
        learniter = 500
        r=isokann(
            dynamics=Doublewell(),
            nx=30,
            nkoop=20,
            poweriter=poweriter,
            learniter=learniter,
            opt=Optimisers.Adam(0.001),
            dt=0.001,
            model=fluxnet([1,5,5,1]),
            keepedges=true,
            usecontrol=controlled
            ; kwargs...)

        p1,p2 = plot_callback(;losses=r.ls, r...)
        @show log10(sqrt(r.ls[end]))
        Plots.ylims!(p1, 10^(-3.0),0.8)
        Plots.plot!(size=(300*1.6,300), title="", dpi=300)
        Plots.xticks!((0:10:poweriter)*learniter, string.(0:10:poweriter))
        Plots.ylims!(p2, -.03, 1.03)
        Plots.plot!(size=(300*1.6,300), title="", dpi=300)
        display(p1)
        display(p2)
        mkpath("plots")
        savefig(p1, "plots/loss-$controlled.png")
        savefig(p2, "plots/fit-$controlled.png")
    end
end

import Flux
import Lux
import Zygote

const DEFAULT_LAYERS = [1,3,3,1]


### FLUX
function fluxnet(layers=DEFAULT_LAYERS, act = Flux.sigmoid, lastact=act)
    Flux.Chain(
        [Flux.Dense(layers[i], layers[i+1], act) for i in 1:length(layers)-2]...,
        Flux.Dense(layers[end-1], layers[end], lastact))
end

function fluxnet1(layers=DEFAULT_LAYERS, act = Flux.sigmoid, lastact=act)
    Flux.Chain(
        [Flux.Dense(layers[i], layers[i+1], act) for i in 1:length(layers)-2]...,
        Flux.Dense(layers[end-1], layers[end], lastact), first)
end

# TODO: statify is not typestable
# we use this to create a copy which uses StaticArrays, for faster d/dx gradients
statify(x::Any) = x
statify(c::Flux.Chain) = Flux.Chain(map(statify, c.layers)...)
function statify(d::Flux.Dense)
    w = d.weight
    W = SMatrix{size(w)...}(w)
    b = d.bias
    B = SVector{length(b)}(b)
    Flux.Dense(W, B, d.σ)
end


### LUX


# multi layer perceptron for the force
function luxnet(layers=DEFAULT_LAYERS, act = Lux.sigmoid, lastact=act)
    model = Lux.Chain(
        [Lux.Dense(layers[i], layers[i+1], act) for i in 1:length(layers)-2]...,
        Lux.Dense(layers[end-1], layers[end], lastact))
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    model, ps, st
end

StatefulModel = Tuple{<:Lux.AbstractExplicitLayer, <:Any, <:NamedTuple}
((mod,ps,st)::StatefulModel)(x) = mod(x,ps,st)[1]


import SimpleChains
using SimpleChains: SimpleChain, TurboDense, init_params, static

SCModel = Tuple{<:SimpleChain, <:Any}

function scnet(layers=DEFAULT_LAYERS, act = SimpleChains.σ, lastact=act)
    schain = SimpleChain(static(layers[1]),
        (TurboDense(act, layers[i]) for i in 2:length(layers)-1)...,
        TurboDense(lastact, layers[end]))
    ps = init_params(schain)
    return (schain, ps)
end

function ((schain, ps)::SCModel)(x)
    s = size(x)
    if length(s) > 2
        x = reshape(x, s[1], :)
        r = schain(x, ps)
        reshape(r, s...)
    else
        schain(x, ps)
    end
end

## gradient implementations/hotfixes
## not needed since Zygote handles them just fine


#=
# I believe this is the natural way to interpret the gradient of a loss wrt the model
function Zygote.gradient(loss, (model,ps,st)::StatefulModel, x)
    Lux.gradient(ps |> Lux.ComponentArray) do ps
        loss(model(x,ps,st)[1])
    end[1]
end

# TODO: this needs testing before we open a PR to Lux
function Zygote.pullback(dy, (model,ps,st)::StatefulModel, x)
    u(p) = Lux.apply(model, x, p, st)[1]
    y, back = Lux.pullback(u, Lux.ComponentArray(ps))
    back(dy)[1]
end


# This is conforming to the default Zygote.gradient/withgradient(m->loss(m(x)), model)
function Zygote.withgradient(f, (model,ps,st)::StatefulModel)
    Zygote.withgradient(ps) do
        mod(x) = model(x, ps, st)[1]
        @show mod([1.])
        f(mod)
    end
end

=#
#=
function gradient(f, (model,ps)::SCModel, x)
    Zygote.gradient(ps) do ps
        f(model(x, ps))
    end[1]
end
=#

#= still invoked runtime dispatch, disabling this
# this worked out of the box, but only with runtime inference, try this now
function Zygote.withgradient(f, (schain,ps)::SCModel)
    Zygote.withgradient(ps) do ps
        f(x->(schain,ps)(x))
    end
end

function Optimisers.update(opt, (model,ps)::SCModel, grad)
    opt, psn = Optimisers.update(opt, ps, grad)
    return opt, (model, psn)
end

function Optimisers.setup(opt, (model,ps)::SCModel)
    Optimisers.setup(opt, ps)
end
=#

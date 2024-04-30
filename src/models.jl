
# Neural Network model for mol

Regularized(opt, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), opt)

""" Adam with L2 regularization. Note that this is different from AdamW (Adam+WeightDecay) (c.f. Decay vs L2 Reg.) """
AdamRegularized(adam=1e-3, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), Optimisers.Adam(adam))

NesterovRegularized(lr=1e-3, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), Optimisers.Nesterov(lr))

optimizerstring(opt) = typeof(opt)
optimizerstring(opt::NamedTuple) = opt.layers[end-1].weight.rule

""" obtain the input dimension of a Flux model """
inputdim(model::Flux.Chain) = inputdim(model.layers[1])
inputdim(model::Flux.Dense) = size(model.weight, 2)

""" Obtain the output dimension of a Flux model """
outputdim(model::Flux.Chain) = outputdim(model.layers[end])
outputdim(model::Flux.Dense) = size(model.weight, 1)



""" convenience wrapper returning the provided model with the default AdamW optimiser """
model_with_opt(model, learnrate=1e-2, decay=1e-5) =
    (; model, opt=Flux.setup(Flux.AdamW(learnrate, (0.9, 0.999), decay), model))

""" given an array of arbitrary shape, select the rows `inds` in the first dimension """
function selectrows(x, inds)
    d, s... = size(x)
    x = reshape(x, d, :)
    x = x[inds, :]
    x = reshape(x, :, s...)
end

""" returns a function `features(x)` which map the system coordinates to its features for the Flux model """
#=
function featurizer(sim)
    dim(sim), identity
end
=#

pairnet(data; kwargs...) = pairnet(featuredim(data); kwargs...)

function pairnet((xs, ys)::Tuple; kwargs...)
    pairnet(size(xs, 1); kwargs...)
end

""" Fully connected neural network with `layers` layers from `n` to `nout` dimensions.
`features` allows to pass a featurizer as preprocessor, 
`activation` determines the activation function for each but the last layer
`lastactivation` can be used to modify the last layers activation function """
function pairnet(n::Int=22; layers=3, features=identity, activation=Flux.sigmoid, lastactivation=identity, nout=1)
    float32(x) = Float32.(x)
    nn = Flux.Chain(
        #float32,
        features,
        [Flux.Dense(
            round(Int, n^(l / layers)),
            round(Int, n^((l - 1) / layers)),
            activation)
         for l in layers:-1:2]...,
        Flux.Dense(round(Int, n^(1 / layers)), nout, lastactivation),
    )
    return nn
end

# TODO: take over previous activation function.
""" Given a model and return a copy with its last layer replaced with given output dimension `n` """
function growmodel(m, n)
    Flux.Chain(m.layers[1:end-1]..., Flux.Dense(ISOKANN.inputdim(m.layers[end]), n))
end

# Used by AbstractLangevin
function smallnet(nin, nout, activation=nl = Flux.sigmoid, lastactivation=identity)
    model = Flux.Chain(
        Flux.Dense(nin, 5, activation),
        Flux.Dense(5, 10, activation),
        Flux.Dense(10, 5, activation),
        Flux.Dense(5, nout, lastactivation))
end
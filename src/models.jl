
# Neural Network model for mol

Regularized(opt, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), opt)

"""
    AdamRegularized(adam=1e-3, reg=1e-4)

Constructs an optimizer that combines weight decay regularization with ADAM.
Uses `reg` for the weight decay parameter and `lr` as the learning rate for ADAM.
Note that this is different from AdamW (Adam+WeightDecay) (c.f. Decay vs L2 Reg.). """
AdamRegularized(adam=1e-3, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), Optimisers.Adam(adam))

"""
    NesterovRegularized(; lr=1e-3, reg=1e-4)

Constructs an optimizer that combines weight decay regularization with Nesterov momentum.
Uses `reg` for the weight decay parameter and `lr` as the learning rate for Nesterov acceleration.
This worked well as alternative where ADAM had problems."""
NesterovRegularized(lr=1e-3, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), Optimisers.Nesterov(lr))

optimizerstring(opt) = typeof(opt)
optimizerstring(opt::NamedTuple) = opt.layers[end-1].weight.rule

""" obtain the input dimension of a Flux model """
inputdim(model::Flux.Chain) = inputdim(model.layers[1])
inputdim(model::Flux.Dense) = size(model.weight, 2)

""" Obtain the output dimension of a Flux model """
outputdim(model::Flux.Chain) = outputdim(model.layers[end])
outputdim(model::Flux.Dense) = size(model.weight, 1)

#iscuda(m::Flux.Chain) = m[2].weight isa CuArray
#iscuda(m::Flux.Chain) = first(Flux.trainables(m)) isa CuArray
iscuda(m::Flux.Chain) = typeof(m).parameters[1].parameters[end].parameters[end] <: CuArray

defaultmodel(x::Tuple) = pairnet(n=size(x[1],1))

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

featuredim((xs, ys)::Tuple) = size(xs, 1)
pairnet(data; kwargs...) = pairnet(n=featuredim(data); kwargs...)


""" Fully connected neural network with `layers` layers from `n` to `nout` dimensions.
`features` allows to pass a featurizer as preprocessor,
`activation` determines the activation function for each but the last layer
`lastactivation` can be used to modify the last layers activation function """
function pairnet(; n::Int, layers=3, features=identity, activation=Flux.sigmoid, lastactivation=identity, nout=1, layernorm=true)
    float32(x) = Float32.(x)
    nn = Flux.Chain(
        #float32,
        features,
        layernorm ? Flux.LayerNorm(n) : identity,
        [Flux.Dense(
            round(Int, n^(l / layers)),
            round(Int, n^((l - 1) / layers)),
            activation)
         for l in layers:-1:2]...,
        Flux.Dense(round(Int, n^(1 / layers)), nout, lastactivation),
    )
    return nn
end

"""
    densenet(; layers::Vector{Int}, activation=Flux.sigmoid, lastactivation=identity, layernorm=true) -> Flux.Chain

Construct a fully connected neural network (`Flux.Chain`) with customizable layer sizes, activations, 
and optional input layer normalization.

# Arguments
- `layers::Vector{Int}`: List of layer dimensions. For example, `[10, 32, 16, 1]` creates
  a network with input size 10, two hidden layers of size 32 and 16, and an output layer of size 1.
- `activation`: Activation function applied to all layers except the last. Defaults to `Flux.sigmoid`.
- `lastactivation`: Activation function for the final layer. Defaults to `identity`.
- `layernorm::Bool`: Whether to prepend a `Flux.LayerNorm` layer to normalize the input. Defaults to `true`.

# Returns
A `Flux.Chain` composed of dense layers (and optionally a leading `LayerNorm`).
"""
function densenet(; layers::Vector{Int}, activation=Flux.sigmoid, lastactivation=identity, layernorm=true)
    L = [Flux.Dense(layers[i], layers[i+1], activation) for i in 1:length(layers)-2]
    L = [L; Flux.Dense(layers[end-1], layers[end], lastactivation)]
    layernorm && (L = [Flux.LayerNorm(layers[1]); L])
    Flux.Chain(L...)
end


# TODO: take over previous activation function.
""" Given a model and return a copy with its last layer replaced with given output dimension `n` """
function growmodel(m, n)
    Flux.Chain(m.layers[1:end-1]..., Flux.Dense(ISOKANN.inputdim(m.layers[end]), n))
end

# Used by AbstractLangevin
function smallnet(nin, nout=1, activation=nl = Flux.sigmoid, lastactivation=identity)
    model = Flux.Chain(
        Flux.Dense(nin, 8, activation),
        Flux.Dense(8, 8, activation),
        Flux.Dense(8, 8, activation),
        Flux.Dense(8, nout, lastactivation))
end
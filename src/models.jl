
# Neural Network model for mol

Regularized(opt, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), opt)
AdamRegularized(adam=1e-3, reg=1e-4) = Optimisers.OptimiserChain(Optimisers.WeightDecay(reg), Optimisers.Adam(adam))

""" obtain the input dimension of a Flux model """
inputdim(model::Flux.Chain) = inputdim(model.layers[1])
inputdim(model::Flux.Dense) = size(model.weight, 2)

""" Obtain the output dimension of a Flux model """
outputdim(model::Flux.Chain) = outputdim(model.layers[end])
outputdim(model::Flux.Dense) = size(model.weight, 1)

# TODO: make this dispatch on the simulation or system
function featureinds(sim::IsoSimulation)
    if dim(sim) == 8751
        1:66
    else
        1:dim(sim)
    end
end

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
function featurizer(sim)
    inds = featureinds(sim)
    function features(x)
        x = selectrows(x, inds)
        x = flatpairdists(x)
        return x
    end
end

" Flux neural network model with `layers` fully connected layers using the 
corresponding simulations features as first layers "
function pairnet(sim::IsoSimulation, layers=3)
    n = div(length(featureinds(sim)), 3)^2
    pairnet(n; layers, features=featurizer(sim))
end

""" Fully connected neural network with `layers` layers from `n` to `nout` dimensions.
`features` allows to pass a featurizer as preprocessor, 
`activation` determines the activation function for each but the last layer
`lastactivation` can be used to modify the last layers activation function """
function pairnet(n=22; layers=3, features=identity, activation=Flux.sigmoid, lastactivation=identity, nout=1)
    float32(x) = Float32.(x)
    nn = Flux.Chain(
        float32,
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
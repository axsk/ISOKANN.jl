
# Neural Network model for mol



" Neural Network model for molecules, using pairwise distances as first layer "
function pairnet(sys::MollyLangevin, layers=3)
    pairnetlin(div(dim(sys), 3), layers)
end

function normalnet(sim)
    nn = Flux.Chain(
    )
end

function pairnetn(n=22, layers=3)
    nn = Flux.Chain(
        x -> Float32.(x),
        flatpairdists,
        [Flux.Dense(
            round(Int, n^(2 * l / layers)),
            round(Int, n^(2 * (l - 1) / layers)),
            Flux.sigmoid)
         for l in layers:-1:1]...,

        #x->x .* 2 .- 1
    )
    return nn
end

function pairnetlin(n=22, layers=3)
    nn = Flux.Chain(
        flatpairdists,
        [Flux.Dense(
            round(Int, n^(2 * l / layers)),
            round(Int, n^(2 * (l - 1) / layers)),
            Flux.sigmoid)
         for l in layers:-1:2]...,
        Flux.Dense(round(Int, n^(2 / layers)), 1),
    )
    return nn
end

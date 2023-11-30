# a simple rewrite of the the top layer api
# functionally equivalent (up to the plots, loggers, ...) to isomolly.jl

using ProgressMeter
using ISOKANN: koopman, shiftscale, learnstep!, ISOKANN
Tensor = Array{<:Any, 3}

"""
bare bones ISOKANN, for educational purposes
model is a Flux neural network, opt an Optimiser,
data a tuple of (x,y) coordinates, n the number of iterations
"""
function isokann_simple(model, opt, data, n)
    xs::Matrix, ys::Tensor = data                # decompose data in start and endpoints
    for _ in 1:n
        ks       = mean(model(ys), dims=3)[1,:]  # Monte-Carlo estimate of Kx
        target   = shiftscale(ks)                # SKx
        learnstep!(model, xs, target, opt)       # Neural Network update
    end
    return model                                 # the learned χ-function
end

"""
ISOKANN with advanced data handling,
dataset handles possible resampling of data
"""
function isokann(model, opt, dataset, n)
    losses = Float64[]
    @showprogress for i in 1:n
        data = getdata!(dataset, model)
        loss = isostep!(model, opt, data)
        push!(losses, loss)
    end
    losses
end

function isostep!(model, opt, data::Tuple{<:Matrix, <:Tensor})
    xs, ys = data
    ks     = koopman(model, ys)  # Monte-Carlo estimate of Kx
    target = shiftscale(ks)      # SKx
    loss   = learnstep!(model, xs, target, opt)  # Neural Network update
    return loss
end

getdata!(data::Tuple{<:Matrix, <:Tensor}, model) = data

"dataset type handling chi-stratified subsampling and traindata-subsampling"
mutable struct ChiStratData
    data
    subdata
    iter
    i_resample
    n_resample
    i_generate
    n_generate
    sim
end

function getdata!(d::ChiStratData, model)
    i = d.iter += 1
    if (i%d.i_generate == 0)  # sample new data
        d.data = ISOKANN.adddata(d.data, model, d.sim, d.n_generate)
    end
    if (d.i_resample) > 0 && (i%d.i_resample == 0) # subsample data
        d.subdata = ISOKANN.datasubsample(model, d.data, n_resample)
    end
    return d.subdata
end

"wrap the old ISORun into the simplified isokann() call"
function isokann(iso::ISOKANN.IsoRun)
    i=iso
    data = ChiStratData(
        iso.data,
        ISOKANN.datasubsample(iso.model, iso.data, iso.nx),
        0,
        i.np * i.nl,
        i.nx,
        i.nl*i.np*i.nres,
        i.ny,
        i.sim)

    isokann(iso.model, iso.opt, data, i.nd*i.np*i.nl)
end











"a tuple which allows mutation and creation of new fields"
mutable struct MutTuple
    tuple::NamedTuple
end
MutTuple(;kwargs...) = MutTuple((;kwargs...))
tuple(m::MutTuple) = getfield(m, :tuple)
Base.getproperty(m::MutTuple, x::Symbol) = getfield(tuple(m), x)
function Base.setproperty!(m::MutTuple, x::Symbol, v)
    t = merge(tuple(m), NamedTuple{(x,)}(v))
    setfield!(m, :tuple, t)
end

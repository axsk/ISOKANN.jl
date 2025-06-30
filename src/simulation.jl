## Interface for simulations

## This is supposed to contain the (Molecular) system + integrator

"""
    abstract type IsoSimulation

Abstract type representing an IsoSimulation.
Should implement the methods `coords`, `propagate`, `dim`

"""
abstract type IsoSimulation end

featurizer(::IsoSimulation) = identity

@deprecate isodata SimulationData

function Base.show(io::IO, mime::MIME"text/plain", sim::IsoSimulation)#
    println(io, "$(typeof(sim)) with $(dim(sim)) dimensions")
end

function randx0(sim::IsoSimulation, nx)
    x0 = reshape(coords(sim), :, 1)
    xs = reshape(propagate(sim, x0, nx), :, nx)
    return xs
end

trajectory(sim::IsoSimulation, steps) = error("not implemented")
laggedtrajectory(sim::IsoSimulation, nx) = error("not implemented")

mutable struct ExternalSimulation <: IsoSimulation
    dict::Dict{Symbol, Any}
end

ExternalSimulation(;kwargs...) = ExternalSimulation(Dict(kwargs))

Base.show(io::IO, mime::MIME"text/plain", sim::ExternalSimulation) = print(io, "ExternalSimulation with parameters $(sim.dict)")
masses(sim::ExternalSimulation) = get(sim.dict, :masses, nothing)
pdbfile(sim::ExternalSimulation) = get(sim.dict, :pdbfile, nothing)


#TODO:



###

"""
    struct SimulationData{S,D,C,F}

A struct combining a simulation with the simulated coordinates and corresponding ISOKANN trainingsdata

# Fields
- `sim::S`: The simulation object.
- `data::D`: The ISOKANN trainings data.
- `coords::C`: The orginal coordinates of the simulations.
- `featurizer::F`: A function mapping coordinates to ISOKANN features.

"""
mutable struct SimulationData{S,D,C,F}
    sim::S
    features::D
    coords::C
    featurizer::F
end


"""
    SimulationData(sim::IsoSimulation, nx::Int, nk::Int; ...)
    SimulationData(sim::IsoSimulation, xs::AbstractMatrix, nk::Int; ...)
    SimulationData(sim::IsoSimulation, (xs,ys); ...)
    SimulationData(xs, ys; pdb="", ...)  # for external simulation data

Generates SimulationData from a simulation with either
- `nx` initial points and `nk` Koopman samples
- `xs` as initial points and `nk` Koopman sample
- `xs` as inintial points and `ys` as Koopman samples
- `xs` and `ys` from external simulations of the given `pdb`
- `xs` a trajectory of an external simulation with given `pdb`, implicitly computing the `data_from_trajectory` of succesive samples
"""
SimulationData(sim::IsoSimulation, nx::Int, nk::Int; kwargs...) =
    SimulationData(sim, values(randx0(sim, nx)), nk; kwargs...)

function SimulationData(sim::IsoSimulation, xs::AbstractMatrix, nk::Int; kwargs...)
    ys = propagate(sim, xs, nk)
    SimulationData(sim, (xs, ys); kwargs...)
end

function SimulationData(xs::AbstractMatrix, ys::AbstractArray; pdb=nothing, kwargs...)
    SimulationData(ExternalSimulation(pdb), (xs, ys); kwargs...)
end

function SimulationData(xs::AbstractMatrix; pdb=nothing, kwargs...)
    SimulationData(ExternalSimulation(pdb), data_from_trajectory(xs); kwargs...)
end



function SimulationData(sim::IsoSimulation, (xs, ys)::Tuple; featurizer=featurizer(sim))
    coords = (xs, ys)
    features = (featurizer(xs), fmap(featurizer, ys)) # fmap preserves girsanov weights
    return SimulationData(sim, features, coords, featurizer)
end

#features(sim::SimulationData, x) = sim.featurizer(x)

gpu(d::SimulationData) = SimulationData(d.sim, gpu(d.features), d.coords, d.featurizer)
cpu(d::SimulationData) = SimulationData(d.sim, cpu(d.features), d.coords, d.featurizer)

function features(d::SimulationData, x)
    d.features[1] isa CuArray && (x = cu(x))
    return d.featurizer(x)
end

defaultmodel(d::SimulationData; kwargs...) = pairnet(n=featuredim(d), kwargs...)
featuredim(d::SimulationData) = size(d.features[1], 1)
nk(d::SimulationData) = size(d.features[2], 2)

Base.length(d::SimulationData) = size(d.features[1], 2)
Base.lastindex(d::SimulationData) = length(d)

# facilitates easy indexing into the data, returning a new data object
Base.getindex(d::SimulationData, i) = SimulationData(d.sim, getobs(d.features, i), getobs(d.coords, i), d.featurizer)

@deprecate getcoords coords
@deprecate getxs features
@deprecate getys propfeatures
coords(d::SimulationData) = d.coords[1]
features(d::SimulationData) = d.features[1]
propcoords(d::SimulationData) = d.coords[2]
propfeatures(d::SimulationData) = d.features[2]

"""
    flattenlast(x)

Concatenate all but the first dimension of `x`. Usefull to convert a tensor of samples into a matrix """
flattenlast(x) = reshape(x, size(x, 1), :)

MLUtils.getobs(d::SimulationData) = d.features

pdbfile(s::SimulationData) = pdbfile(s.sim)


"""
    mergedata(d1::SimulationData, d2::SimulationData)

Merge the data and features of `d1` and `d2`, keeping the simulation and features of `d1`.
Note that there is no check if simulation features agree.
"""
function mergedata(d1::SimulationData, d2::SimulationData)
    coords = lastcat.(d1.coords, d2.coords)
    d2f = if d1.featurizer == d2.featurizer
        d2.features
    else
        d1.featurizer.(d2.coords)
    end
    features = lastcat.(d1.features, d2f)
    return SimulationData(d1.sim, features, coords, d1.featurizer)
end

@deprecate Base.merge(d1::SimulationData, d2::SimulationData) mergedata(d1, d2) false


function addcoords(d::SimulationData, coords::AbstractMatrix)
    mergedata(d, SimulationData(d.sim, coords, nk(d), featurizer=d.featurizer))
end


"""
    resample_strat(d::SimulationData, model, n)

χ-stratified subsampling. Select n samples amongst the provided ys/koopman points of `d` such that their χ-value according to `model` is approximately uniformly distributed and propagate them.
Returns a new `SimulationData` which has the new data appended."""
function resample_strat(d::SimulationData, model, n; keepedges=false)
    n == 0 && return d
    xs = chistratcoords(d, model, n; keepedges)
    addcoords(d, xs)
end

function chistratcoords(d::SimulationData, model, n; keepedges=false)
    fs = d.features[2]
    cs = d.coords[2]

    dim, nk, _ = size(fs)
    fs, cs = flattenlast.((fs, cs))
    idxs = subsample_inds(model, fs, n; keepedges)
    return cs[:, idxs]
end


function resample_kde(data::SimulationData, model, n; bandwidth=0.02, unique=false)
    n == 0 && return data

    selinds = if unique
        sampled = Set(eachcol(data.coords[1]))
        [i for (i, c) in enumerate(eachcol(values(data.coords[2]) |> flattenlast)) if !(c in sampled)]
    else
        (:)
    end





    chix = data.features[1] |> model |> vec |> cpu
    chiy = data.features[2] |> flattenlast |> x -> getindex(x, :, selinds) |> model |> vec |> cpu


    m1 = min(minimum(chix), minimum(chiy))
    m2 = max(maximum(chix), maximum(chiy))

    chix = (chix .- m1) ./ (m2 - m1)
    chiy = (chiy .- m1) ./ (m2 - m1)


    iy = resample_kde_ash(chix, chiy, n)

    ys = values(data.coords[2]) |> flattenlast |> x -> getindex(x, :, selinds)
    newdata = addcoords(data, ys[:, iy])
    return newdata
end


function Base.show(io::IO, mime::MIME"text/plain", d::SimulationData)#
    simstr = sprint() do io
        show(io, mime, d.sim)
    end
    println(
        io, """
        SimulationData(;
            sim=$(simstr),
            features=$(size.(d.features)), $(typeof(d.features[2]).name.name),
            coords=$(size.(d.coords)), $(typeof(d.coords[2]).name.name),
            featurizer=$(d.featurizer))"""
    )
end

datasize((xs, ys)::Tuple) = size(xs), size(ys)
features((xs, ys)::Tuple) = xs

"""
    laggedtrajectory(data::SimulationData, n) = laggedtrajectory(data.sim, n, x0=data.coords[1][:, end])

Simulate a trajectory comprising of `n` simulations from the last point in `data`
"""
laggedtrajectory(data::SimulationData, n) = laggedtrajectory(data.sim, n, x0=data.coords[1][:, end])


"""
    trajectorydata_linear(sim::IsoSimulation, steps; reverse=false, kwargs...)

Simulate a single long trajectory of `steps` times the lagtime and use this "chain" to generate the corresponding ISOKANN data.
If `reverse` is true, also add the time-reversed transitions

x (<)--> x (<)--> x
"""
function trajectorydata_linear(sim::IsoSimulation, steps; reverse=false, kwargs...)
    xs = laggedtrajectory(sim, steps)
    SimulationData(sim, data_from_trajectory(xs; reverse), kwargs...)
end


"""
    trajectorydata_bursts(sim::IsoSimulation, steps, nk; kwargs...)

Simulate a single long trajectory of `steps` times the lagtime and start `nk` burst trajectories at each step for the Koopman samples.


x0---x----x---
    / |  / |
    y y  y y
"""
function trajectorydata_bursts(sim::IsoSimulation, steps, nk; x0=coords(sim), kwargs...)
    xs = laggedtrajectory(sim, steps; x0)
    ys = propagate(sim, xs, nk)
    SimulationData(sim, (xs, ys); kwargs...)
end

function lucadata(; path="/scratch/htc/ldonati/vilin/final/output/", nk=10, frame=1)
    xs = readchemfile(path * "trajectory_solute.dcd")
    dim, nx = size(xs)
    ys = similar(xs, dim, nk, nx)
    @showprogress for i in 1:nx, k in 1:nk
        ys[:, k, i] = readchemfile(path * "final_states/xt_$(i-1)_r$(k-1).dcd", frame)
    end
    return xs, ys
end
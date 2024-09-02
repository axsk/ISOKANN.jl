export getcoords
## Interface for simulations

## This is supposed to contain the (Molecular) system + integrator

"""
    abstract type IsoSimulation

Abstract type representing an IsoSimulation.
Should implement the methods `getcoords`, `propagate`, `dim`

"""
abstract type IsoSimulation end

featurizer(::IsoSimulation) = identity

@deprecate isodata SimulationData

function Base.show(io::IO, mime::MIME"text/plain", sim::IsoSimulation)#
    println(io, "$(typeof(sim)) with $(dim(sim)) dimensions")
end

function randx0(sim::IsoSimulation, nx)
    x0 = reshape(getcoords(sim), :, 1)
    xs = reshape(propagate(sim, x0, nx), :, nx)
    return xs
end

trajectory(sim::IsoSimulation, steps) = error("not implemented")
laggedtrajectory(sim::IsoSimulation, nx) = error("not implemented")

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

Generates SimulationData from a simulation with either
- `nx` initial points and `nk` Koopman samples
- `xs` as initial points and `nk` Koopman sample
- `xs` as inintial points and `ys` as Koopman samples
"""
SimulationData(sim::IsoSimulation, nx::Int, nk::Int; kwargs...) =
    SimulationData(sim, randx0(sim, nx), nk; kwargs...)

function SimulationData(sim::IsoSimulation, xs::AbstractMatrix, nk::Int; kwargs...)
    local ys
    try
        ys = propagate(sim, xs, nk)
    catch e
        if e isa OpenMM.OpenMMOverflow
            nx = size(xs, 2)
            xs = xs[:, e.select]
            ys = e.result[:, :, e.select]
            @warn "SimulationData: discarded $(nx-sum(e.select))/$nx starting points due to simulation errors."
            if sum(e.select) == 0
                return sim
            end
        else
            rethrow(e)
        end
    end
    SimulationData(sim, (xs, ys); kwargs...)
end



function SimulationData(sim::IsoSimulation, (xs, ys)::Tuple; featurizer=featurizer(sim))
    coords = (xs, ys)
    features = featurizer.(coords)
    return SimulationData(sim, features, coords, featurizer)
end

#features(sim::SimulationData, x) = sim.featurizer(x)

gpu(d::SimulationData) = SimulationData(d.sim, gpu(d.features), d.coords, d.featurizer)
cpu(d::SimulationData) = SimulationData(d.sim, cpu(d.features), d.coords, d.featurizer)

function features(d::SimulationData, x)
    d.features[1] isa CuArray && (x = cu(x))
    return d.featurizer(x)
end

defaultmodel(d::SimulationData; kwargs...) = defaultmodel(d.sim; n=featuredim(d), kwargs...)
featuredim(d::SimulationData) = size(d.features[1], 1)
nk(d::SimulationData) = size(d.features[2], 2)

Base.length(d::SimulationData) = size(d.features[1], 2)
Base.lastindex(d::SimulationData) = length(d)

# facilitates easy indexing into the data, returning a new data object
Base.getindex(d::SimulationData, i) = SimulationData(d.sim, getobs(d.features, i), getobs(d.coords, i), d.featurizer)

MLUtils.getobs(d::SimulationData) = d.features

getcoords(d::SimulationData) = d.coords[1]

#getcoords(d::SimulationData) = d.coords[1]
#getkoopcoords(d::SimulationData) = d.coords[2]
#getfeatures(d::SimulationData) = d.features[1]
#getkoopfeatures(d::SimulationData) = d.features[2]


flattenlast(x) = reshape(x, size(x, 1), :)

getxs(d::SimulationData) = getxs(d.features)
getys(d::SimulationData) = getys(d.features)

pdb(s::SimulationData) = pdb(s.sim)


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
    adddata(d::SimulationData, model, n)

χ-stratified subsampling. Select n samples amongst the provided ys/koopman points of `d` such that their χ-value according to `model` is approximately uniformly distributed and propagate them.
Returns a new `SimulationData` which has the new data appended."""
function adddata(d::SimulationData, model, n; keepedges=false)
    n == 0 && return d
    xs = chistratcoords(d, model, n; keepedges)
    addcoords(d, xs)
end

function chistratcoords(d::SimulationData, model, n; keepedges=false)
    fs = d.features[2]
    cs = d.coords[2]

    dim, nk, _ = size(fs)
    fs, cs = flattenlast.((fs, cs))

    xs = cs[:, subsample_inds(model, fs, n; keepedges)]
end


function resample_kde(data, model, n; padding=0.0, bandwidth=0.02)
    n == 0 && return data
    chix = data.features[1] |> model |> vec |> cpu
    chiy = data.features[2] |> model |> vec |> cpu
    needles = kde_needles(chix, n; padding, bandwidth)
    inds = pickclosest(chiy, needles)
    ys = data.coords[2] |> flattenlast
    newdata = addcoords(data, ys[:, inds])
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
            features=$(size.(d.features)), $(split(string(typeof(d.features[1])),",")[1]),
            coords=$(size.(d.coords)), $(split(string(typeof(d.coords[1])),",")[1]),
            featurizer=$(d.featurizer))"""
    )
end

function datasize((xs, ys)::Tuple)
    return size(xs), size(ys)
end

laggedtrajectory(data::SimulationData, n) = laggedtrajectory(data.sim, n, x0=data.coords[1][:, end])

"""
    trajectorydata_linear(sim::IsoSimulation, steps; reverse=false, kwargs...)

Simulate a single long trajectory of `steps` times the lagtime and generate the corresponding ISOKANN data.
If `reverse` is true, also add the time-reversed transitions
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
function trajectorydata_bursts(sim::IsoSimulation, steps, nk; x0=getcoords(sim), kwargs...)
    xs = laggedtrajectory(sim, steps; x0)
    ys = propagate(sim, xs, nk)
    SimulationData(sim, (xs, ys), kwargs...)
end
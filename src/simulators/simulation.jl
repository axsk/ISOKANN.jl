## Interface for simulations

## This is supposed to contain the (Molecular) system + integrator
abstract type IsoSimulation end

dim(sim::IsoSimulation) = dim(sim.sys)
getcoords(sim::IsoSimulation) = getcoords(sim.sys)
rotationhandles(sim::IsoSimulation) = rotationhandles(sim.sys)
defaultmodel(sim::IsoSimulation; kwargs...) = pairnet(sim; kwargs...)

# TODO: this should return a SimulationData
function isodata(sim::IsoSimulation, nx, nk)
    xs = randx0(sim, nx)
    ys = propagate(sim, xs, nk)
    return xs, ys
end

function featurizer(sim::IsoSimulation)
    if dim(sim) == 8751 # diala with water?
        return pairdistfeatures(1:66)
    else
        n = div(dim(sim), 3)^2
        return n, flatpairdists
    end
end

function Base.show(io::IO, mime::MIME"text/plain", sim::IsoSimulation)#
    println(io, "$(typeof(sim)) with $(dim(sim)) dimensions")
end

function savecoords(sim::IsoSimulation, data::AbstractArray, path; kwargs...)
    savecoords(sim.sys, data, path; kwargs...)
end

function randx0(sim::IsoSimulation, nx)
    x0 = reshape(getcoords(sim), :, 1)
    xs = reshape(propagate(sim, x0, nx), :, nx)
    return xs
end

struct SimulationData{S,D,C,F}
    sim::S
    data::D
    coords::C
    featurizer::F
end

function SimulationData(sim::IsoSimulation, nx::Int, nk::Int, featurizer=featurizer(sim))
    xs = randx0(sim, nx)
    ys = propagate(sim, xs, nk)
    coords = (xs, ys)
    data = featurizer.(coords)
    return SimulationData(sim, data, coords, featurizer)
end

gpu(d::SimulationData) = SimulationData(d.sim, gpu(d.data), gpu(d.coords), d.featurizer)

dim(d::SimulationData) = size(d.data[1], 1)
Base.length(d::SimulationData) = size(d.data[1], 2)
Base.lastindex(d::SimulationData) = length(d)

function Base.getindex(d::SimulationData, i)
    SimulationData(d.sim, getobs(d.data, i), getobs(d.coords, i), d.featurizer)
end
#Base.iterate(d::SimulationData) = iterate(d.data)
#Base.iterate(d::SimulationData, i) = iterate(d.data, i)

function MLUtils.getobs(d::SimulationData)
    return d.data
end

getcoords(d::SimulationData) = d.coords
flatend(x) = reshape(x, size(x, 1), :)

getxs(d::SimulationData) = getxs(d.data)
getys(d::SimulationData) = getys(d.data)

function adddata(d::SimulationData, model, n)
    y1 = d.data[2]
    c1 = d.coords[2]

    dim, nk, _ = size(y1)
    y1, c1 = flatend.((y1, c1))

    c1 = c1[:, subsample_inds(model, y1, n, keepedges=false)]
    c2 = propagate(d.sim, c1, nk)

    coords = (c1, c2)
    data = d.featurizer.(coords)

    data = lastcat.(d.data, data)
    coords = lastcat.(d.coords, coords)
    return SimulationData(d.sim, data, coords, d.featurizer)
end
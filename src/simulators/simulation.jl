## Implementation of the Langevin dynamics using Molly as a backend


## This is supposed to contain the (Molecular) system + integrator
abstract type IsoSimulation end

dim(sim::IsoSimulation) = dim(sim.sys)
getcoords(sim::IsoSimulation) = getcoords(sim.sys)
rotationhandles(sim::IsoSimulation) = rotationhandles(sim.sys)
defaultmodel(sim::IsoSimulation; kwargs...) = pairnet(sim; kwargs...)

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

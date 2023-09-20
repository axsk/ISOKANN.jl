## Implementation of the Langevin dynamics using Molly as a backend



## This is supposed to contain the (Molecular) system + integrator
abstract type IsoSimulation end

dim(sim::IsoSimulation) = dim(sim.sys)
getcoords(sim::IsoSimulation) = getcoords(sim.sys)
defaultmodel(sim::IsoSimulation) = defaultmodel(sim.sys)
rotationhandles(sim::IsoSimulation) = rotationhandles(sim.sys)

function Base.show(io::IO, mime::MIME"text/plain", sim::IsoSimulation)
    #print(io, isa(sim, MollySDE) ? "Overdamped Langevin" : "Langevin")
    println(io, " system with $(div(dim(sim.sys),3)) atoms")
    println(io, " dt=$(sim.dt), T=$(sim.T), temp=$(sim.temp), gamma=$(sim.gamma)")
end

function savecoords(sim::IsoSimulation, data::AbstractArray, path; kwargs...)
    savecoords(sim.sys, data, path; kwargs...)
end


## Langevin simulation, containing the system and integration parameters

Base.@kwdef mutable struct MollyLangevin{S} <: IsoSimulation
    sys::S
    temp::Float64 = 298.0 # 298 K = 25 °C
    gamma::Float64 = 1.0
    dt::Float64 = 2e-3  # in ps
    T::Float64 = 2e-1 # in ps   # tuned as to take ~.1 sec computation time
    n_threads::Int = 1  # number of threads for the force computations
end



solve(ml; kwargs...) = reduce(hcat, getcoords.(_solve(ml; kwargs...)))

function _solve(ml::MollyLangevin;
    u0=ml.sys.coords,
    logevery=1)

    sys = setcoords(ml.sys, u0)::Molly.System

    # this seems to be necessary for multithreading, but expensive
    #sys.neighbor_finder = deepcopy(sys.neighbor_finder)
    #sys.loggers = loggers=(coords=CoordinateLogger(logevery))
    sys = Molly.System(sys;
        neighbor_finder=deepcopy(sys.neighbor_finder),
        loggers=(coords=Molly.CoordinateLogger(logevery),)
    )

    Molly.random_velocities!(sys, ml.temp * u"K")
    simulator = Molly.Langevin(
        dt=ml.dt * u"ps",
        temperature=ml.temp * u"K",
        friction=ml.gamma * u"ps^-1",
    )
    n_steps = round(Int, ml.T / ml.dt)
    Molly.simulate!(sys, simulator, n_steps; n_threads=ml.n_threads)
    return sys.loggers.coords.history
end

function solve_end(ml::MollyLangevin; u0)
    n_steps = round(Int, ml.T / ml.dt)
    getcoords(_solve(ml; u0, logevery=n_steps)[end])
end

function propagate(ms::MollyLangevin, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, nx, ny)
    inds = [(i, j) for i in 1:nx, j in 1:ny]
    Threads.@threads for (i, j) in inds
        ys[:, i, j] = solve_end(ms; u0=x0[:, i])
    end
    return ys
end



## DEPRECATED
# commented out since importing solve seems to ruin precompilation
#=

using StochasticDiffEq # for OverdampedLangevin
import StochasticDiffEq: SDEProblem, solve

# Overdamped Langevin using StochasticDiffEq

" Type composing the Molly.System with its integration parameters "
Base.@kwdef mutable struct MollySDE{S,A} <: IsoSimulation
    sys::S
    temp::Float64 = 298. # 298 K = 25 °C
    gamma::Float64 = 10.
    dt::Float64 = 2e-5 # in ps
    T::Float64 = 2e-2  # in ps   # tuned as to take ~.1 sec computation time
    alg::A = EM()
    n_threads::Int = 1  # number of threads for the force computations
end


SDEProblem(ms::MollySDE) = SDEProblem(ms.sys, ms.T;
       dt=ms.dt, alg=ms.alg, gamma=ms.gamma, temp=ms.temp, n_threads=ms.n_threads)

function SDEProblem(sys::System, T=1e-3; dt, alg=SROCK2(),
    n_threads=1, gamma=10, temp=298, kwargs...)

    kb = 8.314462618e−3 * u"kJ/mol/K" ./ 1u"Na"

    gamma = gamma / u"ps"
    temp = temp * u"K"
    M = repeat(masses(sys), inner=3)

    sigma = sqrt(2*kb/gamma*temp) .* M.^ (-1/2)
    sigma = ustrip.(u"nm/ps^(1/2)", sigma)

    function driftf(x,p,t)
        (;neighborlist) = p
        sys = setcoords(sys, x)

        f = forces(sys, neighborlist, n_threads=n_threads)

        drift = f ./ masses(sys) ./ gamma
        drift = reduce(vcat, drift) ./ 1u"Na"
        drift = ustrip.(u"nm/ps", drift)
        return drift
    end

    noise(x,p,t) = sigma
    x0 = getcoords(sys)
    neighborlist = find_neighbors(sys)  # TODO: is 1 setting enough?

    SDEProblem(driftf, noise, x0, T, (;neighborlist), alg=alg, dt=dt, kwargs...)
end


"""
propagting the dynamics `ms` starting at the given `x0` each `ny` times.
return the corresponding endpoints in an array of shape [dim, nx, ny]
"""
function propagate(ms::MollySDE, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, nx, ny)
    sde = SDEProblem(ms)
    inds = [(i,j) for i in 1:nx, j in 1:ny]
    Threads.@threads for (i,j) in inds
        #  the tspan fixes u0 assignment (https://github.com/SciML/DiffEqBase.jl/issues/883)
        # TODO: save only at end position
        sol = solve(sde; u0=x0[:,i], tspan=(0,sde.tspan[2]))
        ys[:, i, j] = sol[end]
    end
    # TODO: at which point do we want to center the data?
    return ys
end

solve(ms::MollySDE) = solve(SDEProblem(ms))

=#

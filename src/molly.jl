## Interface to the Molly.System and implementation of the Overdamped Langevin Equation
## in MollySDE

using Molly
using Unitful
using StochasticDiffEq
using CUDA
import StochasticDiffEq: SDEProblem, solve
export PDB_5XER, PDB_6MRR, PDB_ACEMD,
    MollyLangevin, MollySDE,
    solve,
    pairnet, pairnetn,
    propagate

abstract type IsoSimulation end

dim(sim::IsoSimulation) = dim(sim.sys)
dim(sys::System) = length(sys.atoms) * 3

getcoords(sim::IsoSimulation) = getcoords(sim.sys)
defaultmodel(sim::IsoSimulation) = defaultmodel(sim.sys)
defaultmodel(sys::System) = pairnet(sys::System)


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

function Base.show(io::IO, mime::MIME"text/plain", sim::IsoSimulation)
    print(io, isa(sim, MollySDE) ? "Overdamped Langevin" : "Langevin")
    println(io, " system with $(div(dim(sim.sys),3)) atoms")
    println(io, " dt=$(sim.dt), T=$(sim.T), temp=$(sim.temp), gamma=$(sim.gamma)")
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

#Base.show(io::IO, ::MIME"text/plain", x::System) = print(io, "Molly.System ($(dim(x))D)")
#Base.show(io::IO, ::MIME"text/plain", x::Type{<:System}) = print(io, "Molly.System (TYPE)")


# Neural Network model for mol

import Flux

" Neural Network model for molecules, using pairwise distances as first layer "
function pairnet(sys)
    pairnetlin(div(dim(sys),3), 3)
end

function pairnetn(n=22, layers=3)
    nn = Flux.Chain(
            x->Float32.(x),
            flatpairdists,
            [Flux.Dense(
                round(Int, n^(2*l/layers)),
                round(Int, n^(2*(l-1)/layers)),
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
                round(Int, n^(2*l/layers)),
                round(Int, n^(2*(l-1)/layers)),
                Flux.sigmoid)
            for l in layers:-1:2]...,
            Flux.Dense(round(Int, n^(2/layers)), 1),
        )
    return nn
end

## utils for interaction with the ::Molly.System

## not perfect but better then writing it out (but way slower..)
function Molly.System(s::System; kwargs...)
    System(;
        atoms = s.atoms,
        atoms_data = s.atoms_data,
        pairwise_inters = s.pairwise_inters,
        specific_inter_lists = s.specific_inter_lists,
        general_inters = s.general_inters,
        constraints = s.constraints,
        coords = s.coords,
        velocities = s.velocities,
        boundary = s.boundary,
        neighbor_finder = s.neighbor_finder,
        loggers = s.loggers,
        force_units = s.force_units,
        energy_units = s.energy_units,
        k = s.k,
        kwargs...)
end

""" extract the unitful SVector coords from `sys` and return as a normal vector """
getcoords(sys::System) = getcoords(sys.coords)
function getcoords(coords::AbstractArray)
    x0 = ustrip_vec(coords)
    x0 = reduce(vcat, x0)
    return x0 :: AbstractVector
end

""" convert normal vector of coords to SVectors of unitful system coords """
function vec_to_coords(x::AbstractArray, sys::System)
    xx = reshape(x, 3, :)
    coord = sys.coords[1]
    u = unit(coord[1])
    coords = typeof(coord)[c * u for c in eachcol(xx)]
    return coords
end

""" set the system to the given coordinates """
# TODO: the center shift does not belong here, fix constant
# Note: Actually the Langevin integrator removes center of mass motion, so we should be fine
setcoords(sys::System, coords) = setcoords(sys, vec_to_coords(center(coords) .+ 1.36, sys))
setcoords(sys::System, coords::Array{<:SVector{3}}) = System(sys;
    coords=coords,
    velocities = copy(sys.velocities),
    #    neighbor_finder = deepcopy(sys.neighbor_finder),
)




## Save to pdb files

function savecoords(sys::System, coords::AbstractVector, path; append = false)
    append || rm(path, force=true)
    writer = Molly.StructureWriter(0, path)
    sys = setcoords(sys, coords)
    Molly.append_model!(writer, sys)
end

function savecoords(sys::System, data::AbstractMatrix, path; append = false)
    append || rm(path, force=true)
    for x in eachcol(data)
        savecoords(sys, x, path, append=true)
    end
end

function savecoords(sim::IsoSimulation, data::AbstractArray, path; kwargs...)
    savecoords(sim.sys, data, path; kwargs...)
end


## loaders for the molecules in /data

const molly_data_dir = joinpath(dirname(pathof(Molly)), "..", "data")

function PDB_6MRR()
    ff = MolecularForceField(
        joinpath(molly_data_dir, "force_fields", "ff99SBildn.xml"),
        joinpath(molly_data_dir, "force_fields", "tip3p_standard.xml"),
        joinpath(molly_data_dir, "force_fields", "his.xml"),
    )
    sys = System(joinpath(molly_data_dir, "6mrr_nowater.pdb"), ff)
    return sys
end

function PDB_5XER()
    sys = System(
        joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"),
        joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top");
    )
    #temp = 298.0u"K"
    ## attempt at removing water. not working, some internal data is referring to all atoms
    #nosol = map(sys.atoms_data) do a a.res_name != "SOL" end
    #sys.atoms = sys.atoms[nosol]
    #sys.atoms_data = sys.atoms_data[nosol]
    return sys
end

""" Peptide Dialanine """
function PDB_ACEMD(;kwargs...)
    ff = MolecularForceField(joinpath(molly_data_dir, "force_fields", "ff99SBildn.xml"))
    sys = System(joinpath(@__DIR__, "..", "data", "alanine-dipeptide-nowater av.pdb"), ff,
        rename_terminal_res = false, # this is important,
        #boundary = CubicBoundary(Inf*u"nm", Inf*u"nm", Inf*u"nm")  breaking neighbor search
        ; kwargs...
    )
    return sys
end


## Langevin

Base.@kwdef mutable struct MollyLangevin{S} <: IsoSimulation
    sys::S
    temp::Float64 = 298. # 298 K = 25 °C
    gamma::Float64 = 1.
    dt::Float64 = 2e-3  # in ps
    T::Float64 = 2e-1 # in ps   # tuned as to take ~.1 sec computation time
    n_threads::Int = 1  # number of threads for the force computations
end

solve(ml; kwargs...) = reduce(hcat, getcoords.(_solve(ml; kwargs...)))

function _solve(ml::MollyLangevin;
    u0=ml.sys.coords,
    logevery = 1)

    sys = setcoords(ml.sys, u0) :: System

    # this seems to be necessary for multithreading, but expensive
    #sys.neighbor_finder = deepcopy(sys.neighbor_finder)
    #sys.loggers = loggers=(coords=CoordinateLogger(logevery))
    sys = System(sys;
        neighbor_finder = deepcopy(sys.neighbor_finder),
        loggers=(coords=CoordinateLogger(logevery),)
        )

    random_velocities!(sys, ml.temp * u"K")
    simulator = Langevin(
        dt = ml.dt * u"ps",
        temperature = ml.temp * u"K",
        friction = ml.gamma * u"ps^-1",
    )
    n_steps = round(Int, ml.T / ml.dt)
    simulate!(sys, simulator, n_steps; n_threads = ml.n_threads)
    return sys.loggers.coords.history
end

function solve_end(ml::MollyLangevin; u0)
    n_steps = round(Int, ml.T / ml.dt)
    getcoords(_solve(ml; u0, logevery=n_steps)[end])
end

function propagate(ms::MollyLangevin, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, nx, ny)
    inds = [(i,j) for i in 1:nx, j in 1:ny]
    Threads.@threads for (i,j) in inds
        ys[:, i, j] = solve_end(ms; u0=x0[:,i])
    end
    return ys
end

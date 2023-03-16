## Interface to the Molly.System and implementation of the Overdamped Langevin Equation
## in MollySDE

using Molly
using Unitful
using StochasticDiffEq
using CUDA
import StochasticDiffEq: SDEProblem, solve
export PDB_5XER, PDB_6MRR, PDB_ACEMD,
    MollyLangevin, MollySDE,
    solve, exportdata,
    pairnet, pairnetn

abstract type IsoSimulation end

dim(sim::IsoSimulation) = dim(sim.sys)
dim(sys::System) = length(sys.atoms) * 3

getcoords(sim::IsoSimulation) = getcoords(sim.sys)
defaultmodel(sim::IsoSimulation) = defaultmodel(sim.sys)
defaultmodel(sys::System) = pairnet(sys::System)

function savecoords(sim::IsoSimulation, traj::AbstractArray;
    path="out/$(sim.gamma) $(sim.T) $(sim.dt).pdb", kwargs...)
    savecoords(sim.sys, traj, path; kwargs...)
end



" Type composing the Molly.System with its integration parameters "
Base.@kwdef mutable struct MollySDE{S,A} <: IsoSimulation
    sys::S
    temp::Float64 = 298. # 298 K = 25 °C
    gamma::Float64 = 10.
    dt::Float64 = 2e-6 * gamma  # in ps
    T::Float64 = 2e-3 * gamma # in ps   # tuned as to take ~.1 sec computation time
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
    @floop for i in 1:nx, j in 1:ny
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
    pairnetn(div(dim(sys),3), 3)
end

function pairnet2(sys)
    n = div(dim(sys), 3)
    nn = Flux.Chain(
        flatpairdists,
        Flux.Dense(n*n, n, Flux.sigmoid),
        Flux.Dense(n, 1, Flux.sigmoid))
    return nn
end

function pairnetn(n=22, layers=3)
    nn = Flux.Chain(
            flatpairdists,
            [Flux.Dense(
                round(Int, n^(2*l/layers)),
                round(Int, n^(2*(l-1)/layers)),
                Flux.sigmoid)
            for l in layers:-1:1]...
        )
    return nn
end

## utils for interaction with the ::Molly.System

## not perfect but better then writing it out (but way slower..)
type_to_tuple(x::System) = (;(fn=>getfield(x, fn) for fn ∈ fieldnames(typeof(x)))...)
Molly.System(sys::System; kwargs...) = System(;type_to_tuple(sys)..., kwargs...)

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
setcoords(sys::System, coords) = setcoords(sys, vec_to_coords(center(coords) .+ 1.36, sys))
setcoords(sys::System, coords::Array{<:SVector{3}}) = System(sys;
    coords=coords,
    velocities = copy(sys.velocities),
    #    neighbor_finder = deepcopy(sys.neighbor_finder),
)

""" move the ::System to the GPU, mirroring behavior of Flux.gpu """
gpu(sys::System) = System(sys;
    atoms = cu(sys.atoms),
    atoms_data = cu(sys.atoms_data),
    coords = cu(sys.coords),
    velocities = cu(sys.velocities),
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


## loaders for the molecules in /data

const molly_data_dir = joinpath(dirname(pathof(Molly)), "..", "data")

function PDB_6MRR()
    ff = OpenMMForceField(
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
    ff = OpenMMForceField(joinpath(molly_data_dir, "force_fields", "ff99SBildn.xml"))
    sys = System(joinpath("data", "alanine-dipeptide-nowater av.pdb"), ff,
        rename_terminal_res = false, # this is important
        ; kwargs...
    )
    return sys
end


## Langevin

Base.@kwdef mutable struct MollyLangevin{S} <: IsoSimulation
    sys::S
    temp::Float64 = 298. # 298 K = 25 °C
    gamma::Float64 = 1.
    dt::Float64 = 2e-4 * gamma  # in ps
    T::Float64 = 2e-1 * gamma # in ps   # tuned as to take ~.1 sec computation time
    n_threads::Int = 1  # number of threads for the force computations
end

solve(ml; kwargs...) = reduce(hcat, getcoords.(_solve(ml; kwargs...)))

function _solve(ml::MollyLangevin;
    u0=ml.sys.coords,
    logevery = 1)
    sys = setcoords(ml.sys, u0) :: System

    # this seems to be necessary for multithreading, but expensive
    sys = System(sys; neighbor_finder = deepcopy(sys.neighbor_finder))

    sys = System(sys, loggers=(coords=CoordinateLogger(logevery),))

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
    @floop for i in 1:nx, j in 1:ny
        ys[:, i, j] = solve_end(ms; u0=copy(x0[:,i]))
    end
    return ys
end

" bugged as it reinits velocities"
function solvetraj(ml, u0, T)
    oldT = ml.T
    ml.T = ml.dt
    n_steps = round(Int, T / ml.dt)
    x = zeros(dim(ml.sys), n_steps+1)
    x[:, 1] = u0
    for i in 1:n_steps
        # this resamples random velocities on each timestep
        x[:, i+1] = solve(ml; u0=x[:, i])
    end
    ml.T = oldT
    return x
end

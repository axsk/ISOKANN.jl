"""
    struct MollyLangevin{S<:Molly.System} <: IsoSimulation

The `MollyLangevin` struct represents a Langevin dynamics simulation for the Molly package.
    It contains the system as well as the integration parameters.

## Fields
- `sys::S`: The system to be simulated.
- `temp::Float64`: The temperature of the simulation in Kelvin. Default is 298.0 K.
- `gamma::Float64`: The friction coefficient for the Langevin dynamics. Default is 1.0.
- `dt::Float64`: The time step size in picoseconds. Default is 2e-3 ps.
- `T::Float64`: The total simulation time in picoseconds. Default is 2e-1 ps.
- `n_threads::Int`: The number of threads for force computations. Default is 1.

"""
Base.@kwdef mutable struct MollyLangevin{S<:Molly.System} <: IsoSimulation
    sys::S
    temp::Float64 = 298.0 # 298 K = 25 °C
    gamma::Float64 = 1.0
    dt::Float64 = 2e-3  # in ps
    T::Float64 = 2e-1 # in ps   # tuned as to take ~.1 sec computation time
    n_threads::Int = 1  # number of threads for the force computations
end

function Base.show(io::IO, mime::MIME"text/plain", sim::MollyLangevin)
    println(io, " System with $(div(dim(sim.sys),3)) atoms")
    println(io, " dt=$(sim.dt), T=$(sim.T), temp=$(sim.temp), gamma=$(sim.gamma)")
end

""" sample a single trajectory for the given system """
solve(ml::MollyLangevin; kwargs...) = reduce(hcat, getcoords.(_solve(ml; kwargs...)))

function _solve(ml::MollyLangevin;
    u0=ml.sys.coords,
    logevery=1)

    sys = setcoords(ml.sys, u0)::Molly.System

    sys = Molly.System(sys;
        neighbor_finder=deepcopy(sys.neighbor_finder), # this seems to be necessary for multithreading, but expensive
        loggers=(coords=Molly.CoordinateLogger(logevery),)
    )

    Molly.random_velocities!(sys, ml.temp * u"K")
    simulator = Molly.Langevin(
        dt=ml.dt * u"ps",
        temperature=ml.temp * u"K",
        friction=ml.gamma * u"ps^-1",
    )
    n_steps = round(Int, ml.T / ml.dt)
    Molly.simulate!(sys, simulator, n_steps; n_threads=ml.n_threads, run_loggers=:skipzero)
    return sys.loggers.coords.history
end

function solve_end(ml::MollyLangevin; u0)
    n_steps = round(Int, ml.T / ml.dt)
    getcoords(_solve(ml; u0, logevery=n_steps)[end])
end


"""
    propagate(ms::MollyLangevin, x0::AbstractMatrix, ny)

Burst simulation of the MollyLangeving system `ms`.
Propagates `ny` samples for each initial position provided in the columns of `x0`.

`propagate` is the main interface facilitating sampling of a system.
TODO: specify the actual interface required for a simulation to be runnable by ISOKANN.

# Arguments
- `ms::MollyLangevin`: The MollyLangevin solver object.
- `x0::AbstractMatrix`: The initial positions matrix.
- `ny`: The number of trajectories per initial condition.

# Returns
- `ys`: A 3-dimensional array of size `(dim(ms), nx, ny)` containing the propagated solutions.
"""
function propagate(ms::MollyLangevin, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, ny, nx)
    inds = [(i, j) for j in 1:ny, i in 1:nx]
    Threads.@threads for (i, j) in inds
        ys[:, j, i] = solve_end(ms; u0=x0[:, i])
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

## Interface to the Molly.System


dim(sys::System) = length(sys.atoms) * 3
defaultmodel(sys::System) = pairnet(sys::System)

" Return the rotation handle indices for the standardform of a molecule"
function rotationhandles(sys::Molly.System)
    # TODO: fix this ugly if-elseif recognition
    if dim(sys) == 22 # PDB_ACEMD
        return (2, 11, 19)
    elseif dim(sys) == 1234 # some other poweriteration
        return (1, 2, 3)
    else
        error("No rotation handles known for this molecule")
    end
end

# These were supposed to be for convenience but created problems downstream

#Base.show(io::IO, ::MIME"text/plain", x::System) = print(io, "Molly.System ($(dim(x))D)")
#Base.show(io::IO, ::MIME"text/plain", x::Type{<:System}) = print(io, "Molly.System (TYPE)")


# Molly.System is not mutable. We provide a constructor creating a new instance with updated fields.
# (This was not possible programmatically since the System type is not inferrable from the fields alone.)
#=
" Constructor to update some fields of the (immutable) Molly.System "
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
=#

""" extract the unitful SVector coords from `sys` and return as a normal vector """
getcoords(sys::System) = getcoords(sys.coords)
function getcoords(coords::AbstractArray)
    x0 = Molly.ustrip_vec(coords)
    x0 = reduce(vcat, x0)
    return x0::AbstractVector
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
# TODO: the centercoords shift does not belong here, fix constant
# Note: Actually the Langevin integrator removes centercoords of mass motion, so we should be fine
setcoords(sys::System, coords) = setcoords(sys, vec_to_coords(centercoords(coords) .+ 1.36, sys))
setcoords(sys::System, coords::Array{<:SVector{3}}) = System(sys;
    coords=coords,
    velocities=copy(sys.velocities)
    #    neighbor_finder = deepcopy(sys.neighbor_finder),
)


## Save to pdb files

function savecoords(sys::System, coords::AbstractVector, path; append=false)
    append || rm(path, force=true)
    writer = Molly.StructureWriter(0, path)
    sys = setcoords(sys, coords)
    Molly.append_model!(writer, sys)
end

function savecoords(sys::System, data::AbstractMatrix, path; append=false)
    append || rm(path, force=true)
    for x in eachcol(data)
        savecoords(sys, x, path, append=true)
    end
end


## loaders for the molecules in /data

const molly_data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
const isokann_data_dir = joinpath(dirname(pathof(ISOKANN)), "..", "data")

molly_data(path) = joinpath(molly_data_dir, path)
isokann_data(path) = joinpath(isokann_data_dir, path)

molly_forcefields(ffs) = Molly.MolecularForceField(map(ffs) do ff
    molly_data("force_fields/$ff")
end...)

PDB_6MRR() = System(molly_data("6mrr_nowater.pdb"), molly_forcefields(["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"]))

function PDB_5XER()
    sys = System(
        molly_data("5XER/gmx_coords.gro"),
        molly_data("5XER/gmx_top_ff.top")
    )
    #temp = 298.0u"K"
    ## attempt at removing water. not working, some internal data is referring to all atoms
    #nosol = map(sys.atoms_data) do a a.res_name != "SOL" end
    #sys.atoms = sys.atoms[nosol]
    #sys.atoms_data = sys.atoms_data[nosol]
    return sys
end

""" Create a Molly system for the alanine dipeptide without solvent """
function PDB_ACEMD(; kwargs...)
    System(
        joinpath(isokann_data_dir, "alanine-dipeptide-nowater av.pdb"),
        molly_forcefields(["ff99SBildn.xml"]),
        rename_terminal_res=false, # this is important,
        #boundary = CubicBoundary(Inf*u"nm", Inf*u"nm", Inf*u"nm")  breaking neighbor search
        ; kwargs...
    )
end

""" Create a Molly system for the small Chignolin protein """
function PDB_1UAO(; rename_terminal_res=true, kwargs...)
    System(joinpath(isokann_data_dir, "1uao av.pdb"), molly_forcefields(["ff99SBildn.xml"]),
        ; rename_terminal_res, # this is important,
        #boundary = CubicBoundary(Inf*u"nm", Inf*u"nm", Inf*u"nm")  breaking neighbor search
        kwargs...)
end

""" Create a Molly system for the alanine dipeptide with water """
function PDB_diala_water()
    Molly.System(
        isokann_data("dipeptide_equil.pdb"),
        molly_forcefields(["ff99SBildn.xml", "tip3p_standard.xml"]);
        rename_terminal_res=false)
end


"""
    OverdampedLangevinGirsanov(; <keyword arguments>)

Simulates the overdamped Langevin equation using the Euler-Maruyama method with an auxilliary control w/u
with σ = sqrt(2KT/(mγ))
dX = (-∇U(X)/(γm) + σu) dt + σ dW

where u is the control function, such that u(x,t) = σ .* w(x,t).
The accumulated Girsanov reweighting is stored in the field `g`

# Arguments
- `dt::S`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `friction::F`: the friction coefficient of the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
- `w::Function`: the control function, such that u(x,t) = σ .* w(x,t)
"""
struct OverdampedLangevinGirsanov{S, K, F, Fct}
    dt::S
    temperature::K
    friction::F
    remove_CM_motion::Int
    g::Float64  # the Girsanov integral
    w::Fct
end

function OverdampedLangevinGirsanov(; dt, temperature, friction, w, remove_CM_motion=1, G=0.)
    return OverdampedLangevinGirsanov(dt, temperature, friction, w, Int(remove_CM_motion), G)
end

function simulate!(sys,
                    sim::OverdampedLangevinGirsanov,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    run_loggers=true,
                    rng=Random.GLOBAL_RNG)
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)

    for step_n in 1:n_steps
        old_coords = copy(sys.coords)

        dt = sim.dt
        γ = sim.friction
        m = Molly.masses(sys)
        k = sys.k
        T = sim.temperature

        F = forces(sys, neighbors; n_threads=n_threads)
        w = sim.w(sys.coords, sys.time)

        # we reconstruct dB from the Boltzmann velocities
        # this takes care of units and the correct type
        # but maybe sampling ourselves works just as well and is cleaner?
        v = random_velocities(sys, T; rng=rng)
        dB = @. v / sqrt(k * T / m) * sqrt(dt)

        σ = @. sqrt(2 * k * T / (γ * m))

        b = @. (F / (γ * m))
        u = @. σ * w

        sys.coords += @. (b + σ * u) * dt + σ * dB
        sim.g += dot(u, @. u * (dt / 2) + dB)

        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                   n_threads=n_threads)

        run_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads)
    end
    return sys
end
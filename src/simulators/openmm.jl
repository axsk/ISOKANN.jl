module OpenMM

using PyCall, CUDA
using LinearAlgebra: norm, dot

import JLD2
import ..ISOKANN: ISOKANN, IsoSimulation,
    propagate, dim, randx0,
    featurizer, defaultmodel,
    savecoords, getcoords, force, pdbfile,
    force, potential, lagtime, trajectory

export OpenMMSimulation, FORCE_AMBER, FORCE_AMBER_IMPLICIT
export OpenMMScript
export FeaturesAll, FeaturesAll, FeaturesPairs, FeaturesRandomPairs

DEFAULT_PDB = normpath("$(@__DIR__)/../../data/systems/alanine dipeptide.pdb")
FORCE_AMBER = ["amber14-all.xml"]
FORCE_AMBER_IMPLICIT = ["amber14-all.xml", "implicit/obc2.xml"]
FORCE_AMBER_EXPLICIT = ["amber14-all.xml", "amber/tip3p_standard.xml"]

function __init__()
    # install / load OpenMM
    try
        pyimport_conda("openmm", "openmm", "conda-forge")
        pyimport_conda("openmmforcefields", "openmmforcefields", "conda-forge")
        pyimport_conda("joblib", "joblib")

        @pyinclude("$(@__DIR__)/mopenmm.py")
    catch
        @warn "Could not load openmm."
    end
end

###

#abstract type OpenMMSimulation <: IsoSimulation end

"""
    OpenMMSimulation(; pdb, steps, ...)
    OpenMMSimulation(; py, steps)

Constructs an OpenMM simulation object.
Either use  `OpenMMSimulation(;py, steps)` where `py`` is the location of a .py python script creating a OpenMM simulation object
or supply a .pdb file via `pdb` and the following parameters (see also defaultsystem):

## Arguments
- `pdb::String`: Path to the PDB file.
- `ligand::String`: Path to ligand file.
- `forcefields::Vector{String}`: List of force field XML files.
- `temp::Float64`: Temperature in Kelvin.
- `friction::Float64`: Friction coefficient in 1/picosecond.
- `step::Float64`: Integration step size in picoseconds.
- `steps::Int`: Number of simulation steps.
- `features`: Which features to use for learning the chi function.
              -  A vector of `Int` denotes the indices of all atoms to compute the pairwise distances from.
              -  A vector of CartesianIndex{2} computes the specific distances between the atom pairs.
              -  A number denotes the radius below which all pairs of atoms will be used (computed only on the starting configuration)
              -  If `nothing` all pairwise distances are used.
- `minimize::Bool`: Whether to perform energy minimization on first state.
- `nthreads`: The number of threads to use for parallelization of multiple simulations.
- `mmthreads`: The number of threads to use for each OpenMM simulation. Set to "gpu" to use the GPU platform.

## Returns
- `OpenMMSimulation`: An OpenMMSimulation object.

"""
struct OpenMMSimulation <: IsoSimulation
    pysim::PyObject
    steps::Int
    constructor

    function OpenMMSimulation(; steps=100, k...)
        k = NamedTuple(k)
        if haskey(k, :py)
            @pyinclude(k.py)
            pysim = py"simulation"
            return new(pysim, steps, k)
        elseif haskey(k, :pdb)
            pysim = defaultsystem(; k...)
            return new(pysim, steps, k)
        else
            return OpenMMSimulation(; pdb=DEFAULT_PDB, steps, k...)
        end
    end
end

function defaultsystem(;
    pdb=DEFAULT_PDB,
    ligand="",
    forcefields=["amber14-all.xml"],
    temp=310, # kelvin
    friction=1,  # 1/picosecond
    step=0.002, # picoseconds
    minimize=false,
    addwater=false,
    padding=3,
    ionicstrength=0.0,
    mmthreads=CUDA.functional() ? "gpu" : 1,
    forcefield_kwargs=Dict(),
    kwargs...
)
    @pycall py"defaultsystem"(pdb, ligand, forcefields, temp, friction, step, minimize; addwater, padding, ionicstrength, forcefield_kwargs, mmthreads)::PyObject
end


# TODO: this are remnants of the old detailed openmm system, remove them eventually
mmthreads(sim::OpenMMSimulation) = get(sim.constructor, :mmthreads, CUDA.functional() ? "gpu" : 1)
nthreads(sim::OpenMMSimulation) = get(sim.constructor, :nthreads, CUDA.functional() ? 1 : Threads.nthreads())
momenta(sim::OpenMMSimulation) = get(sim.constructor, :momenta, false)
pdbfile(sim::OpenMMSimulation) = get(() -> createpdb(sim), sim.constructor, :pdb)

steps(sim) = sim.steps::Int
friction(sim) = friction(sim.pysim)
temp(sim) = temp(sim.pysim)
stepsize(sim) = stepsize(sim.pysim)


lagtime(sim::OpenMMSimulation) = steps(sim) * stepsize(sim) # in ps
dim(sim::OpenMMSimulation) = length(getcoords(sim))
defaultmodel(sim::OpenMMSimulation; kwargs...) = ISOKANN.pairnet(; kwargs...)

getcoords(sim::OpenMMSimulation) = getcoords(sim.pysim, momenta(sim))::Vector{Float64}
setcoords(sim::OpenMMSimulation, coords) = setcoords(sim.pysim, coords, momenta(sim))
natoms(sim::OpenMMSimulation) = div(dim(sim), 3)

friction(pysim::PyObject) = pysim.integrator.getFriction()._value # 1/ps
temp(pysim::PyObject) = pysim.integrator.getTemperature()._value # kelvin
stepsize(pysim::PyObject) = pysim.integrator.getStepSize()._value # ps
getcoords(pysim::PyObject, momenta) = py"get_numpy_state($pysim.context, $momenta).flatten()"

iscuda(sim::OpenMMSimulation) = iscuda(sim.pysim)
iscuda(pysim::PyObject) = pysim.context.getPlatform().getName() == "CUDA"

function createpdb(sim)
    pysim = sim.pysim
    file = tempname() * ".pdb"
    pdb = py"app.PDBFile"
    pdb.writeFile(pysim.topology, PyReverseDims(reshape(getcoords(sim), 3, :)), file)  # TODO: fix this
    return file
end

featurizer(sim::OpenMMSimulation) = featurizer(sim, get(sim.constructor, :features, nothing))

featurizer(sim, ::Nothing) = natoms(sim) < 100 ? FeaturesAll() : error("No default featurizer specified")
featurizer(sim, atoms::Vector{Int}) = FeaturesAtoms(atoms)
featurizer(sim, pairs::Vector{Tuple{Int,Int}}) = FeaturesPairs(pairs)
featurizer(sim, features::Function) = features
featurizer(sim, radius::Number) = FeaturesPairs([calpha_pairs(sim.pysim); local_atom_pairs(sim.pysim, radius)] |> unique)

struct FeaturesCoords end
(f::FeaturesCoords)(coords) = coords

struct FeaturesAll end
(f::FeaturesAll)(coords) = ISOKANN.flatpairdists(coords)

struct FeaturesAtoms
    atominds::Vector{Int}
end
(f::FeaturesAtoms)(coords) = ISOKANN.flatpairdists(coords, f.atominds)

struct FeaturesPairs
    pairs::Vector{Tuple{Int,Int}}
end
(f::FeaturesPairs)(coords) = ISOKANN.pdists(coords, f.pairs)


import StatsBase
function FeaturesPairs(sim::OpenMMSimulation; maxdist::Number, atomfilter::Function=a -> true, maxfeatures::Int=0)
    pairs = local_atom_pairs(sim.pysim, maxdist; atomfilter=atomfilter)

    if 0 < maxfeatures < length(pairs)
        pairs = StatsBase.sample(pairs, maxfeatures, replace=false)
    end

    return FeaturesPairs(pairs)
end

# TODO: replace this with laggedtrajectory?
""" generate `n` random inintial points for the simulation `mm` """
randx0(sim::OpenMMSimulation, n) = ISOKANN.laggedtrajectory(sim, n)


"""
    propagate(s::OpenMMSimulation, x0::AbstractMatrix{T}, ny; nthreads=Threads.nthreads(), mmthreads=1) where {T}

Propagates `ny` replicas of the OpenMMSimulation `s` from the inintial states `x0`.

# Arguments
- `s`: An instance of the OpenMMSimulation type.
- `x0`: Matrix containing the initial states as columns
- `ny`: The number of replicas to create.

# Optional Arguments
- `nthreads`: The number of threads to use for parallelization of multiple simulations.
- `mmthreads`: The number of threads to use for each OpenMM simulation. Set to "gpu" to use the GPU platform.

Note: For CPU we observed better performance with nthreads = num cpus, mmthreads = 1 then the other way around.
With GPU nthreads > 1 should be supported, but on our machine lead to slower performance then nthreads=1.
"""
function propagate(sim::OpenMMSimulation, x0::AbstractMatrix{T}, ny; stepsize=stepsize(sim), steps=steps(sim), nthreads=nthreads(sim), mmthreads=mmthreads(sim), momenta=momenta(sim)) where {T}
    iscuda(sim) && CUDA.functional() && CUDA.reclaim()
    dim, nx = size(x0)
    xs = repeat(x0, outer=[1, ny])
    xs = permutedims(reinterpret(Tuple{T,T,T}, xs))
    ys = @pycall py"threadedrun"(xs, sim.pysim, stepsize, steps, nthreads, momenta)::Vector{Float32}
    ys = reshape(ys, dim, nx, ny)
    ys = permutedims(ys, (1, 3, 2))
    checkoverflow(ys)  # control the simulated data for NaNs and too large entries and throws an error
    return ys#convert(Array{Float32,3}, ys)
end

struct OpenMMOverflow{T} <: Exception where {T}
    result::T
    select::Vector{Bool}  # flags which results are valid
end

function checkoverflow(ys, overflow=100)
    select = map(eachslice(ys, dims=3)) do y
        !any(@.(abs(y) > overflow || isnan(y)))
    end
    !all(select) && throw(OpenMMOverflow(ys, select))
end

"""
    trajectory(s::OpenMMSimulation, x0, steps=s.steps, saveevery=1; stepsize = s.step, mmthreads = s.mmthreads)

Return the coordinates of a single trajectory started at `x0` for the given number of `steps` where each `saveevery` step is stored.
"""
function trajectory(sim::OpenMMSimulation; x0::AbstractVector{T}=getcoords(sim), steps=steps(sim), saveevery=1, stepsize=stepsize(sim), mmthreads=mmthreads(sim), momenta=momenta(sim)) where {T}
    x0 = reinterpret(Tuple{T,T,T}, x0)
    xs = py"trajectory"(sim.pysim, x0, stepsize, steps, saveevery, mmthreads, momenta)
    xs = permutedims(xs, (3, 2, 1))
    xs = reshape(xs, :, size(xs, 3))
    return xs
end

@deprecate trajectory(sim, x0, steps, saveevery; kwargs...) trajectory(sim; x0, steps, saveevery, kwargs...)

function ISOKANN.laggedtrajectory(sim::OpenMMSimulation, n_lags, steps_per_lag=steps(sim); x0=getcoords(sim), keepstart=false)
    steps = steps_per_lag * n_lags
    saveevery = steps_per_lag
    xs = trajectory(sim; x0, steps, saveevery)
    return keepstart ? xs : xs[:, 2:end]
end

function minimize!(sim::OpenMMSimulation, coords=getcoords(sim); iter=100)
    setcoords(sim, coords)
    return sim.pysim.minimizeEnergy(maxIterations=iter)
    return nothing
end

function setcoords(sim::PyObject, coords::AbstractVector{T}, momenta) where {T}
    if momenta
        n = length(t) ÷ 2
        x, v = t[1:n], t[n+1:end]
        sim.context.setPositions(PyReverseDims(reshape(x, 3, :)))
        sim.context.setVelocities(PyReverseDims(reshape(v, 3, :)))
    else
        sim.context.setPositions(PyReverseDims(reshape(coords, 3, :)))
    end
end

""" mutates the state in sim """
function set_random_velocities!(sim, x)
    v = py"set_random_velocities($(sim.pysim.context))"
    n = length(x) ÷ 2
    x[n+1:end] = v
    return x
end

force(sim::OpenMMSimulation, x::CuArray) = force(sim, Array(x)) |> cu
potential(sim::OpenMMSimulation, x::CuArray) = potential(sim, Array(x))
masses(sim::OpenMMSimulation) = [sim.pysim.system.getParticleMass(i - 1)._value for i in 1:sim.pysim.system.getNumParticles()] # in daltons

function force(sim::OpenMMSimulation, x)
    CUDA.reclaim()
    setcoords(sim, x)
    pysim = sim.pysim
    return PyArray(py"$pysim.context.getState(getForces=True).getForces(asNumpy=True)._value.reshape(-1)"o)
end

function potential(sim::OpenMMSimulation, x)
    CUDA.reclaim()
    setcoords(sim, x)
    v = sim.pysim.context.getState(getEnergy=true).getPotentialEnergy()
    v = v.value_in_unit(v.unit)
end


function savecoords(path, sim::OpenMMSimulation, coords::AbstractArray{T}) where {T}
    coords = ISOKANN.cpu(coords)
    s = sim.pysim
    p = py"pdbfile.PDBFile"
    file = py"open"(path, "w")
    p.writeHeader(s.topology, file)
    for (i, coords) in enumerate(eachcol(coords))
        pos = reinterpret(Tuple{T,T,T}, coords .* 10)
        p.writeModel(s.topology, pos, file, modelIndex=i)
    end
    p.writeFooter(s.topology, file)
    file.close()
end

### FEATURE FILTERS

function remove_H_H2O_NACL(atom)
    !(
        atom.element.symbol == "H" ||
        atom.residue.name in ["HOH", "NA", "CL"]
    )
end

function local_atom_pairs(pysim::PyObject, radius; atomfilter=remove_H_H2O_NACL)
    coords = reshape(getcoords(pysim, false), 3, :)
    atoms = filter(atomfilter, pysim.topology.atoms() |> collect)
    inds = map(atom -> atom.index + 1, atoms)

    pairs = Tuple{Int,Int}[]
    for i in 1:length(inds)
        for j in i+1:length(inds)
            if norm(coords[:, i] - coords[:, j]) <= radius
                push!(pairs, (inds[i], inds[j]))
            end
        end
    end
    return pairs
end

function calpha_pairs(pysim::PyObject)
    local_atom_pairs(pysim, Inf; atomfilter=x -> x.name == "CA")
end

calpha_inds(sim::OpenMMSimulation) = calpha_inds(sim.pysim)
function calpha_inds(pysim::PyObject)
    map(filter(x -> x.name == "CA", pysim.topology.atoms() |> collect)) do atom
        atom.index + 1
    end
end


### SERIALIZATION

struct OpenMMSimulationSerialized
    steps
    constructor
end

JLD2.writeas(::Type{OpenMMSimulation}) = OpenMMSimulationSerialized

Base.convert(::Type{OpenMMSimulationSerialized}, sim::OpenMMSimulation) =
    OpenMMSimulationSerialized(steps(sim), sim.constructor)

Base.convert(::Type{OpenMMSimulation}, s::OpenMMSimulationSerialized) =
    OpenMMSimulation(; steps=s.steps, s.constructor...)

function Base.show(io::IO, mime::MIME"text/plain", sim::OpenMMSimulation)#
    #featstr = if sim.features isa Vector{Tuple{Int,Int}}
    #    "{$(length(sim.features)) pairwise distances}"
    #else
    #    sim.features
    #end
    # features=$featstr)
    #ligand="$(sim.ligand)",
    #        forcefields=$(sim.forcefields),
    println(
        io, """
        OpenMMSimulation(;
            pdb="$(pdbfile(sim))",
            temp=$(temp(sim)),
            friction=$(friction(sim)),
            step=$(stepsize(sim)),
            steps=$(steps(sim))
        with $(natoms(sim)) atoms"""
    )

end


"""
    integrate_langevin(sim::OpenMMSimulation, x0=getcoords(sim); steps=steps(sim), F_ext::Union{Function,Nothing}=nothing, saveevery::Union{Int, nothing}=nothing)

Integrate the Langevin equations with a Euler-Maruyama scheme, allowing for external forces.

- F_ext: An additional force perturbation. It is expected to have the form F_ext(F, x) and mutating the provided force F.
- saveevery: If `nothing`, returns just the last point, otherwise returns an array saving every `saveevery` frame.
"""
function integrate_langevin(sim::OpenMMSimulation, x0=getcoords(sim); steps=steps(sim), F_ext::Union{Function,Nothing}=nothing, saveevery::Union{Int,Nothing}=nothing)
    x = copy(x0)
    v = zero(x) # this should be either provided or drawn from the Maxwell Boltzmann distribution
    kBT = 0.008314463 * temp(sim)
    dt = stepsize(sim) # sim,step is in picosecond, we calculate in nanoseconds
    gamma = friction(sim) # convert 1/ps = 1000/ns
    m = repeat(masses(sim), inner=3)
    out = isnothing(saveevery) ? x : similar(x, length(x), cld(steps, saveevery))

    for i in 1:steps
        F = force(sim, x)
        isnothing(F_ext) || (F_ext(F, x)) # note that F_ext(F, x) is assumed to be mutating F
        langevin_step!(x, v, F, m, gamma, kBT, dt)
        isnothing(saveevery) || i % saveevery == 0 && (out[:, div(i, saveevery)] = x)
    end
    return out
end

function langevin_step!(x, v, F, m, gamma, kBT, dt)
    db = randn(length(x))
    @. v += 1 / m * ((F - gamma * v) * dt + sqrt(2 * gamma * kBT * dt) * db)
    @. x += v * dt
end

function integrate_girsanov(sim::OpenMMSimulation; x0=getcoords(sim), steps=steps(sim), u::Union{Function,Nothing}=nothing)
    # TODO: check units on the following three lines
    kB = 0.008314463
    dt = stepsize(sim)
    γ = friction(sim)

    M = repeat(masses(sim), inner=3)
    T = temp(sim)
    σ = @. sqrt(2 * kB * T / (γ * M))

    x = copy(x0)
    g = 0.

    z = similar(x, length(x), steps)

    for i in 1:steps
        F = force(sim, x)
        ux = u(x)
        g += od_langevin_step_girsanov!(x, F, M, σ, γ, dt, ux)
        z[:, i] = x
    end

    return x, g, z
end

function od_langevin_step_girsanov!(x, F, M, σ, γ, dt, u)
    dB = randn(length(x)) * sqrt(dt)
    @. x += (1 / (γ * M) * F + (σ * u)) * dt + σ * dB
    dg = dot(u, u) / 2 * dt + dot(u, dB) * sqrt(dt)
    return dg
end

using Random: randn!

langevin_girsanov(sim, q::AbstractMatrix, steps, u) =
    mapreduce(hcat, eachcol(q)) do q
        langevin_girsanov(sim, q, steps, u)
    end

function langevin_girsanov(sim, q, steps, u)
    # from https://pubs.acs.org/doi/full/10.1021/acs.jpcb.4c01702

    kB = 0.008314463
    dt = stepsize(sim)
    ξ = friction(sim)
    T = temp(sim)
    M = repeat(masses(sim), inner=3)

    # Maxwell-Boltzmann distribution
    p = randn(length(M)) .* sqrt.(M .* kB .* T)

    t2 = dt / 2
    a = @. t2 / M # eq 18
    d = @. exp(-ξ * dt) # eq 17
    f = @. sqrt(kB * T * M * (1 - exp(-2 * ξ * dt))) # eq 17

    b = similar(p)
    η = similar(p)
    Δη = similar(p)
    F = similar(p)
    g = 0


    for i in 1:steps
        randn!(η)
        @. q += a * p # A

        # girsanov part
        F .= u(q) # perturbation force ∇U_bias = -F
        @. Δη = (d + 1) / f + dt / 2 * F
        g += η' * Δη + Δη' * Δη / 2
        F .+= force(sim, q) # total force: -∇V - ∇U_bias

        @. b = t2 * F
        @. p += b  # B
        @. p = d * p + f * η # O
        @. p += b  # B
        @. q += a * p # A
    end

    return q, exp(-g)
end

function batcheddot(x, y)
    sum(a .* y, dims=1)
end


end #module
module OpenMM

using PyCall, CUDA
using LinearAlgebra: norm

import JLD2
import ..ISOKANN: ISOKANN, IsoSimulation,
    propagate, dim, randx0,
    featurizer, defaultmodel,
    savecoords, getcoords, force, pdb,
    force, potential, lagtime, trajectory

export OpenMMSimulation, FORCE_AMBER, FORCE_AMBER_IMPLICIT
export OpenMMScript
export FeaturesAll, FeaturesAll, FeaturesPairs, FeaturesRandomPairs

DEFAULT_PDB = "$(@__DIR__)/../../data/systems/alanine dipeptide.pdb"
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

""" A Simulation wrapping the Python OpenMM Simulation object """
struct OpenMMSimulation <: IsoSimulation
    pysim::PyObject
    pdb
    ligand
    forcefields
    temp
    friction
    step
    steps
    features
    nthreads::Int
    mmthreads::Union{Int,String}
    momenta::Bool
end

steps(sim) = sim.steps::Int
# TODO: we have redundant values, decide which ones to use
friction(pysim::PyObject) = pysim.integrator.getFriction()._value # 1/ps
temp(pysim::PyObject) = pysim.integrator.getTemperature()._value # kelvin
stepsize(pysim::PyObject) = pysim.integrator.getStepSize()._value # ps

friction(sim) = friction(sim.pysim)
temp(sim) = temp(sim.pysim)
stepsize(sim) = stepsize(sim.pysim)

function OpenMMScript(filename::String;
    steps::Int,
    features=nothing,
)
    @pyinclude(filename)
    pysim = py"simulation"
    pdb = py"pdb"
    mmthreads = CUDA.has_cuda() ? "gpu" : 1

    OpenMMSimulation(pysim, pdb, nothing, nothing, temp(pysim), friction(pysim), stepsize(pysim), steps, features, 1, mmthreads, false)
end


"""
    OpenMMSimulation(; pdb=DEFAULT_PDB, ligand="", forcefields=["amber14-all.xml", "amber14/tip3pfb.xml"], temp=298, friction=1, step=0.002, steps=100, features=nothing, minimize=false)

Constructs an OpenMM simulation object.

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
function OpenMMSimulation(;
    pdb=DEFAULT_PDB,
    ligand="",
    forcefields=["amber14-all.xml"],
    temp=298, # kelvin
    friction=1,  # 1/picosecond
    step=0.002, # picoseconds
    steps=100,
    features=:all,
    minimize=false,
    gpu=false,
    nthreads=gpu ? 1 : Threads.nthreads(),
    mmthreads=gpu ? "gpu" : 1,
    addwater=false,
    padding=3,
    ionicstrength=0.0,
    forcefield_kwargs=Dict(),
    momenta=false)

    platform, properties = if mmthreads == "gpu"
        "CUDA", Dict()
    else
        "CPU", Dict("Threads" => "$mmthreads")
    end

    pysim = @pycall py"defaultsystem"(pdb, ligand, forcefields, temp, friction, step, minimize; addwater, padding, ionicstrength, forcefield_kwargs, platform, properties)::PyObject

    if features isa Number
        radius = features
        features = [calpha_pairs(pysim); local_atom_pairs(pysim, radius)] |> unique
    end

    return OpenMMSimulation(pysim::PyObject, pdb, ligand, forcefields, temp, friction, step, steps, features, nthreads, mmthreads, momenta)
end

#=
function featurizer(sim::OpenMMSimulation)
    if sim.features isa (Vector{Int})
        ix = vec([1, 2, 3] .+ ((sim.features .- 1) * 3)')
        return ISOKANN.pairdistfeatures(ix)
    elseif sim.features isa (Vector{Tuple{Int,Int}}) # local pairwise distances
        inds = sim.features
        return coords -> ISOKANN.pdists(coords, inds)
    elseif sim.features == :all
        if length(getcoords(sim)) > 100
            @warn "Computing _all_ pairwise distances for a bigger (>100 atoms) molecule. Try using a cutoff by setting features::Number in OpenMMSimulatioon"
        end
        return ISOKANN.flatpairdists
    else
        error("unknown featurizer")
    end
end
=#

featurizer(sim::OpenMMSimulation) = featurizer(sim, sim.features)

featurizer(sim, ::Nothing) = error("No default featurizer specified")
featurizer(sim, atoms::Vector{Int}) = FeaturesAtoms(atoms)
featurizer(sim, pairs::Vector{Tuple{Int,Int}}) = FeaturesPairs(pairs)
featurizer(sim, features::Function) = features

function featurizer(sim, features::Symbol)
    @assert features == :all
    length(getcoords(sim)) > 100 && @warn "Computing _all_ pairwise distances for a bigger (>100 atoms) molecule. Try using a cutoff by setting features::Number in OpenMMSimulatioon"
    return FeaturesAll()
end




# implementation of some featurizers

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

function FeaturesPairs(sim::OpenMMSimulation, maxdist::Number, atomfilter::Function)
    pairs = local_atom_pairs(sim.pysim, maxdist; atomfilter=atomfilter)
    return FeaturesPairs(pairs)
end

# N random pairs
function FeaturesRandomPairs(sim::OpenMMSimulation, features::Int)
    n = natoms(sim)
    @assert features <= (n * n - 1) / 2
    p = Set{Tuple{Int,Int}}()
    while length(p) < features
        i = rand(1:n)
        j = rand(1:n)
        i < j || continue
        push!(p, (i, j))
    end
    pairs = sort(collect(p))
    FeaturesPairs(pairs)
end



""" generate `n` random inintial points for the simulation `mm` """
function randx0(sim::OpenMMSimulation, n)
    x0 = hcat(getcoords(sim))
    xs = propagate(sim, x0, n)
    return dropdims(xs, dims=3)
end


" compute the lagtime in ps "
lagtime(sim::OpenMMSimulation) = sim.step * sim.steps
dim(sim::OpenMMSimulation) = return length(getcoords(sim))
defaultmodel(sim::OpenMMSimulation; kwargs...) = ISOKANN.pairnet(; kwargs...)

pdbfile(s::OpenMMSimulation) = pdbfile(s.pdb)
pdbfile(str::String) = str
function pdbfile(pdb::PyObject)
    file = tempname()
    pdb.writeFile(pdb.topology, pdb.positions, file)
    file
end

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
function propagate(s::OpenMMSimulation, x0::AbstractMatrix{T}, ny; stepsize=s.step, steps=s.steps, nthreads=s.nthreads, mmthreads=s.mmthreads, momenta=s.momenta) where {T}
    s.mmthreads == "gpu" && CUDA.has_cuda() && CUDA.reclaim()
    dim, nx = size(x0)
    xs = repeat(x0, outer=[1, ny])
    xs = permutedims(reinterpret(Tuple{T,T,T}, xs))
    ys = @pycall py"threadedrun"(xs, s.pysim, stepsize, steps, nthreads, mmthreads, momenta)::Vector{Float32}
    ys = reshape(ys, dim, nx, ny)
    ys = permutedims(ys, (1, 3, 2))
    checkoverflow(ys)  # control the simulated data for NaNs and too large entries and throws an error
    return ys#convert(Array{Float32,3}, ys)
end

#propagate(s::OpenMMSimulation, x0::CuArray, ny; nthreads=Threads.nthreads()) = cu(propagate(s, collect(x0), ny; nthreads))

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
function trajectory(s::OpenMMSimulation; x0::AbstractVector{T}=getcoords(s), steps=s.steps, saveevery=1, stepsize=s.step, mmthreads=s.mmthreads, momenta=s.momenta) where {T}
    x0 = reinterpret(Tuple{T,T,T}, x0)
    xs = py"trajectory"(s.pysim, x0, stepsize, steps, saveevery, mmthreads, momenta)
    xs = permutedims(xs, (3, 2, 1))
    xs = reshape(xs, :, size(xs, 3))
    return xs
end

@deprecate trajectory(s, x0, steps, saveevery; kwargs...) trajectory(s; x0, steps, saveevery, kwargs...)

function ISOKANN.laggedtrajectory(s::OpenMMSimulation, n_lags, steps_per_lag=s.steps; x0=getcoords(s), keepstart=false)
    steps = steps_per_lag * n_lags
    saveevery = steps_per_lag
    xs = trajectory(s, x0, steps, saveevery)
    return keepstart ? xs : xs[:, 2:end]
end

getcoords(sim::OpenMMSimulation) = getcoords(sim.pysim, sim.momenta)::Vector{Float64}
setcoords(sim::OpenMMSimulation, coords) = setcoords(sim.pysim, coords, sim.momenta)

getcoords(sim::PyObject, momenta) = py"get_numpy_state($sim.context, $momenta).flatten()"

natoms(sim::OpenMMSimulation) = div(length(getcoords(sim)), 3)

function minimize(sim::OpenMMSimulation, coords, iter=100)
    setcoords(sim, coords)
    sim.pysim.minimizeEnergy(maxIterations=iter)
    return getcoords(sim)
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
    v = py"set_random_velocities($(sim.pysim.context), $(sim.mmthreads))"
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
    sys = sim.pysim.system

    f = sim.pysim.context.getState(getForces=true).getForces(asNumpy=true)
    f = f.value_in_unit(f.unit) |> permutedims
    #m = [sys.getParticleMass(i - 1)._value for i in 1:sys.getNumParticles()]
    #f ./= m'
    f = vec(f)
    @assert(!any(isnan.(f)))
    f
end

function potential(sim::OpenMMSimulation, x)
    CUDA.reclaim()
    setcoords(sim, x)
    v = sim.pysim.context.getState(getEnergy=true).getPotentialEnergy()
    v = v.value_in_unit(v.unit)
end

### PYTHON CODE


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

function atomfilter(atom)
    !(
        atom.element.symbol == "H" ||
        atom.residue.name in ["HOH", "NA", "CL"]
    )
end

function local_atom_pairs(pysim::PyObject, radius; atomfilter=atomfilter)
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
    pdb
    ligand
    forcefields
    temp
    friction
    step
    steps
    features
    nthreads
    mmthreads
end

JLD2.writeas(::Type{OpenMMSimulation}) = OpenMMSimulationSerialized

Base.convert(::Type{OpenMMSimulationSerialized}, a::OpenMMSimulation) =
    OpenMMSimulationSerialized(
        a.pdb,
        a.ligand,
        a.forcefields,
        a.temp,
        a.friction,
        a.step,
        a.steps,
        a.features,
        a.nthreads,
        a.mmthreads)

Base.convert(::Type{OpenMMSimulation}, a::OpenMMSimulationSerialized) =
    OpenMMSimulation(;
        pdb=a.pdb,
        ligand=a.ligand,
        forcefields=a.forcefields,
        temp=a.temp,
        friction=a.friction,
        step=a.step,
        steps=a.steps,
        features=a.features,
        nthreads=a.nthreads,
        mmthreads=a.mmthreads)

function Base.show(io::IO, mime::MIME"text/plain", sim::OpenMMSimulation)#
    featstr = if sim.features isa Vector{Tuple{Int,Int}}
        "{$(length(sim.features)) pairwise distances}"
    else
        sim.features
    end
    println(
        io, """
        OpenMMSimulation(;
            pdb="$(sim.pdb)",
            ligand="$(sim.ligand)",
            forcefields=$(sim.forcefields),
            temp=$(sim.temp),
            friction=$(sim.friction),
            step=$(sim.step),
            steps=$(sim.steps),
            features=$featstr)
        with $(div(length(getcoords(sim)),3)) atoms"""
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
    dt = step(sim) # sim,step is in picosecond, we calculate in nanoseconds
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
    dt = step(sim)
    gamma = friction(sim)

    sigma = sqrt(2 * gamma * kB * temp(sim))
    m = repeat(masses(sim), inner=3)

    x = copy(x0)
    g = 0.

    for i in 1:steps
        g += od_langevin_step_girsanov!(x, F, m, sigma, dt, u, g)
    end

    return x, g
end

function od_langevin_step_girsanov!(x, F, m, sigma, dt, u, g)
    dB = randn(length(x))
    ux = u(x)
    @. x += 1 / m * ((F + sigma * ux) * dt + sigma * sqrt(dt) * dB)
    dg = 1/2 * dt * dot(ux, ux) + sqrt(dt) * dot(ux, dB)
    return dg
end

end #module
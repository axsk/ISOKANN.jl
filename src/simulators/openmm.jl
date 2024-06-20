module OpenMM

using PyCall, CUDA
using LinearAlgebra: norm

import JLD2
import ISOKANN: ISOKANN, IsoSimulation,
    propagate, dim, randx0,
    featurizer, defaultmodel,
    savecoords, getcoords, force, pdb,
    force, potential, lagtime

export OpenMMSimulation, FORCE_AMBER, FORCE_AMBER_IMPLICIT

DEFAULT_PDB = "$(@__DIR__)/../../data/systems/alanine dipeptide.pdb"
FORCE_AMBER = ["amber14-all.xml"]
FORCE_AMBER_IMPLICIT = ["amber14-all.xml", "implicit/obc2.xml"]

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


function featurizer(sim::OpenMMSimulation)
    if sim.features isa (Vector{Int})
        ix = vec([1, 2, 3] .+ ((sim.features .- 1) * 3)')
        return ISOKANN.pairdistfeatures(ix)
    elseif sim.features isa (Vector{Tuple{Int,Int}}) # local pairwise distances
        inds = sim.features
        return coords -> ISOKANN.pdists(coords, inds)
    elseif sim.features == :all
        return ISOKANN.flatpairdists
    else
        error("unknown featurizer")
    end
end

""" generate `n` random inintial points for the simulation `mm` """
function randx0(sim::OpenMMSimulation, n)
    x0 = hcat(getcoords(sim))
    xs = propagate(sim, x0, n)
    return dropdims(xs, dims=3)
end

lagtime(sim::OpenMMSimulation) = sim.step * sim.steps
dim(sim::OpenMMSimulation) = return length(getcoords(sim))
defaultmodel(sim::OpenMMSimulation; kwargs...) = ISOKANN.pairnet(sim; kwargs...)
pdb(s::OpenMMSimulation) = s.pdb

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
    s.mmthreads == "gpu" && CUDA.reclaim()
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
function trajectory(s::OpenMMSimulation, x0::AbstractVector{T}=getcoords(s), steps=s.steps, saveevery=1; stepsize=s.step, mmthreads=s.mmthreads, momenta=s.momenta) where {T}
    x0 = reinterpret(Tuple{T,T,T}, x0)
    xs = py"trajectory"(s.pysim, x0, stepsize, steps, saveevery, mmthreads, momenta)
    xs = permutedims(xs, (3, 2, 1))
    xs = reshape(xs, :, size(xs, 3))
    return xs
end

getcoords(sim::OpenMMSimulation) = getcoords(sim.pysim, sim.momenta)#::Vector
setcoords(sim::OpenMMSimulation, coords) = setcoords(sim.pysim, coords, sim.momenta)

getcoords(sim::PyObject, momenta) = py"get_numpy_state($sim.context, $momenta).flatten()"

function minimize(sim::OpenMMSimulation, coords, iter=100)
    setcoords(sim, coords)
    sim.pysim.minimizeEnergy(maxIterations=iter)
    return getcoords(sim)
end

function setcoords(sim::PyObject, coords::AbstractVector{T}, momenta) where {T}
    t = reinterpret(Tuple{T,T,T}, Array(coords))
    if momenta
        n = length(t) ÷ 2
        x, v = t[1:n], t[n+1:end]
        sim.context.setPositions(x)
        sim.context.setVelocities(v)
    else
        sim.context.setPositions(t)
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
            features=$(sim.features))"""
    )
end

end #module





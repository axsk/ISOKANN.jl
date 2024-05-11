module OpenMM

using PyCall, CUDA

import ISOKANN: ISOKANN, IsoSimulation,
    propagate, dim, randx0,
    featurizer, defaultmodel,
    savecoords, getcoords, force, pdb,
    force, potential, lagtime

import JLD2

export OpenMMSimulation, FORCE_AMBER, FORCE_AMBER_IMPLICIT

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

""" generate `n` random inintial points for the simulation `mm` """
function randx0(sim::OpenMMSimulation, n)
    x0 = stack([getcoords(sim)])
    return reshape(propagate(sim, x0, n), :, n)
end

function dim(sim::OpenMMSimulation)
    return sim.pysim.system.getNumParticles() * 3
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

function defaultmodel(sim::OpenMMSimulation; nout, kwargs...)
    ISOKANN.pairnet(sim; nout, kwargs...)
end

DEFAULT_PDB = "$(@__DIR__)/../../data/systems/alanine dipeptide.pdb"

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
    nthreads=Threads.nthreads(),
    mmthreads=1,
    addwater=false,
    padding=3,
    ionicstrength=0.15,
    forcefield_kwargs=Dict(),
    momenta=false)

    pysim = @pycall py"defaultsystem"(pdb, ligand, forcefields, temp, friction, step, minimize; addwater, padding, ionicstrength, forcefield_kwargs)::PyObject
    if features isa Number
        radius = features
        features = calphas_and_spheres(pdb, pysim, radius)
        features = remove_atom_pairs(pysim, "H", features)
    end
    return OpenMMSimulation(pysim::PyObject, pdb, ligand, forcefields, temp, friction, step, steps, features, nthreads, mmthreads, momenta)
end

localpdistinds(sim::OpenMMSimulation, radius) = localpdistinds(sim.pysim, radius)
localpdistinds(pysim::PyObject, radius) = ISOKANN.localpdistinds(getcoords(pysim, false), radius)


FORCE_AMBER = ["amber14-all.xml"]
#FORCE_AMBER_EXPLICIT = ["amber14-all.xml", "amber14/tip3pfb.xml"]
FORCE_AMBER_IMPLICIT = ["amber14-all.xml", "implicit/obc2.xml"]

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
    CUDA.reclaim()
    dim, nx = size(x0)
    xs = repeat(x0, outer=[1, ny])
    xs = permutedims(reinterpret(Tuple{T,T,T}, xs))
    ys = @pycall py"threadedrun"(xs, s.pysim, stepsize, steps, nthreads, mmthreads, momenta)::PyArray
    ys = reshape(ys, dim, nx, ny)
    ys = permutedims(ys, (1, 3, 2))
    checkoverflow(ys)  # control the simulated data for NaNs and too large entries and throws an error
    return convert(AbstractArray{Float32}, ys)
end

propagate(s::OpenMMSimulation, x0::CuArray, ny; nthreads=Threads.nthreads()) = cu(propagate(s, collect(x0), ny; nthreads))

struct OpenMMOverflow{T} <: Exception where {T}
    result::T
    select::Vector{Bool}
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
function trajectory(s::OpenMMSimulation, x0::AbstractVector{T}, steps=s.steps, saveevery=1; stepsize=s.step, mmthreads=s.mmthreads) where {T}
    x0 = reinterpret(Tuple{T,T,T}, x0)
    xs = py"trajectory"(s.pysim, x0, stepsize, steps, saveevery, mmthreads)
    xs = permutedims(xs, (3, 2, 1))
    xs = reshape(xs, :, size(xs, 3))
    return xs
end

getcoords(sim::OpenMMSimulation) = getcoords(sim.pysim, sim.momenta)#::Vector
setcoords(sim::OpenMMSimulation, coords) = setcoords(sim.pysim, coords, sim.momenta)

function getcoords(sim::PyObject, momenta)
    st = sim.context.getState(getPositions=true, getVelocities=momenta)
    x = st.getPositions(asNumpy=true).flatten()
    if !momenta
        return x
    else
        v = st.getVelocities(asNumpy=true).flatten()
        return vcat(x, v)
    end
end

function getvelocities(sim::PyObject)
    py"$sim.context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(nanometer/picosecond).flatten()"
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

function force(sim::OpenMMSimulation, x)
    CUDA.reclaim()
    sim = sim.pysim
    setcoords(sim, x)
    f = sim.context.getState(getForces=true).getForces(asNumpy=true)
    f = f.value_in_unit(f.unit) |> permutedims |> vec
end

force(sim::OpenMMSimulation, x::CuArray) = force(sim, Array(x)) |> cu
potential(sim::OpenMMSimulation, x::CuArray) = potential(sim, Array(x))

function potential(sim::OpenMMSimulation, x)
    CUDA.reclaim()
    sim = sim.pysim
    setcoords(sim, x)
    v = sim.context.getState(getEnergy=true).getPotentialEnergy()
    v = v.value_in_unit(v.unit)
end

lagtime(sim::OpenMMSimulation) = sim.step * sim.steps

### PYTHON CODE

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


### TESTS

function test_getsetcoords(sim)
    x = getcoords(sim)
    bak = copy(x)
    setcoords(sim, x)
    @assert getcoords(sim) == bak
end

function test_openmm(replicas=2)
    sim = OpenMMSimulation()
    x0 = stack([getcoords(sim) for _ in 1:replicas])
    propagate(sim, x0, 3, nthreads=2)
end

function calphas(pdbfile)
    inds = Int[]
    for l in readlines(pdbfile)
        s = split(l)
        length(s) < 3 && continue
        s[3] == "CA" && push!(inds, parse(Int, s[2]))
    end
    return inds
end

function calphas_and_spheres(pdbfile::String, pysim::PyObject, radius)
    cind = calphas(pdbfile)
    cpairs = [(x, y) for x in cind, y in cind][ISOKANN.halfinds(length(cind))]
    rpairs = localpdistinds(pysim, radius)
    return unique([rpairs; cpairs])
end

calphas_and_spheres(sim::OpenMMSimulation, radius) = calphas_and_spheres(sim.pdb, sim.pysim, radius)

function remove_atom_pairs(pysim::PyObject, symbol, pairs)
    symbols = [a.element.symbol for a in pysim.topology.atoms()]
    inds = findall(symbols .== symbol)
    pairs = filter(p -> !(p[1] in inds || p[2] in inds), pairs)
    return pairs
end

function filteratoms(pdbfile, pred)
    inds = Int[]
    for l in readlines(pdbfile)
        s = split(l)
        length(s) < 3 && continue
        s[1] == "ATOM" || continue
        pred(s[3]) && push!(inds, parse(Int, s[2]))
    end
    return inds
end


function residue_atoms(sim::OpenMMSimulation)
    [res.name => [parse(Int, a.id) for a in res.atoms()] for res in sim.pysim.topology.residues()]
end

noHatoms(pdbfile) = filteratoms(pdbfile, !contains("H"))

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


end #module





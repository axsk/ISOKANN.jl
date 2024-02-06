module OpenMM

using PyCall

import ..ISOKANN: ISOKANN, propagate, dim, randx0, featurizer, defaultmodel, savecoords

export OpenMMSimulation

###

""" A Simulation wrapping the Python OpenMM Simulation object """
struct OpenMMSimulation
    pysim::PyObject
    steps::Int
    features::Vector{Int}
end

""" generate `n` random inintial points for the simulation `mm` """
function randx0(mm::OpenMMSimulation, n)
    x0 = stack([getcoords(mm.pysim)])
    return reshape(propagate(mm, x0, n), :, n)
end

function dim(mm::OpenMMSimulation)
    return mm.pysim.system.getNumParticles() * 3
end

function featurizer(sim::OpenMMSimulation)
    ix = vec([1, 2, 3] .+ ((sim.features .- 1) * 3)')
    n, features = ISOKANN.pairdistfeatures(ix)
end

function defaultmodel(sim::OpenMMSimulation; nout, kwargs...)
    ISOKANN.pairnet(sim; nout, kwargs...)
end

""" Basic construction of a OpenMM Simulation, following the OpenMM documentation example """
function OpenMMSimulation(;
    pdb="$(ENV["HOME"])/.julia/conda/3/share/openmm/examples/input.pdb",
    forcefields=["amber14-all.xml", "amber14/tip3pfb.xml"],
    temp=298,
    friction=1,
    step=0.002,
    steps=1,
    features=calphas(pdb),
    minimize=false)

    pysim = @pycall py"defaultsystem"(pdb, forcefields, temp, friction, step, minimize)::PyObject
    return OpenMMSimulation(pysim, steps, features)
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

"""
function propagate(s::OpenMMSimulation, x0::AbstractMatrix{T}, ny; nthreads=Threads.nthreads(), mmthreads=1) where {T}
    dim, nx = size(x0)
    xs = repeat(x0, outer=[1, ny])
    xs = permutedims(reinterpret(Tuple{T,T,T}, xs))
    ys = @pycall py"threadedrun"(xs, s.pysim, s.steps, nthreads, mmthreads)::PyArray
    return reshape(ys, dim, nx, ny)
end

getcoords(sim::OpenMMSimulation) = getcoords(sim.pysim)
setcoords(sim::OpenMMSimulation, coords) = setcoords(sim.pysim, coords)

function getcoords(sim::PyObject)
    py"$sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer).flatten()"
end

function setcoords(sim::PyObject, coords::AbstractArray{T}) where {T}
    sim.context.setPositions(reinterpret(Tuple{T,T,T}, coords))
end


### PYTHON CODE

function __init__()
    # install / load OpenMM
    pyimport_conda("openmm", "openmm", "conda-forge")
    pyimport_conda("joblib", "joblib")

    @pyinclude("$(@__DIR__)/mopenmm.py")
end

function savecoords(sim, coords, path)
    s = sim.pysim
    p = py"pdbfile.PDBFile"
    #file = py"open("$(path)', 'w')"
    file = py"open"(path, "w")
    p.writeHeader(s.topology, file)
    for (i, coords) in enumerate(eachcol(coords))
        pos = reinterpret(Tuple{Float64,Float64,Float64}, coords .* 10)
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

end
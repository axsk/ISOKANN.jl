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

function defaultmodel(sim::OpenMMSimulation; nout)
    ISOKANN.pairnet(sim; nout)
end

""" Basic construction of a OpenMM Simulation, following the OpenMM documentation example """
function OpenMMSimulation(;
    pdb="$(ENV["HOME"])/.julia/conda/3/share/openmm/examples/input.pdb",
    forcefields=["amber14-all.xml", "amber14/tip3pfb.xml"],
    temp=298,
    friction=1,
    step=0.004,
    steps=1,
    features=calphas(pdb))

    pysim = @pycall py"defaultsystem"(pdb, forcefields, temp, friction, step)::PyObject
    return OpenMMSimulation(pysim, steps, features)
end


""" multi-threaded propagation of an `OpenMMSimulation` """
function propagate(s::OpenMMSimulation, x0::AbstractMatrix, ny; nthreads=1)
    dim, nx = size(x0)
    xs = repeat(x0, outer=[1, ny])
    xs = permutedims(reinterpret(Tuple{Float64,Float64,Float64}, xs))
    ys = @pycall py"threadedrun"(xs, s.pysim, s.steps, nthreads)::PyArray
    return reshape(ys, dim, nx, ny)
end

getcoords(sim::OpenMMSimulation) = getcoords(sim.pysim)
setcoords(sim::OpenMMSimulation, coords) = setcoords(sim.pysim, coords)

function getcoords(sim::PyObject)
    py"$sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer).flatten()"
end

function setcoords(sim::PyObject, coords)
    sim.context.setPositions(reinterpret(Tuple{Float64,Float64,Float64}, coords))
end


### PYTHON CODE

function __init__()
    # install / load OpenMM
    pyimport_conda("openmm", "openmm", "conda-forge")
    pyimport_conda("joblib", "joblib")

    py"""

    from joblib import Parallel, delayed
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    from sys import stdout
    import numpy as np

    def threadedrun(xs, sim, steps, nthreads):
        def singlerun(i):
            c = Context(sim.system, copy.copy(sim.integrator))
            c.setPositions(xs[i])
            c.setVelocitiesToTemperature(sim.integrator.getTemperature())
            c.getIntegrator().step(steps)
            return c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)

        out = Parallel(n_jobs=nthreads, prefer="threads")(delayed(singlerun)(i) for i in range(len(xs)))
        return np.array(out).flatten()

    # from the OpenMM documentation
    def defaultsystem(pdb, forcefields, temp, friction, step):
        pdb = PDBFile(pdb)
        forcefield = ForceField(*forcefields)
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
                nonbondedCutoff=1*nanometer, constraints=HBonds)
        integrator = LangevinMiddleIntegrator(temp*kelvin, friction/picosecond, step*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        return simulation

    """
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

function test_openmm()
    sim = OpenMMSimulation()
    x0 = stack([getcoords(sim) for _ in 1:2])
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
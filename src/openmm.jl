
using PyCall
import ISOKANN: propagate

# install / load OpenMM
pyimport_conda("openmm", "openmm", "conda-forge")
pyimport_conda("joblib", "joblib")

# load into namespace
py"""
from joblib import Parallel, delayed
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import itertools

def unpack(out):
    return list(itertools.chain.from_iterable(out))
"""
# still allocating, but 4x as fast as anything else i could find
mypyvec(out) = reinterpret(Float64, pycall(py"unpack", Vector{Tuple{Float64,Float64,Float64}}, out))

nanometer = py"nanometer"

""" A Simulation wrapping the Python OpenMM Simulation object """
struct OpenMMSimulation
    pysim::PyObject
    steps::Int
end

###

function propagate(s::OpenMMSimulation, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, nx, ny)
    for i in 1:nx, j in 1:ny
        ys[:, i, j] = propagate(s, x0[:, i])
    end
    return ys
end

# producing nans with nthreads = 1, crashing for nthreads > 1
function propagate_threaded(s::OpenMMSimulation, x0::AbstractMatrix, ny; nthreads=1)
    xs = repeat(x0, outer=[1, ny])
    dim, nx = size(x0)

    xs = PyReverseDims(reinterpret(Tuple{Float64,Float64,Float64}, xs))
    steps = s.steps
    sim = s.pysim
    n = nx * ny

    py"""
    def singlerun(i):
        x = $xs[i]
        sim = $sim
        steps = $steps
        c = Context(sim.system, copy.copy(sim.integrator))
        c.setPositions(x)
        c.setVelocitiesToTemperature(sim.integrator.getTemperature())
        c.getIntegrator().step(steps)
        return c.getState(getPositions=True).getPositions().value_in_unit(nanometer)

    out = Parallel(n_jobs=$nthreads, prefer="threads")(delayed(singlerun)(i) for i in range($n))
    """
    return reshape(mypyvec(py"out"o), dim, nx, ny)


    #zs = reshape(reinterpret(Float64, permutedims(py"out")), dim, nx, ny)
    #return zs
end

function propagate(s::OpenMMSimulation, x0)
    setcoords(s.pysim, x0)
    s.pysim.step(s.steps)
    getcoords(s.pysim)
end

function getcoords(sim::PyObject)
    x = sim.context.getState(getPositions=true).getPositions().value_in_unit(nanometer)
    reinterpret(Float64, x)
end

function setcoords(sim::PyObject, coords)
    x = reinterpret(Tuple{Float64,Float64,Float64}, coords) * nanometer
    sim.context.setPositions(x)
end

###

""" Basic construction of a OpenMM Simulation, following the OpenMM documentation example """
function openmm_examplesys(;
    temp=300,
    friction=1,
    step=0.004,
    pdb="/home/htc/bzfsikor/.julia/conda/3/share/openmm/examples/input.pdb",
    forcefields=["amber14-all.xml", "amber14/tip3pfb.xml"],
    steps=1)

    py"""
    pdb = PDBFile($pdb)
    forcefield = ForceField(*$forcefields)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
            nonbondedCutoff=1*nanometer, constraints=HBonds)
    integrator = LangevinMiddleIntegrator($temp*kelvin, $friction/picosecond, $step*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    """
    #
    # simulation.reporters.append(PDBReporter('output.pdb', 1000))
    # simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
    #        potentialEnergy=True, temperature=True))
    # simulation.step(10000)

    sim = py"simulation"
    return OpenMMSimulation(sim, steps)
end

###

function test_getsetcoords(sim)
    x = getcoords(sim)
    bak = copy(x)
    setcoords(sim, x)
    @assert getcoords(sim) == bak
end

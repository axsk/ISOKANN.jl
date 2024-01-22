
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
"""

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

function propagate_threaded(s::OpenMMSimulation, x0::AbstractMatrix, ny; nthreads=1)
    zs = repeat(x0, outer=[1, ny])
    dim, nx = size(x0)

    zs = PyReverseDims(reinterpret(Tuple{Float64,Float64,Float64}, zs))
    steps = s.steps
    sim = s.pysim
    n = nx * ny

    py"""

    def singlerun(i):
        s = copy.copy($sim)  # TODO: this is not enough
        s.context = copy.copy(s.context)
        s.context.setPositions($zs[i] * nanometer)
        s.step($steps)
        z = s.context.getState(getPositions=True).getPositions().value_in_unit(nanometer)
        return z

    out = Parallel(n_jobs=$nthreads, prefer="threads")(
        delayed(singlerun)(i) for i in range($n))
    """

    zs = py"out"
    zs = reinterpret(Float64, permutedims(zs))
    zs = reshape(zs, dim, nx, ny)

    return zs
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


using PyCall
import ISOKANN: propagate

py"""
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
"""

nanometer = py"nanometer"

struct OpenMMSimulation
    pysim::PyObject
    steps::Int
end

###

function propagate(sim::OpenMMSimulation, x0)
    steps = sim.steps
    pysim = sim.pysim
    setcoords(pysim, x0)
    pysim.step(steps)
    getcoords(pysim)
end

function propagate(sim::OpenMMSimulation, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, nx, ny)
    for i in 1:nx, j in 1:ny
        ys[:, i, j] = propagate(sim, x0[:, i])
    end
    return ys
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

function openmm_examplesys(;
    temp=300,
    friction=1,
    step=0.004,
    pdb="/home/htc/bzfsikor/.julia/conda/3/share/openmm/examples/input.pdb",
    forcefields=["amber14-all.xml", "amber14/tip3pfb.xml"],
    steps=100)

    py"""
    pdb = PDBFile($pdb)
    forcefield = ForceField(*$forcefields)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
            nonbondedCutoff=1*nanometer, constraints=HBonds)
    integrator = LangevinMiddleIntegrator($temp*kelvin, $friction/picosecond, $step*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    """
    # simulation.minimizeEnergy()
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

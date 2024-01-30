from joblib import Parallel, delayed
import mdtraj
from openmm import *
from openmm.app import *
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
def defaultsystem(pdb, forcefields, temp, friction, step, minimize):
    pdb = PDBFile(pdb)
    forcefield = ForceField(*forcefields)
    system = forcefield.createSystem(
      pdb.topology, 
      nonbondedMethod=CutoffNonPeriodic,
      nonbondedCutoff=1*nanometer,
      constraints=None)
    integrator = LangevinIntegrator(temp*kelvin, friction/picosecond, step*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    if minimize:
        simulation.minimizeEnergy()
    return simulation

import mdtraj
from openmm import *
from openmm.app import *
from openmm.unit import *

def enric(pdbfile = "data/enric/native.pdb"):
  usemdtraj = True
  
  if usemdtraj:
    pdb = mdtraj.load(pdbfile)
    topology = pdb.topology.to_openmm()
  else:
    pdb = PDBFile(pdbfile)
    topology = pdb.topology
  
  forcefield = ForceField('amber99sbildn.xml', 'amber99_obc.xml')
  system = forcefield.createSystem(topology, nonbondedMethod=app.CutoffNonPeriodic)
  integrator = LangevinIntegrator(330*kelvin, 1.0/picosecond, 2*femtosecond)
  simulation = Simulation(topology, system, integrator)
  if usemdtraj:
    simulation.context.setPositions(pdb.xyz[0]) 
  else:
    simulation.context.setPositions(pdb.positions)
  simulation.context.setVelocitiesToTemperature(330*kelvin)
  return simulation

sim = enric("yourpdbpathhere")
%time sim.step(500)


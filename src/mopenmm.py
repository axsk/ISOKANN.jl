from joblib import Parallel, delayed
import mdtraj
from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
import numpy as np
from functools import wraps
from time import time

def threadedrun(xs, sim, steps, nthreads):
    cp = sim.context.createCheckpoint()
    def singlerun(i):
  #      int = copy.copy(sim.integrator) 
        c = copy.copy(sim.context)
        c.loadCheckpoint(cp)

 #       c = Context(sim.system, int) # this is where the time is spent
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

def enric(pdbfile = "data/enric/native.pdb",
  forcefields = ['amber99sbildn.xml', 'amber99_obc.xml'] ):
  usemdtraj = True
  
  if usemdtraj:
    pdb = mdtraj.load(pdbfile)
    topology = pdb.topology.to_openmm()
  else:
    pdb = PDBFile(pdbfile)
    topology = pdb.topology
  
  forcefield = ForceField(*forcefields)
  system = forcefield.createSystem(topology, nonbondedMethod=app.CutoffNonPeriodic)
  integrator = LangevinIntegrator(330*kelvin, 1.0/picosecond, 2*femtosecond)
  simulation = Simulation(topology, system, integrator)
  if usemdtraj:
    simulation.context.setPositions(pdb.xyz[0]) 
  else:
    simulation.context.setPositions(pdb.positions)
  simulation.context.setVelocitiesToTemperature(330*kelvin)
  return simulation

def timing(f):
  @wraps(f)
  def wrap(*args, **kw):
    t0 = time()
    result = f(*args, **kw)
    t1 = time()
    print('func:%r took: %2.4f sec' % (f.__name__, t1-t0))
    return result
  return wrap

@timing 
def mysim(sim):
  sim.step(500)

def main():
  sim = enric()
  mysim(sim)

if __name__ == '__main__':
  main()

ff99 = ['amber99sbildn.xml', 'amber99_obc.xml']
ff14 = ["amber14-all.xml"]

pdb_nowater = "data/alanine-dipeptide-nowater av.pdb"
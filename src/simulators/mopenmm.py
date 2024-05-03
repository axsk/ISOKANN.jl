from joblib import Parallel, delayed
from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np

def threadedrun(xs, sim, stepsize, steps, nthreads, nthreadssim=1):
    context = sim.context
    if nthreadssim == 'gpu':
      platform = Platform.getPlatformByName('CUDA')
      platformargs = {}
    else:
      platform = context.getPlatform()
      platformargs = {'Threads': str(nthreadssim)}

    def singlerun(i):
        c = Context(context.getSystem(), copy.copy(context.getIntegrator()), platform, platformargs)

        c.setPositions(xs[i])
        c.setVelocitiesToTemperature(sim.integrator.getTemperature())
        c.getIntegrator().setStepSize(stepsize)
        try:
          c.getIntegrator().step(steps)
          x = c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
          return x
        except OpenMMException as e:
          print("Error integrating trajectory", e)
          x = c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
          x.fill(np.nan)
          return x


        return x

    out = Parallel(n_jobs=nthreads, prefer="threads")(delayed(singlerun)(i) for i in range(len(xs)))
    return np.array(out).flatten()

# from the OpenMM documentation
def defaultsystem(pdb, forcefields, temp, friction, step, minimize, platform='CPU', properties={'Threads': '1'}, addwater=False):
    platform = Platform.getPlatformByName(platform)
    pdb = PDBFile(pdb)
    forcefield = ForceField(*forcefields)
    modeller = Modeller(pdb.topology, pdb.positions)
    if addwater:
      modeller.addSolvent(forcefield)
    system = forcefield.createSystem(modeller.topology,
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=1*nanometer,
            constraints=None)
    integrator = LangevinMiddleIntegrator(temp*kelvin, friction/picosecond, step*picoseconds)
    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(simulation.integrator.getTemperature())
    if minimize:
        simulation.minimizeEnergy()
    return simulation

def test():
  ff99 = ['amber99sbildn.xml', 'amber99_obc.xml']
  ff14 = ["amber14-all.xml"]

  pdb_nowater = "data/alanine-dipeptide-nowater av.pdb"

  s = defaultsystem(pdb_nowater, ff14, 298, 1, 0.002, False, 'CPU', {'Threads':'1'})
  x0 = s.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
  threadedrun([x0], s, 500, 1)
  return s

"""
# FROM HERE ON ENRIC AND TEST STUFF
import mdtraj
from functools import wraps
from time import time
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

def testenric():
  sim = enric()
  mysim(sim)
"""
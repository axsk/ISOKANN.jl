from joblib import Parallel, delayed
from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np

def threadedrun(xs, sim, stepsize, steps, nthreads, nthreadssim=1):
    def singlerun(i):
        c = newcontext(sim.context, nthreadssim)

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

def trajectory(sim, x0, stepsize, steps, saveevery, mmthreads):
  n_states = steps // saveevery + 1
  trajectory = np.zeros((n_states,) + np.array(x0).shape)
  trajectory[0] = x0

  c = newcontext(sim.context, mmthreads)
  c.setPositions(x0)
  c.setVelocitiesToTemperature(c.getIntegrator().getTemperature())
  c.getIntegrator().setStepSize(stepsize)

  for n in range(1,n_states):
    c.getIntegrator().step(saveevery)
    trajectory[n] = get_numpy_pos(c)

  return trajectory

# from the OpenMM documentation
def defaultsystem(pdb, ligand, forcefields, temp, friction, step, minimize, platform='CPU', properties={'Threads': '1'}, addwater=False, padding=3, ionicstrength=0, forcefield_kwargs={}):
    platform = Platform.getPlatformByName(platform)
    pdb = PDBFile(pdb)

    if ligand != "":
        from openff.toolkit import Molecule
        from openmmforcefields.generators import SystemGenerator
        ligand_mol = Molecule.from_file(ligand)
        ligand_mol.assign_partial_charges(partial_charge_method="mmff94", use_conformers=ligand_mol.conformers)
        water_force_field = "amber/tip3p_standard.xml"
        ligand_force_field = "gaff-2.11"
        system_generator = SystemGenerator(
            forcefields=[*forcefields, water_force_field],
            small_molecule_forcefield=ligand_force_field,
            molecules=[ligand_mol],
            forcefield_kwargs=forcefield_kwargs)
        modeller = Modeller(pdb.topology, pdb.positions)
        lig_top = ligand_mol.to_topology()
        modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
        if addwater:
            modeller.addSolvent(system_generator.forcefield, model="tip3p",
                                padding=padding * angstroms,
                                positiveIon="Na+", negativeIon="Cl-",
                                ionicStrength=ionicstrength * molar, neutralize=True)
        system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
        integrator = LangevinMiddleIntegrator(temp * kelvin, friction / picosecond, step * picoseconds)
        simulation = Simulation(modeller.topology, system, integrator, platform=platform)

    else:
        forcefield = ForceField(*forcefields)
        modeller = Modeller(pdb.topology, pdb.positions)
        if addwater:
            water_force_field = "amber/tip3p_standard.xml"
            forcefield = ForceField(*forcefields, water_force_field)
            modeller.addSolvent(forcefield, model="tip3p",
                                padding=padding * angstroms,
                                positiveIon="Na+", negativeIon="Cl-",
                                ionicStrength=ionicstrength * molar, neutralize=True)
        system = forcefield.createSystem(modeller.topology,
                nonbondedMethod=CutoffNonPeriodic,
                nonbondedCutoff=1*nanometer,
                **forcefield_kwargs)
        integrator = LangevinMiddleIntegrator(temp*kelvin, friction/picosecond, step*picoseconds)
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)

    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(simulation.integrator.getTemperature())
    if minimize:
        simulation.minimizeEnergy()
    return simulation

def get_numpy_pos(context):
  return context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)

def newcontext(context, mmthreads):
  if mmthreads == 'gpu':
    platform = Platform.getPlatformByName('CUDA')
    platformargs = {}
  else:
    platform = context.getPlatform()
    platformargs = {'Threads': str(mmthreads)}
  c = Context(context.getSystem(), copy.copy(context.getIntegrator()), platform, platformargs)
  return c

def test():
  ff99 = ['amber99sbildn.xml', 'amber99_obc.xml']
  ff14 = ["amber14-all.xml"]

  pdb_nowater = "data/alanine-dipeptide-nowater av.pdb"

  s = defaultsystem(pdb_nowater, ff14, 298, 1, 0.002, False, 'CPU', {'Threads':'1'})
  x0 = s.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
  threadedrun([x0], s, 500, 1)
  return s

from joblib import Parallel, delayed
from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np

def threadedrun(xs, sim, stepsize, steps, nthreads, nthreadssim=1, withmomenta=False):
    def singlerun(i):
        c = newcontext(sim.context, nthreadssim)
        set_numpy_state(c, xs[i], withmomenta)
        c.getIntegrator().setStepSize(stepsize)

        try:
          c.getIntegrator().step(steps)
        except OpenMMException as e:
          print("Error integrating trajectory", e)
          return get_numpy_state(c, withmomenta).fill(np.nan)
        
        x = get_numpy_state(c, withmomenta)
        return x


    if nthreads > 1:
        out = Parallel(n_jobs=nthreads, prefer="threads")(delayed(singlerun)(i) for i in range(len(xs)))
    else:
        out = [singlerun(i) for i in range(len(xs))]
    return np.array(out).flatten()

def trajectory(sim, x0, stepsize, steps, saveevery, mmthreads, withmomenta):
  n_states = steps // saveevery + 1
  trajectory = np.zeros((n_states,) + np.array(x0).shape)
  trajectory[0] = x0

  c = newcontext(sim.context, mmthreads)

  if withmomenta:
    n = len(x0) // 2
    c.setPositions(x0[:n])
    c.setVelocities(x0[n:])
  else:
    c.setPositions(x0)
    c.setVelocitiesToTemperature(sim.integrator.getTemperature())

  c.getIntegrator().setStepSize(stepsize)

  for n in range(1,n_states):
    c.getIntegrator().step(saveevery)
    trajectory[n] = get_numpy_state(c, withmomenta)

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

    simulation.reporters.append(
        StateDataReporter(
            "openmmsimulation.log", 1, step=True,
            potentialEnergy=True, totalEnergy=True,
            temperature=True, speed=True,)
            )

    if minimize:
        simulation.minimizeEnergy(maxIterations=100)
    return simulation

def get_numpy_state(context, withmomenta):
    if withmomenta:
        state = context.getState(getPositions=True, getVelocities=True)
        x = np.concatenate([
            state.getPositions(asNumpy=True).value_in_unit(nanometer),
            state.getVelocities(asNumpy=True).value_in_unit(nanometer/picosecond)])
    else:
        x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
    return x

def set_numpy_state(context, x, withmomenta):
    if withmomenta:
        n = len(x) // 2
        context.setPositions(x[:n])
        context.setVelocities(x[n:])
    else:
        context.setPositions(x)
        context.setVelocitiesToTemperature(sim.integrator.getTemperature())

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

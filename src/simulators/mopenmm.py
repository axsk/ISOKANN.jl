from joblib import Parallel, delayed
from openmm import *
from openmm import app
from openmm.app import *
from openmm.unit import *
import numpy as np

def threadedrun(xs, sim, stepsize, steps, nthreads, withmomenta=False):
    def singlerun(i):
        if nthreads > 1:
            c = newcontext(sim.context)
        else:
            c = sim.context
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

  c = newcontext(sim.context)

  set_numpy_state(c, x0, withmomenta)
  c.getIntegrator().setStepSize(stepsize)

  for n in range(1,n_states):
    c.getIntegrator().step(saveevery)
    trajectory[n] = get_numpy_state(c, withmomenta)

  return trajectory

# from the OpenMM documentation
def defaultsystem(pdb, ligand, forcefields, temp, friction, step, minimize, mmthreads, 
addwater=False, padding=1, ionicstrength=0, forcefield_kwargs={}, 
flexibleConstraints = False, rigidWater=True, nonbondedMethod="auto", nonbondedCutoff=1, constraints= None):
    pdb = PDBFile(pdb)

    if mmthreads == 'gpu':
      platform = Platform.getPlatformByName('CUDA')
      platformargs = {}
    else:
      platform = Platform.getPlatformByName('CPU')
      platformargs = {'Threads': str(mmthreads)}


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
                                padding=padding * nanometer,
                                positiveIon="Na+", negativeIon="Cl-",
                                ionicStrength=ionicstrength * molar, neutralize=True)
        system = system_generator.create_system(modeller.topology, molecules=ligand_mol)


    else:
        forcefield = ForceField(*forcefields)
        modeller = Modeller(pdb.topology, pdb.positions)

        nonbondedMethod = parse_nonbondedMethod(nonbondedMethod, pdb, addwater)

        if addwater:
            water_force_field = "amber/tip3p_standard.xml"
            forcefield = ForceField(*forcefields, water_force_field)
            modeller.addSolvent(forcefield, model="tip3p",
                                padding=padding * nanometer,
                                positiveIon="Na+", negativeIon="Cl-",
                                ionicStrength=ionicstrength * molar, neutralize=True,
                                )

        system = forcefield.createSystem(modeller.topology,
                nonbondedMethod=nonbondedMethod,
                nonbondedCutoff=nonbondedCutoff*nanometer,
                removeCMMotion=False,
                flexibleConstraints=flexibleConstraints,
                rigidWater=rigidWater,
                constraints=parse_constraints(constraints),
                **forcefield_kwargs)

    integrator = LangevinMiddleIntegrator(temp*kelvin, friction/picosecond, step*picoseconds)
    simulation = Simulation(modeller.topology, system, integrator, platform, platformargs)

    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(simulation.integrator.getTemperature())

    # simulation.reporters.append(
    #    StateDataReporter(
    #        "openmmsimulation.log", 1, step=True,
    #        potentialEnergy=True, totalEnergy=True,
    #        temperature=True, speed=True,)
    #        )

    if minimize:
        simulation.minimizeEnergy()
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
        context.setVelocitiesToTemperature(context.getIntegrator().getTemperature())
    if has_barostat(context):
        set_box_from_positions(context, positions)

def set_box_from_current_positions(context, buffer=0.01):
    """
    Reads positions from the context, computes tight box + buffer, 
    and sets periodic box vectors accordingly.
    
    buffer: in nanometers (default 0.01 nm = 0.1 Å)
    """
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True)  # Quantity array
    
    # Convert positions to plain nm ndarray for calculation
    pos_nm = positions.value_in_unit(nanometer)
    
    mins = pos_nm.min(axis=0)
    maxs = pos_nm.max(axis=0)
    lengths = (maxs - mins + buffer)
    
    a = Vec3(lengths[0], 0, 0)
    b = Vec3(0, lengths[1], 0)
    c = Vec3(0, 0, lengths[2])
    
    context.setPeriodicBoxVectors(a * nanometer, b * nanometer, c * nanometer)

def set_box_from_positions(context, x, buffer=0.01):
    """Set the periodic box to tightly fit the positions in `x` + a small buffer (in nm)."""
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    lengths = (maxs - mins + buffer)  # box lengths in nm with buffer
    a = Vec3(lengths[0], 0, 0)
    b = Vec3(0, lengths[1], 0)
    c = Vec3(0, 0, lengths[2])
    context.setPeriodicBoxVectors(a, b, c)

def has_barostat(context):
    for force in context.getSystem().getForces():
        if isinstance(force, (openmm.MonteCarloBarostat, openmm.MonteCarloAnisotropicBarostat)):
            return True
    return False


def set_random_velocities(context):
    context.setVelocitiesToTemperature(context.getIntegrator().getTemperature())
    v = context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(nanometer/picosecond).flatten()
    return v


def newcontext(context):
  return Context(context.getSystem(), copy.copy(context.getIntegrator()), context.getPlatform())

def test():
  ff99 = ['amber99sbildn.xml', 'amber99_obc.xml']
  ff14 = ["amber14-all.xml"]

  pdb_nowater = "data/alanine-dipeptide-nowater av.pdb"

  s = defaultsystem(pdb_nowater, ff14, 298, 1, 0.002, False, 'CPU', {'Threads':'1'})
  x0 = s.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
  threadedrun([x0], s, 500, 1)
  return s

def parse_nonbondedMethod(method: str, pdb, addwater):
    method_map = {
        "NoCutoff": NoCutoff,
        "CutoffNonPeriodic": CutoffNonPeriodic,
        "CutoffPeriodic": CutoffPeriodic,
        "Ewald": Ewald,
        "PME": PME,
        "LJPME": LJPME,
        "auto": CutoffNonPeriodic if pdb.getTopology().getPeriodicBoxVectors() is None and not addwater else CutoffPeriodic
    }

    try:
        return method_map[method]
    except KeyError:
        raise ValueError(f"Invalid method name: {method}. Must be one of {', '.join(method_map.keys())}.")

def parse_constraints(constraints: str):
    constraint_map = {
        None: None,
        "None": None,
        "HBonds": HBonds,
        "AllBonds": AllBonds,
        "HAngles": HAngles,
    }
    return constraint_map[constraints]
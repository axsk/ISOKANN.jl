from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

a,b,c = (10,10,16.5)
psf = CharmmPsfFile('/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/openmm/step5_input.psf',
                    unitCellDimensions=(a,b,c))
pdb = PDBFile('/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/openmm/step5_input.pdb')

params = CharmmParameterSet('/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/toppar/par_all36m_prot.prm', '/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/toppar/top_all36_prot.rtf',
                       '/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/toppar/par_all36_lipid.prm', '/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/toppar/top_all36_lipid.rtf',
                       '/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/toppar/toppar_water_ions.str', '/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/7v7/7v7.prm', '/data/numerik/ag_cmd/arostami/charmm/charmm-gui(6)/charmm-gui-2501706787/7v7/7v7.rtf')

system = psf.createSystem(params, nonbondedMethod=CutoffPeriodic,
         nonbondedCutoff=1*nanometer, constraints=HBonds)

integrator = LangevinIntegrator(310*kelvin,
                                1/picosecond,
                                0.002*picoseconds)

platform = Platform.getPlatformByName('CUDA')

simulation = Simulation(psf.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation

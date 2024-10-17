from openmm.app import *
from openmm import *
from openmm.unit import *

pdbfile = "data/alanine-dipeptide-nowater av.pdb"

pdb = PDBFile(pdbfile) # alaninedipepdite without water
forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, model="tip3p",
    padding=20 * angstroms,
    positiveIon="Na+", negativeIon="Cl-",
    ionicStrength=0 * molar, neutralize=True,
    )
system = forcefield.createSystem(modeller.topology,
    nonbondedMethod=CutoffPeriodic,
    nonbondedCutoff=1*nanometer,
    removeCMMotion=False, # this made no difference
 )

integrator = LangevinMiddleIntegrator(310*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(modeller.topology, system, integrator)

simulation.context.setPositions(modeller.positions)

simulation.minimizeEnergy()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nparticles = 12

st = simulation.context.getState(getForces=True, getPositions=True, enforcePeriodicBox=True)
last_positions = st.getPositions(asNumpy=True)._value[-nparticles:]
last_forces = st.getForces(asNumpy=True)._value[-nparticles:]

# Create a 3D scatter plot
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111, projection='3d')

# Plot the positions as scatter points
ax.scatter(last_positions[:, 0], last_positions[:, 1], last_positions[:, 2], color='blue', label='Positions')

# Plot the forces as arrows
for i in range(nparticles):
    ax.quiver(last_positions[i, 0], last_positions[i, 1], last_positions[i, 2],
              last_forces[i, 0], last_forces[i, 1], last_forces[i, 2],
              color='red', length=0.0001)

# Show the plot
plt.show()
plt.savefig("out.png")

PDBFile.writeFile(simulation.topology, st.getPositions(asNumpy=True), "out.pdb")
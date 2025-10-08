# INSTALLATION INSTRUCTIONS

# download openmm-dlext
# activate the openmm env with openmm to patch
# conda install cmake swig setuptools_scm
# export CUDAToolkit_ROOT=/software/CUDA/cuda-12.5
# export CONDA_PREFIX=...
# mkdir build
# cd build
# cmake .. -B . -DOPENMM_DIR=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DDLExt_BUILD_CUDA=ON -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT
# make
# make install

import numpy as np
import cupy as cp
import openmm
from openmm import unit
from openmm.app import Simulation, Topology, Element
import openmm.dlext as dlext

# -------------------------------
# 1. Topology with 2 atoms
# -------------------------------
top = Topology()
chain = top.addChain()
res = top.addResidue('A', chain)
top.addAtom('a1', Element.getByAtomicNumber(39), res)
top.addAtom('a2', Element.getByAtomicNumber(39), res)

# -------------------------------
# 2. System
# -------------------------------
system = openmm.System()
system.addParticle(39.9*unit.amu)
system.addParticle(39.9*unit.amu)

# harmonic bond: r0 = 0.1 nm, k = 1000 kJ/(mol*nm^2)
bond = openmm.HarmonicBondForce()
bond.addBond(0, 1, 0.1*unit.nanometer,
             1000.0*unit.kilojoule_per_mole/unit.nanometer**2)
system.addForce(bond)

# attach DLExt force
dlforce = dlext.Force()
system.addForce(dlforce)

# -------------------------------
# 3. Integrator + CUDA platform
# -------------------------------
integrator = openmm.VerletIntegrator(0.001)
platform = openmm.Platform.getPlatformByName("CUDA")
sim = Simulation(top, system, integrator, platform)

# -------------------------------
# 4. Initial positions
# -------------------------------
init_positions = np.array([[0.0, 0.0, 0.0],
                           [0.12, 0.0, 0.0]]) * unit.nanometer
sim.context.setPositions(init_positions)

# -------------------------------
# 5. DLExt GPU view
# -------------------------------
ctx_view = dlforce.view(sim.context)
n_particles = 2  # only first 2 rows are valid

# CuPy arrays for positions & forces (zero-copy)
positions_gpu = cp.from_dlpack(dlext.positions(ctx_view))#[:n_particles, :3]
print("Initial GPU positions:")
print(positions_gpu)

def rescale(x):
    return x / 1.717987e+11 * 40

# -------------------------------
# 6. Compute initial forces
# -------------------------------
sim.context.getState(getForces=True)
forces_cpu = sim.context.getState(getForces=True).getForces(asNumpy=True)
forces_gpu = cp.from_dlpack(dlext.forces(ctx_view))
print("Initial CPU forces:")
print(forces_cpu)
print("Initial GPU forces:")
print(rescale(forces_gpu))

# -------------------------------
# 7. Update positions on GPU
# -------------------------------
positions_gpu[1,0] += 0.01  # move particle 1 along x
print("Moved GPU positions:")
print(positions_gpu)
#print("Moved CPU positions:")
#print(sim.context.getState(getPositions=True).getPositions(asNumpy=True))


# recompute (?) forces
forces_cpu = sim.context.getState(getEnergy=True)#.getForces(asNumpy=True)

#capsule = dlext.forces(ctx_view)
#forces_gpu = cp.from_dlpack(capsule)

print("Updated CPU forces:")
print(forces_cpu)
print("Updated GPU forces:")
print(rescale(forces_gpu))
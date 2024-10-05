import numpy as np
from openmm import *
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
from openmm.app import *
from openmm.unit import *
"""Calculate the center of mass of a system."""

def center_of_mass(sim: Simulation):
    """Calculate the center of mass of a system.

    Parameters
    ----------
    t : simtk.openmm.app.Topology
        The Topology object of the system.

    Returns
    -------
    center_of_mass : simtk.unit.Quantity
        The center of mass of the system.
    """
    positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)/nanometer
    masses = np.array([atom.element.mass/dalton for atom in sim.topology.atoms()])
    center_of_mass = np.sum(positions * masses[:, np.newaxis], axis=0) / masses.sum()
    print(type(center_of_mass))
    return (center_of_mass)


# Determine the farthest away atoms from the center of mass for ligand and protein
def furthest_away_atoms(sim: Simulation):
    """Determine the farthest away atoms from the center of mass for ligand and protein.

    Parameters
    ----------
    t : simtk.openmm.app.Topology
        The Topology object of the system.

    Returns
    -------
    furthest_away_ligand : simtk.unit.Quantity
        The farthest away atom from the center of mass for the ligand
    furthest_away_protein : simtk.unit.Quantity
    """
    positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)/nanometer
    center_of_mass_result = center_of_mass(sim)
    ligand_atoms = [atom for atom in sim.topology.atoms() if atom.residue.name == 'UNK']
    protein_atoms = [atom for atom in sim.topology.atoms() if atom.residue.name != 'UNK']
    ligand_positions = np.array([positions[atom.index] for atom in ligand_atoms])
    protein_positions = np.array([positions[atom.index] for atom in protein_atoms])
    ligand_distances = np.linalg.norm(ligand_positions - center_of_mass_result, axis=1)
    protein_distances = np.linalg.norm(protein_positions - center_of_mass_result, axis=1)
    furthest_away_ligand = ligand_atoms[np.argmax(ligand_distances)].index
    furthest_away_protein = protein_atoms[np.argmax(protein_distances)].index
    ligand_vector = ligand_positions[np.argmax(ligand_distances)] - center_of_mass_result
    protein_vector = protein_positions[np.argmax(protein_distances)] - center_of_mass_result
    return [furthest_away_ligand, ligand_vector, furthest_away_protein, protein_vector]


def get_ligand_idx_and_distance(sim: Simulation):
    positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)/nanometer
    center_of_mass_result = center_of_mass(sim)
    ligand_atoms = [atom.index for atom in sim.topology.atoms() if atom.residue.name == 'UNK']
    ligand_positions = np.array([positions[atom] for atom in ligand_atoms])
    ligand_distances = np.linalg.norm(ligand_positions - center_of_mass_result, axis=1)
    forces = sim.context.getState(getForces=True).getForces(asNumpy=True)*nanometer*mole/kilojoule
    avg_ligand_force = np.mean(np.abs(forces[ligand_atoms]), axis=0)
    avg_force = np.mean(np.abs(forces), axis=0)
    ligand_vectors_normed = np.divide((ligand_positions - center_of_mass_result), ligand_distances[:, np.newaxis])
    return [ligand_vectors_normed, ligand_atoms, avg_force, avg_ligand_force]
using ISOKANN

# pdb file of samples to start with - in this case a reactive path
pdb = "data/systems/AmyloidBetaNoWater.pdb"
sim = OpenMMSimulation(; pdb, forcefields=OpenMM.FORCE_AMBER_IMPLICIT, steps=5000 * 5)

traj = load_trajectory(pdb)
featurizer = OpenMM.featurizer(sim, 0)
data = SimulationData(sim, traj, 1; featurizer)
iso = Iso(data)
runadaptive!(iso, iter=100, kde=10, generations=500)
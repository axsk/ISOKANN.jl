using ISOKANN

# Creating a simulation
sim = OpenMMSimulation()
traj = OpenMM.trajectory(sim)

# SimulationData
# option A: load trajectory
traj = load_trajectory("trajectory.pdb")
data = data_from_trajectory(traj)

# option B, generate data
data = SimulationData(sim, 100, 3)

# inspect coords
coords(data)

# create Iso objetct
iso = Iso(data)

run!(iso, 100)

plot_training(iso)

scatter_ramachandran(iso) # only for alaninedipept

save_reactive_path(iso)
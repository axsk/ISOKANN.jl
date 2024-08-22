using ISOKANN

pdb = "data/openmm8ef5-processed.pdb"
radius = 1.0
nk = 2
samples = 100
trainsteps = 100

###

sim = OpenMMSimulation(; pdb, minimize=true, gpu=true)

x0 = getcoords(sim)
cainds = atom_indices(pdb, "name==CA")
featinds = restricted_localpdistinds(x0, radius, cainds)

traj = laggedtrajectory(sim, samples)
data = SimulationData(sim, traj, nk, featurizer=coords -> pdists(coords, featinds))

iso = Iso2(data, opt=NesterovRegularized(), gpu=true)

run!(iso, trainsteps)

iso.data = addcoords(iso.data, laggedtrajectory(sim, samples, x0=iso.data.coords[1][:, end]))

run!(iso, trainsteps)
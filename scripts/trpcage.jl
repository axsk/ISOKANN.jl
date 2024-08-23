sim = OpenMMSimulation(pdb="data/2jof-processed.pdb", forcefields=ISOKANN.OpenMM.FORCE_AMBER_IMPLICIT, features=ISOKANN.OpenMM.noHatoms("data/2jof-processed.pdb"), steps=100, temp=360)

data = SimulationData(sim, nx=100, nk=8)

iso = Iso(data, model=pairnet(data, activation=Flux.swish, layers=4), gpu=true, opt=AdamRegularized(1e-4, 1e-5), minibatch=100, loggers=[ISOKANN.autoplot(10)])

runadaptive!(iso, generations=1000, nx=3, cutoff=2000, iter=500, keepedges=true)

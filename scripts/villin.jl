using ISOKANN
using ISOKANN.OpenMM

pdb = "data/villin nowater.pdb"
steps = 2500
features = 0 # 0 => backbone only
iter = 1000
generations = 10
cutoff = 1000
nx = 10
nk = 2
minibatch = 100
opt = ISOKANN.NesterovRegularized(1e-3, 1e-4)
layers = 4
sigma = 2
forcefields = OpenMM.FORCE_AMBER_IMPLICIT

burstlength = steps * 0.002 / 1000 # in nanoseconds
simtime_per_gen = burstlength * nx * nk # in nanoseconds

@show burstlength, simtime_per_gen, "(in ns)"

sim = OpenMMSimulation(;
  pdb, steps, forcefields, features,
  nthreads=1,
  mmthreads="gpu")

data = SimulationData(sim; nx, nk)

iso = Iso2(data;
  opt, minibatch,
  model=pairnet(length(sim.features); layers),
  gpu = true,
  loggers = [])

run!(iso, iter)

 
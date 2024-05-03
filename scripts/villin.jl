using ISOKANN
using ISOKANN.OpenMM

## Config

pdb = "data/villin nowater.pdb"
steps = 50000  # 100 ps
features = 0 # 0 => backbone only
iter = 1000
generations = 2
cutoff = 3000
nx = 10
nk = 4
keepedges = true
minibatch = 100
opt = ISOKANN.NesterovRegularized(1e-3, 1e-4)
layers = 4
sigma = 2
forcefields = OpenMM.FORCE_AMBER_IMPLICIT

## Initialisation

hash = "$(time())"[4:11]

burstlength = steps * 0.002 / 1000 # in nanoseconds
simtime_per_gen = burstlength * nx * nk # in nanoseconds

@show burstlength, simtime_per_gen, "(in ns)"

sim = OpenMMSimulation(;
  pdb, steps, forcefields, features,
  nthreads=1,
  mmthreads="gpu")

@time "generating initial data" data = SimulationData(sim, nx, nk)

iso = Iso2(data;
  opt, minibatch,
  model=pairnet(length(sim.features); layers),
  gpu = true,
  loggers = [])

run!(iso, iter)

## Running

for i in 1:1_000 # one microsecond
  @time "generation" runadaptive!(iso; generations, nx, iter, cutoff, keepedges)

  time = length(iso.losses) / iter * simtime_per_gen
  time = round(time, digits=2)
  save_reactive_path(iso, out="out/$hash/villin_fold_$(time)ns.pdb"; sigma)
  ISOKANN.Plots.savefig(plot_training(iso), "out/$hash/villin_fold_$(time)ns.png")
  ISOKANN.save("out/villin_fold.jld2", iso)
end
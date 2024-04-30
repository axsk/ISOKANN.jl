using ISOKANN
using ISOKANN.OpenMM

pdb = "data/villin nowater.pdb"
steps = 2500
features = 0.2
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

simtime_per_gen = steps * 0.002 * nx / 1000 * nk # in nanoseconds
println("$simtime_per_gen ns per data generation")


sim = OpenMMSimulation(;
  pdb, steps, forcefields, features,
  nthreads=1,
  mmthreads="gpu")

# 10 ps per datapoint, i.e. 100 ps starting data
@time "initialisation" iso = Iso2(sim;
  nx, nk, opt, minibatch, gpu=true)
  model=pairnet(length(sim.features); layers))

run!(iso, iter)

for i in 1:1_000 # one microsecond
  @time "generation" runadaptive!(iso; generations, nx, iter, cutoff)
  time = length(iso.losses) / iter * simtime_per_gen
  save_reactive_path(iso, out="out/villin_fold_$(time)ns.pdb"; sigma)
  ISOKANN.Plots.savefig(plot_training(iso), "out/villin_fold_$(time)ns.png")
  ISOKANN.save("out/villin_fold.jld2", iso)
end

# Example of applying ISOKANN with adaptive sampling to the VGVAPG Hexapeptide

using ISOKANN


pdb = "/scratch/htc/ldonati/VGVAPG/implicit/input/initial_states/x0_1.pdb"

# define the simulation
sim = ISOKANN.OpenMM.OpenMMSimulation(;
  pdb,
  forcefields=["amber14-all.xml", "implicit/obc2.xml"],  # load the implicit water forcefield as well
  step=0.002, # picoseconds
  steps=500, # steps per simulation
  temp=298, # Kelvin
  friction=1)

# create trainings data from the simulation
@time data = ISOKANN.SimulationData(sim;
  nx=100, # number of data points
  nk=10 # number of koopman burst samplems
)

model = pairnet(data) # create the default `pairnet`` neural network for the data
opt = AdamRegularized(1e-3, 1e-4) # set learning rate and decay parameters

# create the ISOKANN object
iso = Iso(data; model, opt,
  minibatch=100, # size of the minibatch for the SGD,
  gpu=true)

run!(iso, 100) # run 100 ISOKANN iterations

runadaptive!(iso,
  generations=10, # number of resamplings
  nx=10, # data points per resampling
  iter=100 # number of isokann iterations between resamplings
)

# run the simulation until the loss is small enough
for i in 1:30
  runadaptive!(iso, generations=1, nx=10, iter=1000)
  if iso.losses[end] < 1e-3
    break
  end
end

# save the data into a pdb, sorted by the χ value
ISOKANN.exportsorted(iso, "out/sorted.pdb")

# plot the learned reaction coordinate against the distance between the outer carbons
scatter(chis(iso) |> vec |> cpu, ISOKANN.getxs(iso.data)[2416, :] |> vec |> cpu)

# extract the reactive path from the sampling data
save_reactive_path(iso, out="out/reactivepath.pdb", sigma=0.5)


function scatter_rc(iso, a, b)
  x = chis(iso) |> vec
  natoms = div(ISOKANN.dim(iso.data.sim), 3)
  hi = ISOKANN.halfinds(natoms)
  i = findfirst(i -> i == CartesianIndex(a, b), hi)
  scatter(ISOKANN.getxs(iso.data)[i, :] |> vec |> cpu, chis(iso) |> vec |> cpu, xlabel="distance $((a,b))", ylabel="χ")
end

function cor_rc(iso)
  c = cor(ISOKANN.getxs(iso.data) |> cpu, ISOKANN.chis(iso) |> cpu, dims=2) |> vec
  inds = sortperm(c, rev=true)
  natoms = div(ISOKANN.dim(iso.data.sim), 3)
  hi = ISOKANN.halfinds(natoms)
  return zip(hi[inds], c[inds]) |> collect
end

function rc_plots(iso, n=1)
  map(cor_rc(iso)[1:n]) do (a, b)
    @show a, b
    p = scatter_rc(iso, Tuple(a)...)
    plot!(title="$(Tuple(a)), corr = $b") |> display
  end
end
rc_plots(iso)
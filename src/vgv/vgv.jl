using ISOKANN.IsoMu: readchemfile
using ISOKANN: flatpairdists, ISOKANN, run!, AdamRegularized
using ProgressMeter
using JLD2
using Flux: Flux, cpu, gpu, Dense, Chain
using StatsBase: mean, std
using Plots
import Distances: pairwise, Euclidean
using LinearAlgebra: UpperTriangular
using Optimisers: Optimisers

## DATA INGESTION

# the maximal distance is attained between 14-CA2 and 72-O

PDB_TEMPLATE = "data/luca/VGVAPG/implicit/input/initial_states/x0_1.pdb"

function defaultdata()
  global DATA
  @isdefined(DATA) || (DATA = readsim())
  return DATA
end

# this loads the vgv dataset into the global namespace
function read_vgvapg(; kwargs...)
  readnfgail(dir="data/luca/VGVAPG/implicit"; save="data/luca/vgvapg.jld2", natoms=73, kwargs...)
end

function read_traj()
  ISOKANN.IsoMu.readchemfile("data/luca/VGVAPG/implicit/output/trajectory.dcd")
end

function readsim(;
  dir="data/luca/VGVAPG/implicit",
  nx=500,
  nk=100,
  nt=10)

  xs = stack((readchemfile("$dir/input/initial_states/x0_$(i-1).pdb", 1)[:, :, 1] for i in 1:nx))

  dim, atoms, nx = size(xs)
  ys = similar(xs, dim, atoms, nk, nx, nt)

  for i in 1:nx, k in 1:nk
    ys[:, :, k, i, :] .= readchemfile("$dir/output/final_states/xf_$(i-1)_r$(k-1).dcd")
  end

  xs, ys
end

function pairwisedata(; data=defaultdata(), nk=100, lag=10)
  xs, ys = data
  inds = halfinds(size(xs, 2))
  dx = stack(eachslice(xs, dims=3)) do co
    Distances.pairwise(Euclidean(), co |> collect, dims=2)
  end[inds, :] ./ 10

  dy = stack(eachslice(ys[:, :, 1:nk, :, lag], dims=(3, 4))) do co
    pairwise(Euclidean(), co, dims=2)
  end[inds, :, :] ./ 10

  return dx, dy
end

using LinearAlgebra
function halfinds(n)
  a = UpperTriangular(ones(n, n))
  a[diagind(a)] .= 0
  findall(a .> 0)
end

### ISOKANN MODELS

# a copy of lucas python model
function lucaisokann(; data=pairwisedata(), kwargs...)
  model = lucanet2(size(data[1], 1))
  opt = ISOKANN.AdamRegularized(5e-4, 1e-5)

  iso = IsoRunFixedData(; data, model, opt,
    minibatch=100,
    nd=100, # niters
    nl=10, # epochs
    kwargs...
  ) |> gpu

  iso.loggers = [ISOKANN.autoplot(1), reactioncoordlogger(iso)]
  return iso
end

using Flux: Dense

lucanet1(dim; activation=Flux.sigmoid) = Chain(Dense(dim => 2048, activation),
  Dense(2048 => 1024, activation),
  Dense(1024 => 512, activation),
  Dense(512 => 1, identity))

lucanet2(dim; activation=Flux.sigmoid) = Chain(Dense(dim => 204, activation),
  Dense(204 => 102, activation),
  Dense(102 => 51, activation),
  Dense(51 => 1, identity))

alexisokann(; data=pairwisedata(), kw...) = lucaisokann(;
  data,
  model=ISOKANN.pairnet(data),
  opt=ISOKANN.AdamRegularized(1e-3, 1e-3),
  nd=1000,
  nl=1,
  kw...)


function alex2()
  data = pairwisedata()
  ISOKANN.Iso2(data, opt=AdamRegularized(1e-4, 1e-3))
end

function examplerun()

  iso = alexisokann() |> Flux.gpu
  run!(iso)

  ISOKANN.plot_training(iso) |> display

  co = DATA[1]

  export_sorted(iso, coords)
  flatcoords = reshape(co, :, size(co, 3))
  ISOKANN.save_reactive_path(Iso2(iso), flatcoords,
    sigma=1,
    out="out/vgv/reactionpath.pdb",
    source=PDB_TEMPLATE)

  return iso
end





###

IsoRunFixedData(; data, kwargs...) = ISOKANN.IsoRun(;
  data=data,
  model=ISOKANN.pairnet(data),
  nd=1,
  minibatch=0,
  nx=0, # no chi subsampling,
  nres=0, # no resampling,
  np=1, # power iterations,
  nl=1, # weight updates,
  sim=nothing, kwargs...)


function scatter_reactioncoord(iso, xs=defaultdata()[1])
  rc = sqrt.(sum(abs2, xs[:, 1, :] .- xs[:, 71, :], dims=1)) |> vec
  chis = iso.model(iso.data[1]) |> vec |> Flux.cpu
  scatter(rc, chis, xlabel="outer atom distance", ylabel="\\chi")
end

function reactioncoordlogger(iso, xs=defaultdata()[1])
  (; plot=() -> scatter_reactioncoord(iso, xs))
end

function plot_longtraj(iso)
  xs = read_traj()
  inds = halfinds(size(xs, 2))
  dx = stack(eachslice(xs, dims=3)) do co
    pairwise(Euclidean(), co, dims=2)
  end[inds, :] ./ 10
  vals = iso.model(dx |> gpu) |> cpu |> vec
  plot(vals, xlabel="frame #", ylabel="chi", title="long traj")
end



using ISOKANN.IsoMu: aligntrajectory
function export_sorted(iso, xs, path="out/vgv/chisorted.pdb", source="data/luca/VGVAPG/implicit/input/initial_states/x0_1.pdb")
  i = sortperm(iso.model(iso.data[1]) |> cpu |> vec)
  xs = reshape(xs[:, :, i], :, length(i))
  #xs = ISOKANN.standardform(xs, (21, 28, 44))
  xs = vec.(aligntrajectory(Flux.MLUtils.unbatch(xs))) |> stack
  println("saved sorted trajectory to $path")

  ISOKANN.IsoMu.writechemfile(path, xs; source)
end


#=
function pairwisedata_my((xs, ys), nk, lag)
  nx = size(xs, 3)
  xs = reshape(xs, :, nx)
  ys = reshape(ys[:, :, 1:nk, :, lag], :, nk, nx)

  # note that this is the squared distances

  xs = ISOKANN.flatpairdists(xs) .|> sqrt
  ys = ISOKANN.flatpairdists(ys) .|> sqrt
  inds = halfinds(xs)
  return (xs[inds, :], ys[inds, :, :])

  #return normalizedata((xs[inds, :], ys[inds, :, :]))
end

function test(vgv=read_vgvapg())
  d1 = pairwisedata_my(vgv, 100, 10)[1]
  d2 = pairwisedata(vgv, 100, 10)[1]
  isapprox(d1, d2)
end


## A CHECK HOW GOOD MODELS CAN LEARN A GIVEN DISTANCE
function benchmarkmodels(; reps=3, epochs=10, iso=lucaisokann(), activation=Flux.sigmoid)
  dim = size(iso.data[1], 1)
  target = let xs = vgv[1]
    sqrt.(sum(abs2, xs[:, 1, :] .- xs[:, 71, :], dims=1))
  end |> ISOKANN.shiftscale
  plot(target')
  for _ in 1:reps
    for (model, color) in [(lucanet(dim; activation), :blue), (ISOKANN.pairnet(dim, layers=4; activation), :red)]
      iso.model = model
      Optimisers.setup(iso)
      @time for i in 1:epochs
        Flux.train!((m, x, y) -> Flux.mse(m(x), y), iso.model, Flux.DataLoader((iso.data[1], target), batchsize=50), iso.opt)
      end
      @show Flux.mse(iso.model(iso.data[1]), target)
      plot!(iso.model(iso.data[1])'; color) |> display
    end
  end

  plot!() |> display

end

=#
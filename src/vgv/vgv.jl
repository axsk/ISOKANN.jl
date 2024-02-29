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
using Flux: Dense


## DATA HANDLING

abstract type VGVData end

struct VGVData2 <: VGVData
  dir 
  data
  coords
end

function VGVData(dir= "data/luca/VGVAPG/implicit"; lag=1, nx=500, nk=100, nt=10)
  xs, ys = alldata(dir, nx, nk, nt)
  coords = reshape(xs, :, size(xs,3))
  data = pairwisedata(xs, ys[:, :, :, :, lag])
  VGVData2(dir, data, coords)
end

pdbfile(v::VGVData) = joinpath(v.dir, "input/initial_states/x0_1.pdb")
getcoords(v::VGVData) = v.coords :: AbstractMatrix
function reactioncoord(v::VGVData)
  i = findfirst(CartesianIndex(1,71).==halfinds(73))
  v.data[1][i,:]
end

# TODO: we need a proper caching idea, probably Memoize or Albert
#=function alldata_cached(args...)
  global VGVDATA
  @isdefined(VGVDATA) || (VGVDATA = alldata(args...))
  return VGVDATA
end
=#

function alldata(dir, nx, nk, nt)
  xs = stack((readchemfile("$dir/input/initial_states/x0_$(i-1).pdb", 1)[:, :, 1] for i in 1:nx))

  dim, atoms, nx = size(xs)
  ys = similar(xs, dim, atoms, nk, nx, nt)

  for i in 1:nx, k in 1:nk
    ys[:, :, k, i, :] .= readchemfile("$dir/output/final_states/xf_$(i-1)_r$(k-1).dcd")
  end

  xs, ys
end

# TODO: implment batched dists instead
# cf pairwisedists
function pairwisedata(xs, ys)
  dx = stack(eachslice(xs, dims=3)) do co
    pairwise(Euclidean(), co, dims=2)
  end

  dy = stack(eachslice(ys, dims=(3, 4))) do co
    pairwise(Euclidean(), co, dims=2)
  end

  inds = halfinds(size(xs, 2))

  return dx[inds, :] ./ 10, dy[inds, :, :] ./ 10
end

function halfinds(n)
  a = UpperTriangular(ones(n, n))
  a[diagind(a)] .= 0
  findall(a .> 0)
end


### ISOKANN MODELS

# a copy of lucas python model
function vgv_luca(;v=VGVData(), kwargs...)
  model = lucanet2(size(v.data, 1))
  opt = AdamRegularized(5e-4, 1e-5)

  iso = IsoRunFixedData(; v.data, model, opt,
    minibatch=100,
    nd=100, # niters
    nl=10, # epochs
    kwargs...
  ) |> gpu

  iso.loggers = [ISOKANN.autoplot(1), 
    (; plot=() -> scatter_reactioncoord(iso, v))]
  return iso
end

# TODO: should make these defaults for sim==nothing
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



lucanet1(dim; activation=Flux.sigmoid) = Chain(Dense(dim => 2048, activation),
  Dense(2048 => 1024, activation),
  Dense(1024 => 512, activation),
  Dense(512 => 1, identity))

lucanet2(dim; activation=Flux.sigmoid) = Chain(Dense(dim => 204, activation),
  Dense(204 => 102, activation),
  Dense(102 => 51, activation),
  Dense(51 => 1, identity))

vgv_alex(; v=VGVDATA(), kw...) = vgv_luca(;
  v,
  model=ISOKANN.pairnet(v.data),
  opt=ISOKANN.AdamRegularized(0e-3, 1e-3),
  nd=1000,
  nl=1,
  kw...)


### OUTPUT

function scatter_reactioncoord(iso, v::VGVData)
  chi = chis(iso) |> vec |> cpu
  rc = reactioncoord(v) |> vec
  scatter(rc, chi, xlabel="outer atom distance", ylabel="\\chi")
end

function plot_longtraj(iso, v::VGVData)
  xs = ISOKANN.IsoMu.readchemfile("$(v.dir)/implicit/output/trajectory.dcd")
  inds = halfinds(size(xs, 2))
  dx = stack(eachslice(xs, dims=3)) do co
    pairwise(Euclidean(), co, dims=2)
  end[inds, :] ./ 10
  vals = iso.model(dx |> gpu) |> cpu |> vec
  plot(vals, xlabel="frame #", ylabel="chi", title="long traj")
end

using ISOKANN.IsoMu: aligntrajectory
function save_sorted_path(iso, v::VGVData, path="out/vgv/chisorted.pdb")
  source = pdbfile(v)
  i = sortperm(chis(iso) |> cpu |> vec)
  xs = aligntrajectory(getcoords(v)[:,  i])
  println("saved sorted trajectory to $path")
  IsoMu.writechemfile(path, xs; source)
end


### EXAMPLE

function vgv_examplerun(v=VGVData())
  iso = vgv_alex(;v)
  run!(iso)
  plot_training(iso) |> display

  save_sorted_path(iso, v)
  save_reactive_path(Iso2(iso), v.coords,
    sigma=1,
    out="out/vgv/reactionpath.pdb",
    source=pdbfile(v))

  return iso
end
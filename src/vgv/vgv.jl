#using ISOKANN: readchemfile, writechemfile
#using ISOKANN: flatpairdists, ISOKANN, run!, AdamRegularized
using ProgressMeter
using JLD2
using Flux: Flux, cpu, gpu, Dense, Chain
using StatsBase: mean, std
using Plots

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

VGV_DATA_DIR = "/scratch/htc/ldonati/VGVAPG/implicit"
VGV_DATA_5000 = "/scratch/htc/ldonati/VGVAPG/implicit5000"

VGV5000(; kw...) = VGVData(VGV_DATA_5000, nx=5000, nk=10, t=1, natoms=73; kw...)

function VGVData(dir=VGV_DATA_DIR; nx=500, nk=100, t=1, natoms=73)
  xs, ys = vgv_readdata(dir, nx, nk, t, natoms)
  coords = reshape(xs, :, size(xs,3))
  dx = flatpairdists(flattenfirst(xs)) ./ 10
  dy = flatpairdists(flattenfirst(ys)) ./ 10
  data = (dx, dy)
  VGVData2(dir, data, coords)
end

MLUtils.getobs(v::VGVData) = v.data

pdbfile(v::VGVData) = joinpath(v.dir, "input/initial_states/x0_1.pdb")
getcoords(v::VGVData) = v.coords :: AbstractMatrix

function reactioncoord(v::VGVData)
  i = findfirst(CartesianIndex(1,71).==halfinds(73))
  v.data[1][i,:]
end

# TODO: we need a proper caching idea, probably Memoize or Albert

function vgv_readdata(dir, nx, nk, t, natoms)
  xs = zeros(3, natoms, nx)
  ys = zeros(3, natoms, nk, nx)
  @showprogress for i in 1:nx
    xs[:, :, i] .= readchemfile("$dir/input/initial_states/x0_$(i-1).pdb", 1)
    for k in 1:nk
      ys[:, :, k, i] .= readchemfile("$dir/output/final_states/xf_$(i-1)_r$(k-1).dcd", t)
    end
  end
  xs, ys
end

### ISOKANN MODELS

# a copy of lucas python model
function vgv_luca(v::VGVData=VGVData(); kwargs...)
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

vgv_alex(v::VGVData=VGVData(); kw...) = vgv_luca(v;
  model=ISOKANN.pairnet(v.data),
  opt=ISOKANN.AdamRegularized(1e-3, 1e-3),
  nd=100,
  nl=1,
  kw...)

lucanet1(dim; activation=Flux.sigmoid) = Chain(Dense(dim => 2048, activation),
  Dense(2048 => 1024, activation),
  Dense(1024 => 512, activation),
  Dense(512 => 1, identity))

lucanet2(dim; activation=Flux.sigmoid) = Chain(Dense(dim => 204, activation),
  Dense(204 => 102, activation),
  Dense(102 => 51, activation),
  Dense(51 => 1, identity))

### OUTPUT

function scatter_reactioncoord(iso, v::VGVData)
  chi = chis(iso) |> vec |> cpu
  rc = reactioncoord(v) |> vec
  scatter(rc, chi, xlabel="outer atom distance", ylabel="\\chi")
end

function plot_longtraj(iso, v::VGVData)
  xs = readchemfile("$(v.dir)/output/trajectory.dcd")
  dx = flatpairdists(flattenfirst(xs)) ./ 10
  vals = iso.model(dx |> gpu) |> cpu |> vec
  plot(vals, xlabel="frame #", ylabel="chi", title="long traj")
end


function save_sorted_path(iso, v::VGVData, path="out/vgv/chisorted.pdb")
  source = pdbfile(v)
  i = sortperm(chis(iso) |> cpu |> vec)
  xs = aligntrajectory(getcoords(v)[:,  i])
  println("saved sorted trajectory to $path")
  writechemfile(path, xs; source)
end


### EXAMPLE

function vgv_examplerun(v=VGV5000(nk=1), outdir="out/vgv_examplerun")
  mkpath(outdir)
  iso = vgv_alex(v)
  run!(iso)

  plot_training(iso) |> display
  savefig("$outdir/training.png")

  plot_longtraj(iso, v) |> display
  savefig("$outdir/longtraj.png")

  save_sorted_path(iso, v, "$outdir/chisorted.pdb")
  save_reactive_path(Iso2(iso), getcoords(v),
    sigma=1,
    out="$outdir/reactionpath.pdb",
    source=pdbfile(v))

  open("$outdir/parameters.txt", "w") do io
    show(io, MIME"text/plain"(), iso)
  end

  println("saved vgv output to $outdir")

  return iso
end
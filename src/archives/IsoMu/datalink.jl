"""
    DataLink

A reference to the directory with the `traj.dcd` and `struct.pdb` file.

# Fields
- `dir::String`: stores the path
- `stride::Int=1`: steps
- `startpos::Int`=1: index of first frame to read
- `radius::Float64=12`: max radius of the local pairwise distances
- `reverse::Bool=true`: whether to also use the reverse trajectory as data
"""
Base.@kwdef mutable struct DataLink
  dir::String
  stride::Int = 1
  startpos::Int = 1
  radius::Float64 = 1.2
  reverse::Bool = true
end

DataLink(dir::String; kwargs...) = DataLink(; dir, kwargs...)

trajfile(d::DataLink) = joinpath(d.dir, "traj.dcd")
pdbfile(d::DataLink) = joinpath(d.dir, "struct.pdb")
outdir(d::DataLink) = joinpath("out", splitpath(d.dir)[2:end]...)
trajectory(d::DataLink) = ISOKANN.readchemfile(trajfile(d))[:, :, d.startpos:d.stride:end]
coords(d::DataLink) = d.reverse ? trajectory(d)[:, :, 2:end-1] : trajectory(d)[:, :, 2:end]  # returns the coordinates corresponding to the isodata

coords(ds::Vector{DataLink}) = mapreduce(coords, (a, b) -> cat(a, b, dims=3), ds)
outdir(ds::Vector{DataLink}) = outdir(first(ds))
pdbfile(ds::Vector{DataLink}) = pdbfile(first(ds))

""" load the trajectory and pdb file and filter to CA atoms """
function readdata(d::DataLink)
  xs = trajectory(d)

  cainds = c_alpha_inds(d)
  cas = xs[:, cainds, :]

  distinds = localpdistinds(cas, d.radius)
  dists = pdists(cas, distinds)

  mc = mean_and_std(dists, 2)
  ndists = (dists .- mc[1]) ./ mc[2]

  isodata = ISOKANN.data_from_trajectory(ndists; d.reverse)
  return (; xs, cas, cainds, dists, distinds, isodata, mc)
end

isodata(d::DataLink) = readdata(d).isodata

ISOKANN.SimulationData(d::DataLink) = ISOKANN.SimulationData(d, isodata(d), coords(d), :implicit)

function isodata(ds::Vector{DataLink})
  radius, cainds, reverse = let d1=ds[1]
    d1.radius, c_alpha_inds(d1), d1.reverse
  end

  lens, xs = mapreduce(((l1,x1),(l2,x2))->(vcat(l1, l2), cat(x1, x2, dims=3)), ds) do d
    x = trajectory(d)
    l = size(x, 3)
    ([l], x)
  end

  inds2 = cumsum(lens)
  inds1 = [1; inds2[1:end-1] .+ 1]

  cas = xs[:, cainds, :]

  distinds = localpdistinds(cas, radius)
  dists = pdists(cas, distinds)

  mc = mean_and_std(dists, 2)
  ndists = (dists .- mc[1]) ./ mc[2]

  data = mapreduce(ISOKANN.joindata, 1:length(inds1)) do i
    rng = inds1[i]:inds2[i]
    ISOKANN.data_from_trajectory(@view(ndists[:, rng]); reverse)
  end

  return data
end

## Indices of the Cα atoms from the .pdb

c_alpha_inds(d::DataLink) = c_alpha_inds(pdbfile(d))
function c_alpha_inds(pdbfile="data/struct.pdb")
  pdb = BioStructures.read(pdbfile, BioStructures.PDBFormat)
  #df = BioStructures.DataFrame(BioStructures.collectatoms(pdb))
  df = DataFrame(BioStructures.collectatoms(pdb))
  ids = filter(x -> x["atomname"] == "CA", df)[!, :serial]
  return ids
end

## pairwise distances

# note there is some overlap here with what we have in pairdists.jl
# however, here we use only some features

function localpdistinds(traj::AbstractArray{<:Any,3}, radius)
  elmin(x, y) = min.(x, y)
  d = mapreduce(elmin, eachslice(traj, dims=3)) do coords
    UpperTriangular(pairwise(Euclidean(), coords, dims=2))
  end
  inds = findall(0 .< d .<= radius)
  return inds
end

function pdists(traj::Array{<:Any,3}, inds)
  nframes = size(traj, 3)
  dists = zeros(eltype(traj), length(inds), nframes)
  for j in 1:nframes
    for (i, ind) in enumerate(inds)
      dists[i, j] = @views norm(traj[:, ind[1], j] - traj[:, ind[2], j])
    end
  end
  return dists
end

function localpdists(traj; radius=12)
  inds = localpdistinds(traj, radius)
  dists = pdists(traj, inds)
  dists, inds
end

## experimental code for streaming data
# most methods work fine, training doesnt due to some typeerror though
# while a good first operation, we probably should think about using mapobs more

using Chemfiles
using StatsBase

### Lazy trajectory reading only on access

mutable struct LazyTrajectory
    path::String
    traj::Chemfiles.Trajectory
    size::Tuple{Int,Int,Int}
end

dim(t::LazyTrajectory) = t.size[1] * t.size[2]

function LazyTrajectory(path::String)
    traj = Chemfiles.Trajectory(path, 'r')
    frame = read(traj)
    s = (size(Chemfiles.positions(frame))..., Int(length(traj)))
    return LazyTrajectory(path, traj, s)
end

Base.length(ltl::LazyTrajectory) = ltl.size[3]
Base.size(ltl::LazyTrajectory) = ltl.size
Base.getindex(ltl::LazyTrajectory, idx) = readchemfile(ltl.traj, idx)


### Pairwise Atom Distances 

mutable struct PairwiseAtomDistanceData{D,I,M,S}
    coords::D
    cainds::Vector{Int}
    distinds::I
    mean::M
    std::S
end

dim(x::PairwiseAtomDistanceData) = length(x.distinds)

function PairwiseAtomDistanceData(coords, cainds, estinds, radius=nothing)
    est = coords[estinds]::AbstractArray{<:Any,3}
    dists, distinds = localpdists(est[:, cainds, :], radius=radius)
    mean, std = mean_and_std(dists, 2)

    return PairwiseAtomDistanceData(coords, cainds, distinds, mean, std)
end

Base.size(x::PairwiseAtomDistanceData) = size(x.coords)
Base.length(x::PairwiseAtomDistanceData) = length(x.coords)
function Base.getindex(x::PairwiseAtomDistanceData, idx)
    (; coords, cainds, distinds, mean, std) = x
    d = cadists(coords[idx], cainds, distinds)
    return (d .- mean) ./ std
end

function cadists(xs, cainds, distinds)
    cas = xs[:, cainds, :]
    dists = zeros(eltype(cas), length(distinds), size(cas, 3))
    dim = size(cas, 1)
    for j in axes(cas, 3)
        for (i, ind) in enumerate(distinds)
            for k in 1:dim
                dists[i, j] += (cas[k, ind[1], j] - cas[k, ind[2], j])^2
            end
        end
    end
    dists .= sqrt.(dists)
    return dists
end


### Koopman Trajectory Data

mutable struct KoopmanTrajectoryData{T,M,S}
    traj::T
    model::M
    steps::S
end

dim(x::KoopmanTrajectoryData) = dim(x.traj)
Base.length(d::KoopmanTrajectoryData) = length(d.traj) - d.steps

Base.getindex(d::KoopmanTrajectoryData, i::Int) = d[i:i]

function Base.getindex(d::KoopmanTrajectoryData, idx::UnitRange)
    (; traj, model, steps) = d
    xs = traj[idx.start:idx.stop+steps]
    cs = model(xs)
    ks = [StatsBase.mean(@view cs[i+1:i+steps]) for i in 1:length(idx)]
    xs = xs[:, 1:length(idx)]
    xs, ks
end

function Base.getindex(d::KoopmanTrajectoryData, idx::Vector)
    (; traj, model, steps) = d
    idy = idx' .+ (1:steps) |> vec
    ys = traj[idy]
    cs = model(ys)
    cs = reshape(cs, steps, :)
    ks = StatsBase.mean(cs, dims=1) |> vec
    xs = traj[idx]
    return xs, ks
end

### chi subsampling

## extra experimental using cached k values

mutable struct ChiSubSampler
    xy::KoopmanTrajectoryData
    k::Vector{Float64} # cache of seen k values
end

ChiSubSampler(kd::KoopmanTrajectoryData) = ChiSubSampler(kd, rand(length(kd)))


Base.length(D::ChiSubSampler) = length(D.xy)
dim(d::ChiSubSampler) = dim(d.xy)

function Base.getindex(D::ChiSubSampler, idx)
    k = D.k .- minimum(D.k)
    k ./= maximum(k)
    #k = D.k
    ids = ISOKANN.subsample_uniformgrid(k, length(idx))
    xs, ys = D.xy[ids]
    @show ys
    ys = ISOKANN.shiftscale(ys)
    @show ys
    D.k[ids] = ys
    return (xs, ys)
end

###

using MLUtils

function test_KoopmanTrajectoryData()
    o = ObsView(rand(30, 100))
    model(x) = sum(x, dims=1)
    k = KoopmanTrajectoryData(o, model, 2)
    @assert k[1] == k[1:1] == k[[1]]
end

using MLUtils

function lazyisodata(d::DataLink, model; koopsteps=1)
    traj = LazyTrajectory(trajfile(d))
    cainds = c_alpha_inds(pdbfile(d))
    pairdists = PairwiseAtomDistanceData(traj, cainds, 1:10:length(traj), d.radius)
    koop = KoopmanTrajectoryData(pairdists, model, koopsteps)
    sampled = ChiSubSampler(koop)
    return sampled
end

ISOKANN.datasubsample(model, data::ChiSubSampler, nx) = @show data[1:nx]


function IsoLazyMu(d::DataLink;
    learnrate=0.001,
    regularization=0.0001,
    networkargs=(),
    kwargs...)

    data = lazyisodata(d, nothing)
    model = pairnet(size(data[1], 1); networkargs...)
    data = lazyisodata(d, model)

    iso = ISOKANN.IsoRun(;
        nx=0, # no subsampling
        np=1, # no subreuse
        nl=1, # no powerreuse
        nres=0, # no resampling
        sim=nothing,
        model=model,
        data=data,
        opt=AdamRegularized(learnrate, regularization),
        kwargs...)

    return IMu(data, iso)
end
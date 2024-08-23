"""
Functions to generate and handle data for the ISOKANN algorithm.
"""

""" DataTuple = Tuple{Matrix{T},Array{T,3}} where {T<:Number}

We represent data as a tuple of xs and ys.

xs is a matrix of size (d, n) where d is the dimension of the system and n the number of samples.
ys is a tensor of size (d, k, n) where k is the number of koopman samples.
"""
DataTuple = Tuple{<:AbstractArray{T},<:AbstractArray{T,3}} where {T<:Number}

getxs(x) = getxs(getobs(x))
getys(x) = getys(getobs(x))

getxs(d::Tuple) = d[1]
getys(d::Tuple) = d[2]

""" collapse the first and second dimension of the array `A` into the first dimension """
function flattenfirst(A)
    _, _, r... = size(A)
    reshape(A, :, r...)
end

"""
    bootstrap(sim, nx, ny) :: DataTuple

compute initial data by propagating the molecules initial state
to obtain the xs and propagating them further for the ys """
function bootstrap(sim::IsoSimulation, nx, ny)
    xs = randx0(sim, nx)
    ys = propagate(sim, xs, ny)
    return xs, ys
end

"""
    subsample_inds(model, xs, n) :: Vector{Int}

Returns `n` indices of `xs` such that `model(xs[inds])` is approximately uniformly distributed.
"""
function subsample_inds(model, xs, n; keepedges=true)
    mapreduce(vcat, eachrow(model(xs))) do row
        subsample_uniformgrid(shiftscale(row), n; keepedges)
    end::Vector{Int}
end

"""
    subsample(model, data::Array, n) :: Matrix
    subsample(model, data::Tuple, n) :: Tuple

Subsample `n` points of `data` uniformly in `model`.
If `model` returns multiple values per sample, subsample along each dimension.
"""
subsample(model, xs::AbstractArray{<:Any,2}, n) =
    xs[:, subsample_inds(model, xs, n)]

subsample(model, ys::AbstractArray{<:Any,3}, n) =
    subsample(model, reshape(ys, size(ys, 1), :), n)

subsample(model, data::Tuple, n) = getobs(data, subsample_inds(model, first(data), n))


## TODO: this seems to be deprecated by addata in simulation.jl
"""
    adddata(data::D, model, sim, ny, lastn=1_000_000)::D

Generate new data for ISOKANN by adaptive subsampling using the chi-stratified/-uniform method.

1. Adaptively subsample `ny` points from `data` uniformly along their `model` values.
2. propagate according to the simulation `model`.
3. return the newly obtained data concatenated to the input data

The subsamples are taken only from the `lastn` last datapoints in `data`.

# Examples
```julia-repl
julia> (xs, ys) = adddata((xs,ys), chi, mollysim)
```
"""
function adddata(data, model, sim, ny)
    ny == 0 && return data
    nk = size(last(data), 2)
    xs = subsample(model, last(data), ny)
    ys = propagate(sim, xs, nk)
    return joindata(data, (xs, ys))
end

mergedata(d1::Tuple, d2::Tuple) = lastcat.(d1, d2)

lastcat(x::T, y::T) where {N,T<:AbstractArray{<:Any,N}} = cat(x, y, dims=N)
lastcat(x::T, y) where {T} = lastcat(x, convert(T, y))

@deprecate joindata (x, y) -> lastcat.(x, y)
#=
function datastats(data)
    xs, ys = data
    ext = extrema.(eachrow(xs))
    uni = length(unique(xs[1, :]))
    _, ks, n = size(ys)
    println("\n Dataset has $n entries (approx $uni unique) with $ks koop's. Extrema: $ext")
end
=#

@deprecate stratified_x0(model, ys, n) subsample(model, ys, n)

@deprecate datasubsample(model, data, nx) subsample(model, data, nx)


"""
    data_from_trajectory(xs::Matrix; reverse=false) :: DataTuple

Generate the lag-1 data from the trajectory `xs`.
If `reverse` is true, also take the time-reversed lag-1 data.
"""
function data_from_trajectory(xs::AbstractMatrix; reverse=false)
    if reverse
        @views ys = stack([xs[:, 3:end], xs[:, 1:end-2]])
        ys = permutedims(ys, [1, 3, 2])
        #ys = similar(xs, size(xs, 1, 2, size(xs, 2) - 2))
        #@views ys[:, 1, :] .= xs[:, 3:end]
        #@views ys[:, 2, :] .= xs[:, 1:end-2]
        xs = xs[:, 2:end-1]
    else
        ys = unsqueeze(xs[:, 2:end], dims=2)
        xs = xs[:, 1:end-1]
    end
    return xs, ys
end

"""
    subsample(data, nx)

return a random subsample of `nx` points from `data` """
function subsample(data, nx)
    i = sample(1:numobs(data), nx, replace=false)
    return getobs(data, i)
end

@deprecate data_sliced(data, inds) getobs(data, inds)

@deprecate shuffledata(data) shuffleobs(data)



### Data I/O

"""
    exportdata(data::AbstractArray, model, sys, path="out/data.pdb")

Export data to a PDB file.

This function takes an AbstractArray `data`,
sorts it according to the `model` evaluation,
removes duplicates, transforms it to standard form
and saves it as a PDB file  to `path`."""
function exportdata(data::AbstractArray, model, sys, path="out/data.pdb")
    dd = data
    dd = reshape(dd, size(dd, 1), :)
    ks = model(dd)
    i = sortperm(vec(ks))
    dd = dd[:, i]
    i = uniqueidx(dd[1, :] |> vec)
    dd = dd[:, i]
    dd = standardform(dd)
    savecoords(path, sys, dd)
    dd
end

@deprecate extractdata(data, model, sys, path) exportdata(data, model, sys, path)

uniqueidx(v) = unique(i -> v[i], eachindex(v))

function exportsorted(iso, path="out/sorted.pdb")
    xs = ISOKANN.getxs(iso.data)
    p = iso.model(xs) |> vec |> sortperm
    xs = ISOKANN.getcoords(iso.data)
    println("saving sorted data to $path")
    traj = ISOKANN.aligntrajectory(xs[:, p] |> cpu)
    save_trajectory(path, traj, top=pdb(iso.data))
end
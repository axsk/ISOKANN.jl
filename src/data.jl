"""
Functions to generate and handle data for the ISOKANN algorithm.
"""

""" DataTuple = Tuple{Matrix{T},Array{T,3}} where {T<:Number}

We represent data as a tuple of xs and ys.

xs is a matrix of size (d, n) where d is the dimension of the system and n the number of samples.
ys is a tensor of size (d, n, k) where k is the number of koopman samples.
"""
DataTuple = Tuple{Matrix{T},Array{T,3}} where {T<:Number}

"""
    bootstrap(sim, nx, ny) :: DataTuple

compute initial data by propagating the molecules initial state
to obtain the xs and propagating them further for the ys """
function bootstrap(sim::IsoSimulation, nx, ny)
    x0 = reshape(getcoords(sim), :, 1)
    xs = reshape(propagate(sim, x0, nx), :, nx)
    ys = propagate(sim, xs, ny)
    centercoords(xs), centercoords(ys)  # TODO: centercoords shouldn't be here
end

"""
    subsample_inds(model, xs, n) :: Vector{Int}

Returns `n` indices of `xs` such that `model(xs[inds])` is approximately uniformly distributed.
"""
function subsample_inds(model, xs, n)
    reduce(vcat, eachrow(model(xs))) do row
        subsample_uniformgrid(shiftscale(row), n)
    end::Vector{Int}
end

"""
    subsample(model, data::Array, n) :: Matrix
    subsample(model, data::Tuple, n) :: Tuple

Subsample `n`` points of `data` uniformly in `model`.
If `model` returns multiple values per sample, subsample along each dimension.
"""
subsample(model, xs::AbstractArray{<:Any,2}, n) =
    xs[:, subsample_inds(model, xs, n)]

subsample(model, ys::AbstractArray{<:Any,3}, n) =
    subsample(model, reshape(ys, size(ys, 1), :), n)

function subsample(model, data::Tuple, n)
    xs, ys = data
    ix = subsample_inds(model, xs, n)
    return (xs[:, ix], ys[:, ix, :])
end

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
function adddata(data, model, sim::IsoSimulation, ny; lastn=1_000_000)
    ny == 0 && return data
    _, ys = data
    nk = size(ys, 3)
    firstind = max(size(ys, 2) - lastn + 1, 1)
    x0 = subsample(model, ys[:, firstind:end, :], ny)
    ys = propagate(sim, x0, nk)
    ndata = centercoords(x0), centercoords(ys)  # TODO: this does not really belong here
    data = hcat.(data, ndata)
    return data
end

function datastats(data)
    xs, ys = data
    ext = extrema.(eachrow(xs))
    uni = length(unique(xs[1, :]))
    _, n, ks = size(ys)
    println("\n Dataset has $n entries ($uni unique) with $ks koop's. Extrema: $ext")
end

@deprecate stratified_x0(model, ys, n) subsample(model, ys, n)

@deprecate datasubsample(model, data, nx) subsample(model, data, nx)


"""
    data_from_trajectory(xs::Matrix [, nx]) :: DataTuple

Generate the lag-1 data from the trajectory `xs`.
If `nx` is given, use only the first `nx` transitions of the trajectory.
"""
function data_from_trajectory(xs::Matrix, nx=size(xs, 2) - 1)
    ys = reshape(xs[:, 2:nx+1], :, nx, 1)
    return xs[:, 1:nx], ys
end

"""
    subsample(data, nx)

return a random subsample of `nx` points from `data` """
function subsample(trajdata, nx)
    xs, ys = trajdata
    i = sample(1:size(xs, 2), nx, replace=false)
    return xs[:, i], ys[:, i, :]
end

function data_sliced(data::Tuple, slice)
    xs, ys = data
    (xs[:, slice], ys[:, slice, :])
end

import Random
function shuffledata(data)
    xs, ys = data
    n = size(xs, 2)
    i = Random.randperm(n)
    return xs[:, i], ys[:, i, :]
end

"""  trajectory(sim, nx)
generate a trajectory of length `nx` from the simulation `sim`"""
function trajectory(sim, nx)
    siml = deepcopy(sim)
    logevery = round(Int, sim.T / sim.dt)
    siml.T = sim.T * nx
    xs = solve(siml; logevery=logevery)
    return xs
end


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
    savecoords(sys, dd, path)
    dd
end

@deprecate extractdata(data, model, sys, path) exportdata(data, model, sys, path)

uniqueidx(v) = unique(i -> v[i], eachindex(v))
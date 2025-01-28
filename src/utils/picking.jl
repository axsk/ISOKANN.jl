using CUDA: CuArray
using StatsBase: mean
using NNlib: batched_mul, batched_transpose
using ProgressMeter
using LinearAlgebra: svd
using ISOKANN: ISOKANN

""" 
    picking(X, n; dists = pairwise_one_to_many)

The picking algorithm, i.e. greedy farthest point sampling, for `n` points on the columns of `X`.
A custom distance function (::Vector, ::Matrix)->(::Vector) may be passed through `dists`.

Returns `X[:,qs]`, i.e. the picked samples, their former indices `qs` and their distances `d` to all other points.
"""
function picking(X, n; dists = pairwise_one_to_many)
    @assert size(X, 2) >= n

    d = similar(X, size(X, 2), n) .= 0
    mins = similar(X, size(X, 2)) .= Inf
    origin = similar(X, size(X, 1)) .= 0
    qs = Int[]
    q = argmax(dists(origin, X))  # start with the point furthest from origin

    @showprogress 1 "Picking..." for i in 1:n
        push!(qs, q)
        d[:, i] .= dists(X[:, q], X)
        mins .= min.(mins, d[:, i])
        q = argmax(mins)
    end

    return X[:, qs], qs, d
end

""" 
    picking_aligned(x::AbstractMatrix, m::Integer)

The picking algorithm using pairwise aligned distances, e.g. for molecular coordinates 
"""
function picking_aligned(x::AbstractMatrix, m::Integer)
    n = size(x, 2)
    x = copy(x)
    x = reshape(x, 3, :, n)
    x .-= mean(x, dims=2)
    x = reshape(x, :, n)

    _, qs, d = picking(x, m, dists=ISOKANN.batched_kabsch_rmsd)
    return x[:, qs], qs, d
end


using Distances: SqEuclidean, pairwise
# no CUDA support yet
pairwise_one_to_many(x, xs) = pairwise(SqEuclidean(), xs, reshape(x, :, 1), dims=2) |> vec


#


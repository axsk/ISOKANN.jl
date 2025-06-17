""" 
    module Align

This module provides functionality for structural alignment of molecular structures 
based on the Kabsch transformation / RMSD minimization with CUDA support and atom weighting.

By default the data is is assumed to be in structural format (e.g. a single structure = 3xn matrix) 
with convencience wrappers for flattened representations (in that case vectors).

The core functionality is provided through `align`, `aligned_rmsd` and `pairwise_aligned_rmsd`.
"""
module Align

export align, aligned_rmsd, pairwise_aligned_rmsd, aligntrajectory

using NNlib: batched_mul, batched_transpose
using SparseArrays
using CUDA: CuArray
using LinearAlgebra: svd

MaybeWeights = Union{AbstractVector,Nothing}
weights_and_sum(weights::AbstractVector, n) = weights, sum(weights)
weights_and_sum(weights::Nothing, n) = 1, n

MatOrTensor = Union{AbstractMatrix,AbstractArray{<:Any,3}}

"""
    align(x::MatOrTensor, y::MatOrTensor; weights=nothing)  # structural data (one-one / one-many / many-many)
    align(x::AbstractVector, y::AbstractVector; weights)    # flattened one-one
    align(x::AbstractVector, ys::AbstractMatrix; kwargs...) # flattened one-many

Align all structures in `y` to `x`. Supports simgle structure matching (matrix-matrix), many to one (matrix-tensor) or (tensor-tensor).
"""
function align(x::MatOrTensor, y::AbstractArray{<:Any,3}; weights::MaybeWeights=nothing)
    w, ws = weights_and_sum(weights, size(x, 2))
    m = sum(x .* w', dims=2) / ws
    x = x .- m
    y = y .- sum(y .* w', dims=2) / ws

    h = batched_mul(x .* w', batched_transpose(y))
    s = batched_svd(h)
    r = batched_mul(s.U, batched_transpose(s.V))

    y = batched_mul(r, y) .+ m
    return y
end
align(x::MatOrTensor, y::AbstractMatrix; kwargs...) = align(x, reshape(y, size(y)..., 1); kwargs...) |> z -> reshape(z, size(y)...)
align(x::AbstractVector, y::AbstractMatrix; kwargs...) = align(reshape(x, 3, :), reshape(y, 3, :, size(y, 2)); kwargs...) |> z->reshape(z, :, size(y, 2))
align(x::AbstractVector, y::AbstractVector; kwargs...) = align(reshape(x, 3, :), reshape(y, 3, :, 1); kwargs...) |> vec


"""
    aligned_rmsd(x::MatOrTensor, ys; weights) # conformational data
    aligned_rmsd(x::AbstractVector, y::AbstractVector; weights) # flattened one-one
    aligned_rmsd(x::AbstractVector, ys::AbstractMatrix; weights) # flattened one-many

Returns the vector of aligned Root mean square distances of conformation `x` to all conformations in `ys`
"""
function aligned_rmsd(x::MatOrTensor, ys; weights::MaybeWeights=nothing)
    ys = align(x, ys; weights)
    delta = ys .- x
    w, ws = weights_and_sum(weights, size(x, 2))
    d = sqrt.(sum(abs2, delta .* w', dims=(1, 2)) ./ ws)
    return vec(d)
end

aligned_rmsd(x::AbstractVector, y::AbstractVector; kwargs...) = aligned_rmsd(reshape(x, 3, :), reshape(y, 3, :); kwargs...)|>only
aligned_rmsd(x::AbstractVector, ys::AbstractMatrix; kwargs...) = aligned_rmsd(reshape(x, 3, :), reshape(ys, 3, :, size(ys, 2)); kwargs...)



"""
    pairwise_aligned_rmsd(xs::AbstractMatrix; mask::AbstractMatrix{Bool}, weights)

Compute the respectively aligned pairwise root mean squared distances between all conformations.

- `mask`: Allows to restrict the computation of the pairwise distances to only those pairs where the `mask` is `true`
- `weights`: Weights for the individual atoms in the alignement and distance calculations

Each column of `xs` represents a flattened conformation.
Returns the (n, n) matrix with the pairwise distances.
"""
function pairwise_aligned_rmsd(xs::AbstractMatrix, mask::AbstractSparseMatrix;
    weights::MaybeWeights=nothing,
    memsize=1_000_000_000)

    n = size(xs, 2)
    @assert size(mask) == (n, n)
    dists = similar(mask, eltype(xs))
    xs = reshape(xs, 3, :, n)

    i, j, _ = findnz(dists)

    batchsize = floor(Int, memsize / sizeof(@view xs[:, :, 1]))
    @views for l in Iterators.partition(1:length(i), batchsize)
        x = xs[:, :, i[l]]
        y = xs[:, :, j[l]]
        nonzeros(dists)[l] .= aligned_rmsd(x, y; weights)
    end
    return dists
end


batched_svd(x::CuArray) = svd(x)

function batched_svd(x)
    u = similar(x)
    v = similar(x)
    s = similar(x, size(x, 2), size(x, 3))
    for i in 1:size(x, 3)
        u[:, :, i], s[:, i], v[:, :, i] = svd(x[:, :, i])
    end
    return (; U=u, S=s, V=v)
end

"""
    aligntrajectory(traj::AbstractVector)
    aligntrajectory(traj::AbstractMatrix)

Align the framse in `traj` successively to each other.
`traj` can be passed either as Vector of vectors or matrix of flattened conformations.
"""
function aligntrajectory(traj::AbstractVector; kwargs...)
    aligned = [centermean(traj[1])]
    for x in traj[2:end]
        push!(aligned, align(aligned[end], x; kwargs...))
    end
    return aligned
end
aligntrajectory(traj::AbstractMatrix; kwargs...) = reduce(hcat, aligntrajectory(eachcol(traj); kwargs...))

centermean(x::AbstractMatrix) = x .- sum(x, dims=2) ./ size(x, 2)
centermean(x::AbstractVector) = reshape(x, 3, :) |> centermean |> vec



### Old or simple implementations

#=

function s_align(x::AbstractMatrix, y::AbstractMatrix; weights::MaybeWeights=nothing)
    w, ws = weights_and_sum(weights, size(x, 2))
    my = sum(y .* w', dims=2) / ws
    mx = sum(y .* w', dims=2) / ws
    wx = (x .- mx) .* sqrt.(weights')
    wy = (y .- my) .* sqrt.(weights')
    z = kabschrotation(wx, wy) * (x .- mx) .+ my
end
s_align(x::S, target::T; kwargs...) where {S<:AbstractVector,T<:AbstractVector} = as3dmatrix((x, y) -> align(x, y; kwargs...), x, target)

" compute R such that R*p is closest to q"
function kabschrotation(p::AbstractMatrix, q::AbstractMatrix)
    h = p * q'
    s = svd(h)
    R = s.V * s.U'
    return R
end


function pairwise_aligned_rmsd_delineated(xs::AbstractMatrix; mask::AbstractMatrix{Bool}=fill(true, size(xs, 2), size(xs, 2)), weights::MaybeWeights=nothing)
    n = size(xs, 2)
    @assert size(mask) == (n, n)
    mask = LinearAlgebra.triu(mask .|| mask', 1) .> 0 # compute each pairwise dist only once
    dists = fill!(similar(xs, n, n), 0)

    xs = reshape(xs, 3, :, n)
    @views for i in 1:n
        m = findall(mask[:, i])
        length(m) == 0 && continue
        x = xs[:, :, i]
        y = xs[:, :, m]
        dists[m, i] .= aligned_rmsd(x, y; weights)
    end

    dists .+= dists'
    dists[(mask+mask').==0] .= NaN
    return dists
end


function _pairwise_aligned_rmsd(xs)
    d3, n = size(xs)
    p = div(d3, 3)
    xs = reshape(xs, 3, p, n)
    xs = xs .- mean(xs, dims=2)
    dists = similar(xs, n, n)
    for i in 1:n
        for j in 1:n
            x = @view xs[:, :, i]
            y = @view xs[:, :, j]
            s = svd(x * y')
            r = s.V * s.U'
            dists[i, j] = dists[j, i] = sqrt(sum(abs2, r * x - y) / p)
        end
    end
    return dists
end

=#

end #module
using NNlib: batched_mul, batched_transpose
using SparseArrays

"""
    aligntrajectory(traj::AbstractVector)
    aligntrajectory(traj::AbstractMatrix)

Align the framse in `traj` successively to each other.
`traj` can be passed either as Vector of vectors or matrix of flattened conformations.
"""
function aligntrajectory(traj::AbstractVector; kwargs...)
    aligned = [centermean(traj[1])]
    for x in traj[2:end]
        push!(aligned, align(x, aligned[end]; kwargs...))
    end
    return aligned
end
aligntrajectory(traj::AbstractMatrix; kwargs...) = reduce(hcat, aligntrajectory(eachcol(traj); kwargs...))

centermean(x::AbstractMatrix) = x .- mean(x, dims=2)
centermean(x::AbstractVector) = as3dmatrix(centermean, x)


"""
    align(x::AbstractMatrix, target::AbstractMatrix)
    align(x::AbstractVector, target::AbstractVector)

Return `x` aligned to `target`
"""
align(x::AbstractMatrix, y::AbstractMatrix; kwargs...) = batched_kabsch_align(reshape(y, 3, :), reshape(x, 3, :, 1); kwargs...)[:, :, 1]
align(x::AbstractVector, y::AbstractVector; kwargs...) = batched_kabsch_align(reshape(y, 3, :), reshape(x, 3, :, 1); kwargs...) |> vec

MaybeWeights = Union{AbstractVector,Nothing}
weights_and_sum(weights::AbstractVector, n) = weights, sum(weights)
weights_and_sum(weights::Nothing, n) = 1, n

function s_align(x::AbstractMatrix, y::AbstractMatrix; weights::MaybeWeights=nothing)
    w, ws = weights_and_sum(weights, size(x, 2))
    my = sum(y .* w', dims=2) / ws
    mx = sum(y .* w', dims=2) / ws
    wx = (x .- mx) .* sqrt.(weights')
    wy = (y .- my) .* sqrt.(weights')
    z = kabschrotation(wx, wy) * (x .- mx) .+ my
end
s_align(x::S, target::T; kwargs...) where {S<:AbstractVector,T<:AbstractVector} = as3dmatrix((x, y) -> align(x, y; kwargs...), x, target)

function alignalong(x::AbstractMatrix, atoms::AbstractVector)
    n = size(x, 2)
    x = reshape(x, 3, :, n)
    y = alignalong(x, atoms)
    reshape(y, :, n)
end



""" align all frames to first frame along given atoms """
function alignalong(x::AbstractArray, atoms::AbstractVector)
    x = x .- mean(x[:, atoms, :], dims=2)
    for i in 1:size(x, 3)
        r = kabschrotation(x[:, atoms, i], x[:, atoms, 1])
        x[:, :, i] = r * x[:, :, i]
    end
    return x

end


" compute R such that R*p is closest to q"
function kabschrotation(p::AbstractMatrix, q::AbstractMatrix)
    h = p * q'
    s = svd(h)
    R = s.V * s.U'
    return R
end

"""
    aligned_rmsd(p::AbstractMatrix, q::AbstractMatrix)
    aligned_rmsd(p::AbstractVector, q::AbstractVector)

Return the aligned root mean squared distance between conformations `p` and `q`, passed either flattened or as (3,d) matrix
"""
function aligned_rmsd(p::AbstractMatrix, q::AbstractMatrix)
    p = align(p, q)
    n = size(p, 2)
    norm(p - q) / sqrt(n)
end
aligned_rmsd(p::AbstractVector, q::AbstractVector) = aligned_rmsd(reshape(p, 3, :), reshape(q, 3, :))



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
        nonzeros(dists)[l] = batched_kabsch_rmsd(x, y; weights) |> cpu
    end
    return dists
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
        dists[m, i] .= batched_kabsch_rmsd(x, y; weights)
    end

    dists .+= dists'
    dists[(mask+mask').==0] .= NaN
    return dists
end

"""
    batched_kabsch_rmsd(x::AbstractMatrix, ys::AbstractArray{<:Any, 3})
    batched_kabsch_rmsd(x::AbstractVector, ys::AbstractMatrix)

Returns the vector of aligned Root mean square distances of conformation `x` to all conformations in `ys`
"""
batched_kabsch_rmsd(x::AbstractVector, ys::AbstractMatrix; kwargs...) =
    batched_kabsch_rmsd(reshape(x, 3, :), reshape(ys, 3, :, size(ys, 2)); kwargs...)

function batched_kabsch_rmsd(x, ys; weights::MaybeWeights=nothing)
    ys = batched_kabsch_align(x, ys; weights)
    delta = ys .- x
    w, ws = weights_and_sum(weights, size(x, 2))
    d = sqrt.(sum(abs2, delta .* w', dims=(1, 2)) ./ ws)
    return vec(d)
end

"""
    batched_kabsch_align(x::AbstractMatrix, ys::AbstractArray{<:Any, 3}, weights=nothing)

Align all structures in `ys` to the one in `x`, where `weights` specifies the weight of individual atoms for the alignment.

`size(x) = (d, n), size(y) = (d, n, m)` where d is the systems dimension (usually 3), n number of particles and m number of structures to align`
"""
function batched_kabsch_align(x::Union{AbstractMatrix,AbstractArray{<:Any,3}}, ys::AbstractArray{<:Any,3}; weights::MaybeWeights=nothing)
    w, ws = weights_and_sum(weights, size(x, 2))
    m = sum(x .* w', dims=2) / ws
    x = x .- m
    ys = ys .- sum(ys .* w', dims=2) / ws

    h = batched_mul(x .* w', batched_transpose(ys))
    s = batched_svd(h)
    r = batched_mul(s.U, batched_transpose(s.V))

    ys = batched_mul(r, ys) .+ m
    return ys
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

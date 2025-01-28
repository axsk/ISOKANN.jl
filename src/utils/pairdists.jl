"""
    flatpairdists(x)

Assumes each col of x to be a flattened representation of multiple 3d coords.
Returns the flattened pairwise distances as columns."""
function flatpairdists(x, cols=:)
    d = size(x, 1)
    s = size(x)[2:end]
    #d, s... = size(x)  # slower for zygote autodiff
    c = div(d, 3)

    b = reshape(x, 3, c, :)
    if !(cols isa Colon)
        b = b[:, cols, :]
        c = length(cols)
    end
    p = sqpairdist(b)

    inds = halfinds(c)
    p = p[inds, :]
    p = sqrt.(p)
    return reshape(p, length(inds), s...)
end

"""
    sqpairdist(x::AbstractArray)

Compute the squared pairwise distances between the columns of `x`.
If `x` has 3 dimensions, the computation is batched along the 3rd dimension.
"""
function sqpairdist(x::AbstractArray{<:Any,3})
    p = -2 .* batched_mul(batched_adjoint(x), x) .+ sum(abs2, x, dims=1) .+ PermutedDimsArray(sum(abs2, x, dims=1), (2, 1, 3))
    return p
end

sqpairdist(x::AbstractMatrix) = sqpairdist(reshape(x, size(x)..., 1))[:, :, 1]

pairdist(x::AbstractArray) = sqrt.(sqpairdist(x))
pairdist(x::AbstractMatrix) = Distances.pairwise(Distances.Euclidean(), x, dims=2) # this is saving on computing the symmetric part but doesnt work batched or with cuda

using LinearAlgebra: diagind, UpperTriangular


# return the indices of the upperdiagonal """
function halfinds(n)
    a = UpperTriangular(ones(n, n))
    ChainRulesCore.ignore_derivatives() do
        a[diagind(a)] .= 0
    end
    findall(a .> 0)
end

"""
    pairdistfeatures(inds::AbstractVector)

Returns a featurizer function which computes the pairwise distances between the particles specified by `inds`
"""
function pairdistfeatures(inds::AbstractVector)
    n = div(length(inds), 3)^2
    function features(x)
        x = selectrows(x, inds)
        x = flatpairdists(x)
        return x
    end
    features
end


### Localized pairwise distances

"""
    localpdistinds(coords::AbstractMatrix, radius)

Given `coords` of shape ( 3n x frames ) return the pairs of indices whose minimal distance along all frames is at least once lower then radius
"""
function localpdistinds(coords::AbstractMatrix, radius)
    traj = reshape(coords, 3, :, size(coords, 2))
    ds = sqpairdist(traj)
    mds = dropdims(minimum(ds, dims=3), dims=3)
    pairs = Tuple.(findall(0 .< UpperTriangular(mds) .<= radius^2))
end

localpdistinds(coords::Vector, radius) = localpdistinds(hcat(coords), radius)

""" restricted_localpdistinds(coords, radius, atoms)

Like `localdists`, but consider only the atoms with index in `atoms`
"""
function restricted_localpdistinds(coords, radius, atoms)
    rc = reshape(reshape(coords, 3, :, size(coords, 2))[:, atoms, :], :, size(coords, 2))
    pairs = localpdistinds(rc, radius)
    map(pairs) do (a, b)
        (atoms[a], atoms[b])
    end
end


"""
    pdists(coords::AbstractArray, inds::Vector{<:Tuple})

Compute the pairwise distances between the particles specified by the tuples `inds` over all frames in `traj`.
Assumes a column contains all 3n coordinates.
"""
function pdists(coords::AbstractMatrix, inds::Vector{Tuple{T,T}}) where {T}
  a = first.(inds)
  b = last.(inds)
  n = size(coords, 2)
  traj = reshape(coords, 3, :, n)
  A = @views traj[:, a, :]
  B = @views traj[:, b, :]
  D = sqrt.(sum(abs2.(A .- B), dims=1))
  return dropdims(D, dims=1)
end


# batched variant
function pdists(x::AbstractArray, inds::Vector{Tuple{T,T}}) where {T}
  d, s... = size(x)
  b = reshape(x, d, :)
  p = pdists(b, inds)
    return reshape(p, length(inds), s...)
end

# convenience wrapper
function localpdists(coords, radius)
  inds = localpdistinds(coords, radius)
  dists = pdists(coords, inds)
  dists, inds
end
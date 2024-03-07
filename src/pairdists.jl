"""
    flatpairdists(x)

Assumes each col of x to be a flattened representation of multiple 3d coords.
Returns the flattened pairwise distances as columns."""
function flatpairdists(x)
    d, s... = size(x)
    c = div(d, 3)
    inds = halfinds(c)
    b = reshape(x, 3, c, :)
    p = sqpairdist(b)[inds, :]
    p = sqrt.(p)
    return reshape(p, :, s...)
end

"""
    simplepairdists(x::AbstractArray{<:Any,3})

Compute the pairwise distances between the columns of `x`, batched along the 3rd dimension.
"""
function sqpairdist(x::AbstractArray{<:Any,3})
    p = -2 .* batched_mul(batched_adjoint(x), x) .+ sum(abs2, x, dims=1) .+ PermutedDimsArray(sum(abs2, x, dims=1), (2, 1, 3))
    return p
end

#using Distances: pairwise, Euclidean
using LinearAlgebra: diagind, UpperTriangular
# using Distances
#function batchedpairdists(x)
#    inds = halfinds(size(x, 2))
#    dropdims(mapslices(x -> pairwise(Euclidean(), x, dims=2)[inds], x, dims=(1, 2)), dims=2)
#end

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


### custom implementation of multithreaded pairwise distances
#=

function batchedpairdists_threaded(x::AbstractArray)
    ChainRulesCore.@ignore_derivatives begin
        d, n, cols = size(x)
        out = similar(x, n, n, cols)
        @views Threads.@threads for i in 1:cols
            pairdistkernel(@views(out[:, :, i]), x[:, :, i])
        end
        out
    end
end

function pairdistkernel(out::AbstractMatrix, x::AbstractMatrix)
    @assert size(x, 1) == 3
    n, k = size(out)
    @views for i in 1:n, j in i:k
        out[i, j] = out[j, i] = ((x[1, i] - x[1, j])^2 + (x[2, i] - x[2, j])^2 + (x[3, i] - x[3, j])^2)
    end
end
=#

### alternative implementation using SliceMap.jl and Distances.jl, was a little bit slower

#=
import SliceMap  # provides slicemap == mapslices but with zygote gradients
using Distances

" Threaded computation of pairwise dists using SliceMap.jl and Distances.jl"
function threadpairdists(x)
    ChainRulesCore.@ignore_derivatives begin
        d, s... = size(x)
        x = reshape(x, d, :)
        pd = SliceMap.ThreadMapCols(pairwisevec, x)
        pd = reshape(pd, :, s...)
        return pd
    end
end

" Non-threaded computation of pairwise dists using SliceMap.jl and Distances.jl"
slicemapdist(x) = ChainRulesCore.@ignore_derivatives SliceMap.slicemap(pairwisevec, x, dims=1)

pairwisevec(col::AbstractVector) = vec(pairwise(SqEuclidean(), reshape(col,3,:), dims=2))

=#

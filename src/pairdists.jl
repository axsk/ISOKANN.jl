
import ChainRulesCore


# Take an array of shape (i,j...) where the i=3*n vectors are the flattened coords of n 3d points.
# returns the flattened pairwise dists in an array of shape (n^2, j)
" Threaded computation of pairwise dists for x::AbstractArray.
Assumes each col of x to be a flattened representation of multiple 3d coords.
Returns the flattened pairwise distances as columns."
function flatpairdists(x)
    d, s... = size(x)
    c = div(d,3)
    b = reshape(x, 3, c, :)
    p = simplepairdists(b)
    return reshape(p, c*c, s...)
end

function simplepairdists(x)
        p = -2 .* batched_mul(batched_adjoint(x), x) .+ sum(abs2, x, dims=1) .+ PermutedDimsArray(sum(abs2, x, dims=1), (2,1,3))
        return p
end


### custom implementation of multithreaded pairwise distances
function batchedpairdists(x::AbstractArray)
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
        out[i,j] = out[j,i] = ((x[1,i]-x[1,j])^2 + (x[2,i]-x[2,j])^2 + (x[3,i]-x[3,j])^2)
    end
end


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

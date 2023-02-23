
import ChainRulesCore

### custom implementation of multithreaded pairwise distances

" Threaded computation of pairwise dists for x::AbstractArray.
Assumes each col of x to be a flattened representation of multiple 3d coords.
Returns the flattened pairwise distances as columns."
function mythreadpairdists(x::AbstractArray)
    ChainRulesCore.@ignore_derivatives begin
        d, s... = size(x)
        xx = reshape(x, d, :)
        cols = size(xx, 2)
        n = div(d, 3)
        out = similar(x, n, n, cols)
        @views Threads.@threads for i in 1:cols
            c = reshape(xx[:, i], 3, :)
            pairdistkernel(@views(out[:, :, i]), c)
        end
        reshape(out, n*n, s...)
    end
end

function pairdistkernel(out::AbstractMatrix, x::AbstractMatrix)
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

""" This file contains 'fixes' that enable the GPU/CUDA support for ISOKANN """

pickclosest(haystack::CuArray, needles::AbstractVector) = pickclosest(collect(haystack), needles)

""" gpu(iso::IsoRun)

move the model and data of the given `IsoRun` to the GPU for CUDA support
"""
function Flux.gpu(iso::IsoRun)
    (; nd, nx, np, nl, nres, ny, nk, nxmax, sim, model, opt, data, losses, loggers, minibatch) = iso
    model = Flux.gpu(model)
    data = Flux.gpu(data)
    opt = Flux.gpu(opt)
    return IsoRun(; nd, nx, np, nl, nres, ny, nk, nxmax, sim, model, opt, data, losses, loggers, minibatch)
end

function Flux.cpu(iso::IsoRun)
    (; nd, nx, np, nl, nres, ny, nk, nxmax, sim, model, opt, data, losses, loggers, minibatch) = iso
    model = Flux.cpu(model)
    data = Flux.cpu(data)
    opt = Flux.cpu(opt)
    return IsoRun(; nd, nx, np, nl, nres, ny, nk, nxmax, sim, model, opt, data, losses, loggers, minibatch)
end

propagate(s::OpenMMSimulation, x0::CuArray, ny; nthreads=Threads.nthreads()) = cu(propagate(s, collect(x0), ny; nthreads))




# given an (i,j,k) sized array compute the pairwise dists between j points in i dimensions,
# batched along dimension k. return a (j,j,k) sized array.
# in principle this also works for normal Arrays.
# Its twice as slow however since it is not using the symmetry
#function batchedpairdists(x::CuArray{<:Any,3})
#    p = -2 .* batched_mul(batched_adjoint(x), x) .+ sum(abs2, x, dims=1) .+ PermutedDimsArray(sum(abs2, x, dims=1), (2, 1, 3))
#    return p
#end

#function batchedpairdists(x::CuArray{<:Any,4})
#    gpu(batchedpairdists(cpu(x)))
#end

centercoordscoords(xs::CuArray) = cu(centercoordscoords(collect(xs)))

datastats(data::Tuple{<:CuArray,<:CuArray}) = datastats(collect.(data))


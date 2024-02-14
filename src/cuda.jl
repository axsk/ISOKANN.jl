""" This file contains 'fixes' that enable the GPU/CUDA support for ISOKANN """

pickclosest(haystack::CuArray, needles::AbstractVector) = pickclosest(collect(haystack), needles)

""" gpu(iso::IsoRun)

move the model and data of the given `IsoRun` to the GPU for CUDA support
"""
function gpu(iso::IsoRun)
    (; nd, nx, np, nl, nres, ny, nk, nxmax, sim, model, opt, data, losses, loggers, minibatch) = iso
    model = Flux.gpu(model)
    data = Flux.gpu(data)
    opt = Flux.gpu(opt)
    return IsoRun(; nd, nx, np, nl, nres, ny, nk, nxmax, sim, model, opt, data, losses, loggers, minibatch)
end

propagate(s::OpenMMSimulation, x0::CuArray, ny; nthreads=Threads.nthreads()) = cu(propagate(s, collect(x0), ny; nthreads))

""" Fallback to simulate MD dynamics on the CPU """
propagate(ms::MollyLangevin, x0::CuMatrix, ny) = propagate(ms, collect(x0), ny)


# given an (i,j,k) sized array compute the pairwise dists between j points in i dimensions,
# batched along dimension k. return a (j,j,k) sized array.
# in principle this also works for normal Arrays.
# Its twice as slow however since it is not using the symmetry
function batchedpairdists(x::CuArray)
    p = -2 .* batched_mul(batched_adjoint(x), x) .+ sum(abs2, x, dims=1) .+ PermutedDimsArray(sum(abs2, x, dims=1), (2, 1, 3))
    return p
end

centercoordscoords(xs::CuArray) = cu(centercoordscoords(collect(xs)))

datastats(data::Tuple{<:CuArray,<:CuArray}) = datastats(collect.(data))

""" move the ::System to the GPU, mirroring behavior of Flux.gpu """
gpu(sys::System) = System(sys;
    atoms=cu(sys.atoms),
    atoms_data=cu(sys.atoms_data),
    coords=cu(sys.coords),
    velocities=cu(sys.velocities)
)

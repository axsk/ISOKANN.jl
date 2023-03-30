""" This file contains 'fixes' that enable the GPU/CUDA support for ISOKANN """

using CUDA
using Flux

pickclosest(haystack::CuArray, needles::AbstractVector) = pickclosest(collect(haystack), needles)

function gpu!(iso::IsoRun)
    iso.model = Flux.gpu(iso.model)
    iso.data = Flux.gpu(iso.data)
    return
end

propagate(ms::MollyLangevin, x0::CuMatrix, ny) = propagate(ms, collect(x0), ny)


using NNlib, NNlibCUDA
# given an (i,j,k) sized array compute the pairwise dists between j points in i dimensions,
# batched along dimension k. return a (j,j,k) sized array.
# in principle this also works for normal Arrays.
# Its twice as slow however since it is not using the symmetry
function batchedpairdists(x::CuArray)
    p = -2 .* batched_mul(batched_adjoint(x), x) .+ sum(abs2, x, dims=1) .+ PermutedDimsArray(sum(abs2, x, dims=1), (2,1,3))
    return p
end

center(xs::CuArray) = cu(center(collect(xs)))

datastats(data::Tuple{<:CuArray, <:CuArray}) = datastats(collect.(data))

""" move the ::System to the GPU, mirroring behavior of Flux.gpu """
gpu(sys::System) = System(sys;
    atoms = cu(sys.atoms),
    atoms_data = cu(sys.atoms_data),
    coords = cu(sys.coords),
    velocities = cu(sys.velocities),
)

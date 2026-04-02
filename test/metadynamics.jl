using ISOKANN
using ISOKANN: MetadynamicsSimulator, MetadynamicsStateMatrix, MetadynamicsState,
               bias_potential, rescale_welltempered, wt_free_energy
using Test

@testset "MetadynamicsSimulator" begin
    iso = Iso(OpenMMSimulation(), nx=10)
    run!(iso, 1)
    ms = MetadynamicsSimulator(iso)
    trajectory(ms)
end

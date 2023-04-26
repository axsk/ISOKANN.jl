using ISOKANN
#using Flux
using Test

@testset "IsoRun CPU" begin
    iso = ISOKANN.IsoRun(loggers=[])
    ISOKANN.run!(iso)
end

@testset "IsoRun GPU" begin
    using Flux
    using CUDA
    CUDA.allowscalar(false)
    iso = ISOKANN.IsoRun()
    iso.loggers = []
    ISOKANN.gpu!(iso)
    ISOKANN.run!(iso)
end

@testset "ISOKANN.jl" begin
    using ISOKANN.IsoForce: test_GirsanovSDE, test_optcontrol, isokann, Doublewell
    test_GirsanovSDE()
    test_optcontrol()

    @time isokann()
    @time isokann(dynamics = Doublewell(dim=2))
end

using ISOKANN
#using Flux
using Test

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
    ISOKANN.test_GirsanovSDE()
    ISOKANN.test_optcontrol()

    @time isokann()
    @time isokann(dynamics = ISOKANN.Doublewell(dim=2))
end

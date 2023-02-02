using ISOKANN
using Test

@testset "ISOKANN.jl" begin
    ISOKANN.test_GirsanovSDE()
    ISOKANN.test_optcontrol()

    @time isokann()
    @time isokann(dynamics = ISOKANN.Doublewell(dim=2))
end

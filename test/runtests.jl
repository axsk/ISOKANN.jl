using ISOKANN
#using Flux
using Test

@testset "ISOKANN.jl" begin

    @testset "IsoRun CPU" begin
        iso = ISOKANN.IsoRun(loggers=[])
        ISOKANN.run!(iso)
    end

    @testset "IsoRun GPU" begin
        using Flux
        using CUDA
        if CUDA.functional()
            CUDA.allowscalar(false)
            iso = ISOKANN.IsoRun(loggers=[])
            ISOKANN.gpu!(iso)
            ISOKANN.run!(iso)
        end
    end

    @testset "IsoForce" begin
        @test_broken ISOKANN.IsoForce.isokann(usecontrol=true)  # broken since IsoForce is not included in ISOKANN any more
    end
    #=
        @testset "IsoForce (deprecated)" begin
            using ISOKANN.IsoForce: test_GirsanovSDE, test_optcontrol, isokann, Doublewell
            test_GirsanovSDE()
            test_optcontrol()

            @test_broken @time isokann()
            @test_broken @time isokann(dynamics = Doublewell(dim=2))
        end
    =#
end

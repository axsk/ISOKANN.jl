using ISOKANN
#using Flux
using Test

@testset "ISOKANN.jl" begin

    @testset "IsoRun CPU" begin
        iso = ISOKANN.IsoRun()
        ISOKANN.run!(iso)
        @test true
    end

    @testset "IsoRun GPU" begin
        using Flux
        using CUDA
        if CUDA.functional()
            CUDA.allowscalar(false)
            iso = ISOKANN.IsoRun()
            ISOKANN.gpu!(iso)
            ISOKANN.run!(iso)
            @test true
        else
            @info "No functional GPU found. Marking GPU test as broken."
            @test_broken false
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

    @testset "iso2" begin

        @testset "iso2 Doublewell" begin
            iso2(nd=2, sim=Doublewell())
            @test true
        end

        @testset "iso2 Triplewell" begin
            iso2(nd=3, sim=Triplewell())
            @test true
        end

        @testset "iso2 MuellerBrown" begin
            iso2(nd=3, sim=MuellerBrown())
            @test true
        end

        @testset "iso2 TransformPseudoInv" begin
            iso2(nd=3, sim=Triplewell(), transform=ISOKANN.TransformPseudoInv())
            @test true
        end
    end

    @testset "iso2 OpenMM" begin
        sim = ISOKANN.OpenMM.OpenMMSimulation2()
        exp = iso2(; sim, nx=20, ny=5, n=100, lr=1e-3)
        @test true
    end

end

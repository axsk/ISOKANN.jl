using ISOKANN
#using Flux
using Test

@testset "ISOKANN.jl" verbose = true begin
    for (sim, name) in zip([Doublewell(), Triplewell(), MuellerBrown(), ISOKANN.OpenMM.OpenMMSimulation()], ["Doublewell", "Triplewell", "MuellerBrown", "OpenMM"])
        @testset "Iso2 $name" begin
            i = Iso2(sim)
            @test true
            run!(i)
            @test true
        end
    end

    @testset "Iso2 GPU" begin
        using CUDA
        if CUDA.functional()
            CUDA.allowscalar(false)
            iso = Iso2(MuellerBrown()) |> gpu |> run!
            @test true
        else
            @info "No functional GPU found. Marking GPU test as broken."
            @test_broken false
            end
        end

        @testset "Iso2 Dialanine with adaptive sampling" begin
        sim = OpenMMSimulation()
        data = SimulationData(sim, 10, 10)
        iso = Iso2(sim)
        runadaptive!(iso)
            @test true
        end

        @testset "Iso2 Transforms" begin
        sim = MuellerBrown()
        for t in [ISOKANN.TransformShiftscale(), ISOKANN.TransformPseudoInv(), ISOKANN.TransformISA()]
            run!(Iso2(sim, transform=t))
            @test true
        end
    end

    @testset "IsoMu.jl" begin
        using ISOKANN.IsoMu
        datapath = "/data/numerik/ag_cmd/trajectory_transfers_for_isokann/data/8EF5_Aspargine_116_100ns_7.4_v2"
        if !ispath(datapath)
            println("could not access IsoMu datapath $datapath")
            return @test_broken false
        end
        iter = 10

        data = DataLink(datapath, startpos=2)
        mu = isokann(data, gpu=false)
        train!(mu, iter)
        out = Base.Filesystem.tempname() * ".pdb"
        save_reactive_path(mu, sigma=0.05, out=out)

        @test true
    end

    @testset "vgv" begin
        datapath = ISOKANN.VGV_DATA_DIR
        if !ispath(datapath)
            println("could not access VGV datapath $datapath")
            return @test_broken false
        end
        v = ISOKANN.VGVData(datapath, nx=10, nk=2)
        ISOKANN.vgv_examplerun(v, Base.Filesystem.tempname())
        @test true
    end

end
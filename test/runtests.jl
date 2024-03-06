using ISOKANN
#using Flux
using Test
using CUDA

if CUDA.functional()
    CUDA.allowscalar(false)
else
    @info "No functional GPU found. Marking GPU test as broken."
    @test_broken false
end


simulations = zip([Doublewell(), Triplewell(), MuellerBrown(), ISOKANN.OpenMM.OpenMMSimulation()], ["Doublewell", "Triplewell", "MuellerBrown", "OpenMM"])

@testset "ISOKANN.jl" verbose = true begin

    for backend in [cpu, gpu]
        for (sim, name) in simulations
            @testset "Testing ISOKANN with $name ($backend)" begin
                i = Iso2(sim) |> backend
                @test true
                run!(i)
                @test true
                @test runadapative!(i, generations=2, nx=1, iter=1)
                @test true
            end
        end

        @testset "Iso2 Transforms ($backend)" begin
            sim = MuellerBrown()
            for t in [ISOKANN.TransformShiftscale(), ISOKANN.TransformPseudoInv(), ISOKANN.TransformISA()]
                run!(Iso2(sim, transform=t) |> backend)
                @test true
            end
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
using ISOKANN
#using Flux
using Test

@testset "ISOKANN.jl" verbose = true begin

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
            iso = ISOKANN.gpu(iso)
            ISOKANN.run!(iso)
            @test true
        else
            @info "No functional GPU found. Marking GPU test as broken."
            @test_broken false
        end
    end

    @testset "iso2" verbose = true begin

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

    @testset "Iso2" begin
        run!(Iso2(IsoRun()))
    end

    @testset "iso2 OpenMM" begin
        sim = ISOKANN.OpenMM.OpenMMSimulation()
        exp = iso2(; sim, nx=20, ny=5, n=100, lr=1e-3)
        @test true
    end

    @testset "iso2 Molly" begin
        sim = MollyLangevin(sys=PDB_ACEMD())
        exp = iso2(; sim, nx=20, ny=5, n=100, lr=1e-3)
        @test true
    end

    @testset "compare simulators" begin
        sim_molly = MollyLangevin(sys=PDB_ACEMD(), T=1.0)
        sim_omm = OpenMMSimulation(pdb="$(@__DIR__)/../data/alanine-dipeptide-nowater av.pdb",
            steps=500,
            features=1:22,
            forcefields=["amber14-all.xml"])

        x0 = ISOKANN.randx0(sim_molly, 2)
        @time propagate(sim_molly, x0, 100)
        @time propagate(sim_omm, x0, 100)
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
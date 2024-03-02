using ISOKANN
#using Flux
using Test

@testset "ISOKANN.jl" verbose = true begin

    @testset "IsoRun CPU" begin
        iso = ISOKANN.IsoRun(nd=10) |> run!
        @test true
    end

    @testset "IsoRun GPU" begin
        using Flux
        using CUDA
        if CUDA.functional()
            CUDA.allowscalar(false)
            iso = ISOKANN.IsoRun(nd=10) |> gpu |> run!
            @test true
        else
            @info "No functional GPU found. Marking GPU test as broken."
            @test_broken false
        end
    end

    @testset "Iso2" begin
        @testset "Iso2 from IsoRun" begin
            run!(Iso2(IsoRun(nd=10)))
            @test true
        end

        for (sim, name) in zip([Doublewell(), Triplewell(), MuellerBrown(), ISOKANN.OpenMM.OpenMMSimulation(), MollyLangevin(sys=PDB_ACEMD())], ["Doublewell", "Triplewell", "MuellerBrown", "OpenMM", "Molly"])
            @testset "Iso2 $name" begin
                i = Iso2(sim)
                @test true
                run!(i)
                @test true
            end
        end

        @testset "Iso2 Dialanine with adaptive sampling" begin
            sim = MollyLangevin(sys=PDB_ACEMD())
            model = ISOKANN.pairnet(484, features=ISOKANN.flatpairdists, nout=3)
            opt = Flux.setup(Flux.AdamW(1e-3, (0.9, 0.999), 1e-4), model)

            iso = Iso2(sim; nx=100, nk=10, nd=1, model, opt)
            iso = run!(iso, sim, 10, 100, ny=10)
            @test true
        end

        @testset "Iso2 Transforms" begin
            @test_broken false
        end
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
        @test_broken sim.molly == sim.omm
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
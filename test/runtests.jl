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

@time begin


@testset "ISOKANN.jl" verbose = true begin

    simulations = zip([Doublewell(), Triplewell(), MuellerBrown(), ISOKANN.OpenMM.OpenMMSimulation(), ISOKANN.OpenMM.OpenMMSimulation(features=0.3)], ["Doublewell", "Triplewell", "MuellerBrown", "OpenMM", "OpenMM localdists"])

    for backend in [cpu, gpu]

        @testset "Running basic system tests" begin
            for (sim, name) in simulations
                @testset "Testing ISOKANN with $name ($backend)" begin
                    i = Iso2(sim) |> backend
                    @test true
                    run!(i)
                    @test true
                    runadaptive!(i, generations=2, nx=1, iter=1)
                    @test true
                    ISOKANN.addextrapolates!(i, 1, stepsize=0.1, steps=1)
                    @test true
                end
            end
        end

        @testset "Iso2 Transforms ($backend)" begin
            sim = MuellerBrown()
            for (d, t) in zip([1, 2, 2], [ISOKANN.TransformShiftscale(), ISOKANN.TransformPseudoInv(), ISOKANN.TransformISA()])
                    @test begin
                        run!(Iso2(sim, model=pairnet(2, nout=d), transform=t) |> backend)
                        true
                    end
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

    @testset "Iso2 and IsoSimulation operations" begin
        iso = Iso2(OpenMMSimulation(), nx=10)
        iso.data = iso.data[6:10] # data slicing
        path =  Base.Filesystem.tempname() * ".jld2"
        ISOKANN.save(path, iso)
        isol = ISOKANN.load(path, iso)
        @assert iso.data.coords == isol.data.coords
        runadaptive!(isol, generations=1, nx=1, iter=1)
        @test true
    end
end

end
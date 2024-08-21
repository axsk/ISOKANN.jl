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
                    ISOKANN.addextrapolates!(i, 1, stepsize=0.01, steps=1)
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
using ISOKANN
#using Flux
using Test
using CUDA

@time @testset "ISOKANN.jl" verbose = true begin

backends = Any[cpu]
if CUDA.functional()
    CUDA.allowscalar(false)
    push!(backends, gpu)
else
    @info "No functional GPU found. Skipping GPU tests"
end

@time @testset "workflow run" verbose = true begin
    include("workflow.jl")
end

function with_possible_broken_domain(f)
    try
        r = f()
        @test true
        return r
    catch e
        if e isa DomainError
            @test_broken rethrow(e)
        else
            @test rethrow(e)
        end
    end
end

data = SimulationData(OpenMMSimulation(), 100, 8)



simulations = zip([Doublewell(), Triplewell(), MuellerBrown(), ISOKANN.OpenMM.OpenMMSimulation()], ["Doublewell", "Triplewell", "MuellerBrown", "OpenMM", ])

for backend in backends

    @testset "Running basic system tests on $backend" begin
        for (sim, name) in simulations
            @testset "Testing ISOKANN with $name" begin
                i = Iso(sim) |> backend
                run!(i)
                runadaptive!(i, generations=2, kde=1, iter=1)
                @test true
            end
        end
    end

    @testset "Iso Transforms ($backend)" begin
        for (d, t) in zip([1, 2, 2], [ISOKANN.TransformShiftscale(), ISOKANN.TransformPseudoInv(), ISOKANN.TransformISA()])
            @testset "$t" begin
                run!(Iso(data, model=pairnet(data, nout=d), transform=t) |> backend)
                @test true
            end
        end
    end
end


@testset "Iso and IsoSimulation operations" begin
    iso = Iso(OpenMMSimulation(), nx=10)
    iso.data = iso.data[6:10] # data slicing
    path = Base.Filesystem.tempname() * ".jld2"
    ISOKANN.save(path, iso)
    isol = ISOKANN.load(path)
    @assert iso.data.coords == isol.data.coords
    runadaptive!(isol, generations=1, kde=1, iter=1)
    @test true
end

end


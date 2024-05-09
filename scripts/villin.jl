using ISOKANN
using ISOKANN.OpenMM
using Dates

## Config

pdb = "data/villin nowater.pdb"
steps = 5_000
temp = 310
features = 0.5 # 0 => backbone only
forcefields = OpenMM.FORCE_AMBER_IMPLICIT
addwater = true
padding = 1
ionicstrength = 0.0

nx = 8
nk = 4
iter = 1000
generations = 1000
cutoff = 3000

kde_padding = 0.02
extrapolates = 0 # *2
extrapolate = 0.05
keepedges = false

layers = 4
minibatch = 100
opt = ISOKANN.NesterovRegularized(1e-3, 1e-4)

sigma = 2

path = "out/villin/$(now())"[1:end-4]

readdata = nothing
#readdata = "latest/iso.jld2"

## Initialisation

lagtime = steps * 0.002 / 1000 # in nanoseconds
simtime_per_gen = lagtime * (nx + 2 * extrapolates) * nk # in nanoseconds

println("lagtime: $lagtime ns")
println("simtime per generation: $simtime_per_gen ns")

@time "creating system" sim = OpenMMSimulation(;
    pdb, steps, forcefields, features,
    temp, nthreads=1, mmthreads="gpu")

if addwater
    @time "adding water" sim = OpenMMSimulation(;
        pdb, steps, forcefields,
        temp, nthreads=1, mmthreads="gpu",
        features=sim.features, addwater=true, padding, ionicstrength)
end

data = if readdata isa String
    @time "reading initial data" let i = ISOKANN.load(readdata)
        SimulationData(sim, i.data.coords)
    end
else
    @time "generating initial data" SimulationData(sim, nx, nk)
end

iso = Iso2(data;
    opt, minibatch,
    model=pairnet(length(sim.features); layers),
    gpu=true,
    loggers=[])

@time "initial training" run!(iso, iter)

## Running

for i in 1:generations
    GC.gc()

    @time "extrapolating" ISOKANN.addextrapolates!(iso, extrapolates, stepsize=extrapolate)
    @time "kde sampling" ISOKANN.resample_kde!(iso, nx; padding=kde_padding)
    @time "training" run!(iso, iter)

    simtime = ISOKANN.simulationtime(iso)

    @time "saving" begin

        if i == 1
            mkpath(path)
            run(`ln -snf $path latest`)
            cp(@__FILE__, "$path/script.jl")
            open("$path/script.jl", "a") do io
                sha = readchomp(`git rev-parse HEAD`)
                write(io, "\n#git-commit: $sha")
            end
            chmod("$path/script.jl", 0o444)
        end

        save_reactive_path(iso, out="$path/villin_fold_$(simtime)ps.pdb"; sigma, maxjump=0.05)
        ISOKANN.savecoords("$path/data.pdb", iso)
        ISOKANN.Plots.savefig(plot_training(iso), "$path/villin_fold_$(simtime)ps.png")
        println("\n status: $path/villin_fold_$(simtime)ps.png \n")
        ISOKANN.save("$path/iso.jld2", iso)
    end
end

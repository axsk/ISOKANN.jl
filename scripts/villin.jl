using ISOKANN
using ISOKANN.OpenMM
using Dates
using PyCall

## Config

comment = "test"

pdb = "data/villin nowater.pdb"
steps = 10_000
step = 0.002
temp = 310
friction = 0
minimize = true
features = 0.5 # 0 => backbone only
#forcefields = OpenMM.FORCE_AMBER_IMPLICIT  # TODO: this shouldnb be an option the way we build addwater now
forcefields = ISOKANN.OpenMM.FORCE_AMBER_IMPLICIT
addwater = false
padding = 0.01
ionicstrength = 0.0

nx0 = 1000
nx = 30
nchistrat = 30
nk = 1
iter = 300
generations = 2500
cutoff = 10_000

kde_padding = 0.02
extrapolates = 0 # *2
extrapolate = 0.05
keepedges = true

layers = 4
minibatch = 1000
opt = ISOKANN.NesterovRegularized(1e-3, 1e-4)

sigma = 2
maxjump = 1

path = "out/villin/$(now())"[1:end-4] * comment


readdata = nothing
#readdata = "latest/iso.jld2"

## Initialisation

lagtime = steps * step / 1000 # in nanoseconds
simtime_per_gen = lagtime * (nx + nchistrat + 2 * extrapolates) * nk # in nanoseconds

println("lagtime: $lagtime ns")
println("simtime per generation: $simtime_per_gen ns")

@time "creating system" sim = OpenMMSimulation(;
    pdb, steps, forcefields, features, friction, step, temp, addwater, padding, ionicstrength, minimize)





data = if readdata isa String
    @time "reading initial data" let i = ISOKANN.load(readdata)
        SimulationData(sim, i.data.coords)
    end
else
    @time "generating initial data" SimulationData(sim, nx0, nk)
end

iso = Iso(data;
    opt, minibatch,
    model=pairnet(data; layers),
    gpu=true,
    loggers=[])

@time "initial training" run!(iso, iter)
data = nothing
## Running

newsim = true
simtime = ISOKANN.simulationtime(iso)

function loop()
    for i in 1:generations
        global simtime

        GC.gc()
        if length(iso.data) > cutoff
            iso.data = iso.data[end-cutoff+1:end]
        end

        #@show varinfo()
        GC.gc()

        #simtime -= ISOKANN.simulationtime(iso)
        nchistrat > 0 && @time "chistratsampling" ISOKANN.resample_strat!(iso, nchistrat; keepedges)
        extrapolates > 0 && @time "extrapolating" ISOKANN.addextrapolates!(iso, extrapolates, stepsize=extrapolate)
        nx > 0 && @time "kde sampling" ISOKANN.resample_kde!(iso, nx)
        #simtime += ISOKANN.simulationtime(iso)

        @time "training" run!(iso, iter)

        #@time "plot" plot_training(iso) |> display
        if i % 10 == 0
            @time "save" ISOKANN.save("vil.iso", iso)
        end
    end
end
loop()
#=
    @time "saving" begin

        try
            global newsim
            if newsim
                mkpath(path)
                run(`ln -snf $path latest`)
                cp(@__FILE__, "$path/script.jl")
                open("$path/script.jl", "a") do io
                    sha = readchomp(`git rev-parse HEAD`)
                    write(io, "\n#git-commit: $sha")
                end
                chmod("$path/script.jl", 0o444)
                newsim = false
            end


            pdblen = ISOKANN.readchemfile(iso.data.sim.pdb) |> length
            save_reactive_path(iso, ISOKANN.getcoords(iso.data)[1:pdblen, :] |> cpu;
                out="$path/snapshot/villin_fold_$(simtime)ps.pdb", sigma, maxjump)
            #ISOKANN.savecoords("$path/data.pdb", iso)
            ISOKANN.Plots.savefig(plot_training(iso), "$path/snapshot/villin_fold_$(simtime)ps.png")

            ISOKANN.save("$path/iso.jld2", iso)

            run(`ln -sf snapshot/villin_fold_$(simtime)ps.pdb $path/path.pdb`)
            run(`ln -sf snapshot/villin_fold_$(simtime)ps.png $path/training.png`)

            println("status: $path/training.png")
        catch e
            @show e

        end
    end

end
=#
using ISOKANN
using ISOKANN.OpenMM
using Dates

## Config

pdb = "data/villin nowater.pdb"
steps = 10_000  # 20 ps
temp = 310
features = 0.5 # 0 => backbone only
iter = 1000
generations = 2
cutoff = 3000
nx = 8
extrapolates = 1 # *2
extrapolate = 0.05
nk = 4
keepedges = false
minibatch = 100
opt = ISOKANN.NesterovRegularized(1e-3, 1e-4)
layers = 4
sigma = 2
forcefields = OpenMM.FORCE_AMBER_IMPLICIT
path = "out/villin/$(now())"[1:end-4]

## Initialisation



burstlength = steps * 0.002 / 1000 # in nanoseconds
simtime_per_gen = burstlength * (nx + 2 * extrapolates) * nk # in nanoseconds

@show burstlength, simtime_per_gen, "(in ns)"

sim = OpenMMSimulation(;
    pdb, steps, forcefields, features,
    nthreads=1,
    temp,
    mmthreads="gpu")

@time "generating initial data" data = SimulationData(sim, nx, nk)

iso = Iso2(data;
    opt, minibatch,
    model=pairnet(length(sim.features); layers),
    gpu=true,
    loggers=[])

run!(iso, iter)

mkpath(path)
cp(@__FILE__, "$path/script.jl")

## Running

for i in 1:1_000 # one microsecond
    @time "running" runadaptive!(iso; generations, nx, iter, cutoff, keepedges, extrapolates)

    #time = length(iso.losses) / iter * simtime_per_gen
    #time = round(time, digits=2)
    time = ISOKANN.simulationtime(iso)
    @time "saving" begin
        save_reactive_path(iso, out="$path/villin_fold_$(time)ps.pdb"; sigma)
        ISOKANN.savecoords("$path/data.pdb", iso)
        ISOKANN.Plots.savefig(plot_training(iso), "$path/villin_fold_$(time)ps.png")
        ISOKANN.save("$path/iso.jld2", iso)
    end
end


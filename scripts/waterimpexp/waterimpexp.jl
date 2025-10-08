using ISOKANN
using Plots


VAL_NX = 1_000 # number of validation samples
VAL_STEPS = 10_000 # steps between validation samples
VAL_NK = 100 # validation burst size

NX = 10_000 # number of training samples
NITER = 5000 # number of training iterations
LAGSTEPS = 1000

REVERSE = true # also count reverse transitions
PDB = "data/systems/alanine dipeptide.pdb"

features = collect(1:22) # use pairwise dists between all 22 atoms
sims = [
    OpenMMSimulation(forcefields=OpenMM.FORCE_AMBER; features), # vacuum
    OpenMMSimulation(forcefields=OpenMM.FORCE_AMBER_IMPLICIT; features), # implicit water
    OpenMMSimulation(addwater=true, padding=1; features)] # explicit water


@time "computing validation data" begin
    valsim = OpenMMSimulation(addwater=true, padding=1; features)
    valtraj = laggedtrajectory(valsim, VAL_NX, steps=VAL_STEPS)
    valdata = SimulationData(valsim, valtraj, VAL_NK) |> gpu
end

scatter_ramachandran(valtraj)

isos = []
for (i, sim) in enumerate(sims)
    @time "generating data $i" begin
        xs = laggedtrajectory(sim, NX, steps=LAGSTEPS)
        scatter_ramachandran(xs) |> display
        savefig("water rama $i.png")
        data = SimulationData(sim, xs, 1)

        iso = Iso(data, loggers=Any[vallog], minibatch=1000, validation=valdata)

        push!(isos, iso)
        try
            ISOKANN.save("iso_$i.jld2", iso)
        catch
        end
    end
end

for (i, iso) in enumerate(isos)
    @time "training model $i" begin
        run!(iso, NITER)
        try
            ISOKANN.save("iso_$i.jld2", iso)
        catch
        end
        plot_training(iso)
        savefig("water trained $i.png")
    end
end


function deltadchi(iso1=isos[1], iso2=isos[2], coords=valtraj)
    mapreduce(hcat, eachcol(coords |> gpu)) do x
        ISOKANN.dchidx(iso2, x) - ISOKANN.dchidx(iso1, x)
    end
end

using ISOKANN.Makie.Observables
function visgraddelta()
    coords = Observable(valtraj)
    grads = Observable(deltadchi() |> cpu)
    ISOKANN.plotmol(coords, sim.pysim; grad=grads)
end

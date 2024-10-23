using ISOKANN
using Plots


VAL_NX = 1000 # number of validation samples
VAL_STEPS = 10_000 # steps between validation samples
VAL_NK = 100 # validation burst size

NX = 100_000 # number of training samples
NITER = 1000 # number of training iterations

REVERSE = true # also count reverse transitions
PDB = "data/systems/alanine dipeptide.pdb"

features = collect(1:22) # use pairwise dists between all 22 atoms
sims = [
    OpenMMSimulation(forcefields=OpenMM.FORCE_AMBER; features), # vacuum
    OpenMMSimulation(forcefields=OpenMM.FORCE_AMBER_IMPLICIT; features), # implicit water
    OpenMMSimulation(addwater=true, padding=1; features)] # explicit water


@info "computing validation data"
valsim = OpenMMSimulation(addwater=true, padding=1; features)
valtraj = laggedtrajectory(valsim, VAL_NX, steps=VAL_STEPS)
valdata = SimulationData(valsim, valtraj, VAL_NK) |> gpu

scatter_ramachandran(valtraj)

isos = []
for (i, sim) in enumerate(sims)

    @info "generating data $i"
    xs = laggedtrajectory(sim, NX)
    scatter_ramachandran(xs) |> display
    savefig("water rama $i.png")
    data = SimulationData(sim, data_from_trajectory(xs; reverse=REVERSE))

    vallog = ISOKANN.ValidationLossLogger(data=valdata)
    @info "training model $i"
    iso = Iso(data, loggers=Any[vallog], minibatch=1000)

    push!(isos, iso)

    run!(iso, NITER)
    plot_training(iso)
    savefig("water trained $i.png")
end


function deltadchi(iso1=isos[1], iso2=isos[2], coords=valtraj)
    delta = reduce(hcat, [ISOKANN.dchidx(iso2, x) - ISOKANN.dchidx(iso1, x) for x in eachcol(coords |> gpu)])
end

using ISOKANN.Makie.Observables
function visgraddelta()
    coords = Observable(valtraj)
    grads = deltadchi()
    ISOKANN.plotmol(coords, sim.pysim; grad=grads)
end

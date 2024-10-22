using ISOKANN
using Plots


VAL_NX = 10#00
VAL_STEPS = 100#00
VAL_NK = 10

NX = 100#00
NITER = 100#00

REVERSE = true # also count reverse transitions
PDB = "data/systems/alanine dipeptide.pdb"

features = collect(1:22)
sims = [
    OpenMMSimulation(forcefields=OpenMM.FORCE_AMBER; features),
    OpenMMSimulation(forcefields=OpenMM.FORCE_AMBER_IMPLICIT; features),
    OpenMMSimulation(addwater=true, padding=10; features)]



valsim = OpenMMSimulation(addwater=true, padding=10; features)
valtraj = laggedtrajectory(valsim, VAL_NX, steps=VAL_STEPS)
valdata = SimulationData(valsim, valtraj, VAL_NK) |> gpu

for (i, sim) in enumerate(sims)
    xs = laggedtrajectory(sim, NX)
    #scatter_ramachandran(xs) |> display
    data = SimulationData(sim, data_from_trajectory(xs; reverse=REVERSE))

    vallog = ISOKANN.ValidationLossLogger(data=valdata)
    iso = Iso(data, loggers=[vallog])

    run!(iso, NITER)
    plot_training(iso)
    savefig("$i.png")
end

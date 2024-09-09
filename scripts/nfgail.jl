using Revise
using ISOKANN

PDB = "data/systems/nfgail nowater.pdb"
NX_START = 10_000
NK = 2

inds = atom_indices(PDB, "!type H")
sim = OpenMMSimulation(pdb=PDB, forcefields=ISOKANN.OpenMM.FORCE_AMBER_IMPLICIT, gpu=true, features=inds, steps=1000)
#@show ISOKANN.lagtime(sim) * u"ps"

@time data = ISOKANN.trajectorydata_bursts(sim, NX_START, NK)
@show ISOKANN.simulationtime(data)

iso = Iso(data, gpu=true, loggers=[])

run!(iso, 10_000)


ndata = 50
ntrain = 2000

lastx = iso.data.coords[1][:, end]
function addtraj!(nx)
    global lastx
    @time "Generating trajectory data" iso.data = addcoords(iso.data, laggedtrajectory(iso.data.sim, nx, x0=lastx))
    lastx = iso.data.coords[1][:, end]
    return nothing
end

for i in 1:10
    addtraj!(ndata)
    @time "Generating adaptive samples" iso.data = ISOKANN.resample_kde(iso.data, iso.model, ndata)
    @time "Training the model" run!(iso, ntrain, showprogress=false)
    plot_training(iso)
    display(iso)
end

function prepare_system(; kwargs...)
    PDB = "data/systems/nfgail nowater.pdb"
    NX_START = 1_000
    NK = 1
    _prepare_system(; PDB, NX_START, NK, kwargs...)
end

function _prepare_system(; PDB, NX_START, NK,)
    inds = atom_indices(PDB, "name CA")
    sim = OpenMMSimulation(pdb=PDB, forcefields=ISOKANN.OpenMM.FORCE_AMBER_IMPLICIT, gpu=true, features=inds, steps=1000)

end
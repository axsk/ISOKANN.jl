using ISOKANN:
    ISOKANN,
    load_trajectory,
    restricted_localpdistinds,
    pdists,
    atom_indices,
    data_from_trajectory,
    lastcat,
    *

DATADIR = "/data/numerik/ag_cmd/trajectory_transfers_for_isokann/data/8EF5_500ns_pka_7.4_no_capping_310.10C"
MAXRADIUS = 0.5 # angstrom
STRIDE = 10

trajfiles = ["$DATADIR/traj.dcd" for i in 1:2]
pdbfile = "$DATADIR/struct.pdb"


molecule = load_trajectory(pdbfile)
pdist_inds = restricted_localpdistinds(molecule, MAXRADIUS, atom_indices(pdbfile, "not water and name==CA"))



datas = map(trajfiles) do trajfile
    traj = load_trajectory(trajfile, top=pdbfile, stride=STRIDE)  # for large datasets you may use the memory-mapped LazyTrajectory
    xs, ys = data_from_trajectory(traj, reverse=true)
end

xs, ys = reduce(mergedata, datas)

data = pdists(xs, pdist_inds), pdists(ys, pdist_inds)

iso = Iso(data, opt=NesterovRegularized())
run!(iso, 1000)

# saving the reactive path for multiple trajectories could work like this
# note that the above data is probably too big for this to terminate in sufficient time

# coords = ISOKANN.LazyMultiTrajectory(ISOKANN.LazyTrajectory.(trajfiles))
save_reactive_path(iso, xs, sigma=0.1, source=pdbfile)
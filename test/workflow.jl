using ISOKANN

sim = OpenMMSimulation()

data = trajectorydata_bursts(sim, 10, 1)

iso = Iso(data)

isofile = tempname() * ".iso"
ISOKANN.save(isofile, iso)
iso = ISOKANN.load(isofile)

run!(iso, 10)

runadaptive!(iso, generations=1, iter=1, kde=1)

save_reactive_path(iso, out=tempname() * ".pdb")

#xs = reactionpath_minimum(iso, xtol=1)  # taken out of test since its experimental only and takes very long
#ISOKANN.savecoords(tempname() * ".pdb", iso, xs)

using ISOKANN


function exportsorted(iso)
  xs = ISOKANN.getxs(iso.data)
  p = iso.model(xs) |> vec |> sortperm
  xs = ISOKANN.getcoords(iso.data)[1]
  ISOKANN.writechemfile("out/sorted.pdb", ISOKANN.aligntrajectory(xs[:, p] .* 10 |> cpu); source=pdb)
end

pdb = "/scratch/htc/ldonati/VGVAPG/implicit/input/initial_states/x0_1.pdb"
sim = ISOKANN.OpenMM.OpenMMSimulation(; pdb, steps=1000, forcefields=["amber14-all.xml", "implicit/obc2.xml"])

data = ISOKANN.SimulationData(sim, 100, 1, ISOKANN.flatpairdists)

iso = Iso2(data, minibatch=100) |> gpu

run!(iso, 100)

runadaptive!(iso)

exportsorted(iso)

scatter(chis(iso) |> vec |> cpu, ISOKANN.getxs(iso.data)[2416, :] |> vec |> cpu)

save_reactive_path(iso, iso.data.coords[1] .* 10 |> cpu, source=pdb)
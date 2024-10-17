using ISOKANN
using LinearAlgebra
using StatsBase

pdb = "data/chris/af2_multimer_pdbfixer copy.pdb"
ligand = "data/chris/rigid_docking_af2multimer_1.sdf"
temp = 300

sim = OpenMMSimulation(; pdb, ligand, temp);
OpenMM.savecoords("test.pdb", sim);

chainid = [a.residue.chain.index for a in sim.pysim.topology.atoms()]
m = OpenMM.masses(sim)
x = getcoords(sim) |> x -> reshape(x, 3, :)
forcemask = chainid .== 1

centerofmass = sum(x .* m', dims=2) ./ sum(m)


centerofligand = let x = x[:, forcemask], m = m[forcemask]
    sum(x .* m', dims=2) ./ sum(m)
end

force = normalize!(centerofligand .- centerofmass) |> vec
strength = 10

bias(q; kwargs...) = (force .* forcemask' .* strength) |> vec


sim = OpenMMSimulation(; pdb, ligand, temp, bias)
OpenMM.minimize!(sim)

traj = laggedtrajectory(sim, 10)
@show mean_and_std(traj.weights)
savecoords("worm.pdb", sim, traj.values)
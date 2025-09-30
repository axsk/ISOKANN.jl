using ISOKANN

# each simulation is carried out for 100 * 2fs.
# we compute 100 simulations each, for a total time of 0.02 ns per `laggedtrajectory` call
# to obtain ns/s, take (t*50)^-1 for below timings t [s]

# setting parameters for optional usage
bias(x; kwargs...) = 0
minimize = true
addwater = true
forcefields = OpenMM.FORCE_AMBER_IMPLICIT

# 22 atoms
sim = OpenMMSimulation()
sim2 = OpenMMSimulation(; bias)

@time laggedtrajectory(sim, 100);  # 0.28s
@time laggedtrajectory(sim2, 100); # 1.35s


# 2278 atoms
pdb = "data/systems/6O0K.pdb"
sim = OpenMMSimulation(; pdb, minimize)
sim2 = OpenMMSimulation(; pdb, bias, minimize)

@time laggedtrajectory(sim, 100);  #1.04s
@time laggedtrajectory(sim2, 100); #4.01s

# 52967 atoms with explicit water
sim = OpenMMSimulation(; pdb, minimize, addwater);
sim2 = OpenMMSimulation(; pdb, bias, minimize, addwater);

@time laggedtrajectory(sim, 100);  #6.18s
@time laggedtrajectory(sim2, 100);  #92.6s

# 7518 atoms
pdb = "data/openmm8ef5-processed.pdb"
sim = OpenMMSimulation(; pdb, minimize)
sim2 = OpenMMSimulation(; pdb, bias, minimize)

@time laggedtrajectory(sim, 100);  #1.41s
@time laggedtrajectory(sim2, 100);  #10.87s

# 7518 atoms with implicit water
pdb = "data/openmm8ef5-processed.pdb"
sim = OpenMMSimulation(; pdb, minimize, forcefields)
sim2 = OpenMMSimulation(; pdb, bias, minimize, forcefields)

@time laggedtrajectory(sim, 100);  #3.52s
@time laggedtrajectory(sim2, 100);  #12.89s



# all these timings were computed obtaining the systems forces only
# obtaining the energy as well (required for gaussian) adds around 10% to the force/energy computation
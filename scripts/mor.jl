using ISOKANN
using ISOKANN: OpenMM

using PyCall

try
    global openmm = @pyimport openmm
catch
end

getPos(simulation) = simulation.context.getState(getPositions=true).getPositions(asNumpy=true)

function create_simulation(topology, positions; minimize=true)
    forcefield = openmm.app.ForceField("amber14-all.xml", "tip3p.xml")
    system = forcefield.createSystem(topology, nonbondedMethod=openmm.app.PME, removeCMMotion=false)
    integrator = openmm.LangevinMiddleIntegrator(310 * openmm.unit.kelvin, 1 / openmm.unit.picoseconds, 0.001 * openmm.unit.picoseconds)
    simulation = openmm.app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    minimize && simulation.minimizeEnergy()
    return simulation
end


function load_mor()
    # the input file is from the OPM database, where I removed the other chains via openmm-setup
    # that removal can (and should) be done here as well, similar to how we remove parts of the second chain below
    sim = OpenMMSimulation(pdb="8ef5-processed.pdb").pysim

    # load pdb
    modeller = openmm.app.modeller.Modeller(sim.topology, getPos(sim))
    forcefield = openmm.app.ForceField("amber14-all.xml", "tip3p.xml")
    modeller.topology.setUnitCellDimensions((6, 6, 12))  # the add membrane routine is quite sensitive to the unit cell

    # remove the sidechain
    A = modeller.topology.chains() |> collect |> last
    modeller.delete((A.residues()|>collect)[1:52])

    # fix terminals
    modeller.delete([a for a in modeller.topology.atoms() if a.name == "H"])
    modeller.addHydrogens(forcefield)
    #openmm.app.PDBFile.writeFile(modeller.topology, modeller.positions, "fixed.pdb")

    # create system for energy minimization pre membrane
    #= we didnt even use this, right?
    positions = getPos(create_system(modeller.topology, modeller.positions, minimize=true))
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME,)
    integrator = openmm.VerletIntegrator(0.001 * openmm.unit.picoseconds)
    simulation = openmm.app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    openmm.app.PDBFile.writeFile(modeller.topology, getPos(simulation), "minimized.pdb")
    =#

    # add membrane
    modeller.addMembrane(forcefield, "POPC", membraneCenterZ=-0.1, neutralize=false, minimumPadding=0.0)
    #openmm.app.PDBFile.writeFile(modeller.topology, modeller.positions, "membraned.pdb")

    # create system for energy minimization post membrane
    simulation = create_simulation(modeller.topology, modeller.positions)

    openmm.app.PDBFile.writeFile(modeller.topology, getPos(simulation), "mor_membrane_minimized.pdb")

    sim = OpenMMSimulation(simulation, 10_000, Dict())
    centermor!(sim)
    return sim
end

function test()
    sim = load_mor()
    xs = laggedtrajectory(sim, 1)
    OpenMM.savecoords("mortrajtest.pdb", sim, xs)
end

function centermor!(sim, i=7518)
    c = reshape(coords(sim), 3, :)
    unitcell = collect(sim.pysim.topology.getUnitCellDimensions()._value)
    c = c .- sum(c[:, 1:i], dims=2) ./ i .+ unitcell ./ 2
    OpenMM.setcoords(sim, vec(c))
end


using ISOKANN
using ISOKANN.OpenMM
using LinearAlgebra
using StatsBase

pdb = "data/chris/af2_multimer_pdbfixer copy.pdb"
ligand = "data/chris/rigid_docking_af2multimer_1.sdf"
temp = 300
forcefield_kwargs = Dict("nonbondedCutoff" => 3)
simargs = (; pdb, ligand, temp, forcefield_kwargs)

sim = OpenMMSimulation(; simargs...);

chainid = [a.residue.chain.index for a in sim.pysim.topology.atoms()]
m = OpenMM.masses(sim)
x = getcoords(sim) |> x -> reshape(x, 3, :)
forcemask = chainid .== 1

centerofmass = sum(x .* m', dims=2) ./ sum(m)

centerofligand = let x = x[:, forcemask], m = m[forcemask]
    sum(x .* m', dims=2) ./ sum(m)
end

outward = normalize!(centerofligand .- centerofmass) |> vec

STRENGTH::Float64 = 1
PUSH::Float64 = 0.1

function constacc(q; F, kwargs...)
    PUSH .* vec(outward .* forcemask' .* m')
end

function adaptivebias(q; F, kwargs...)
    meanforce = mean((reshape(F, 3, :)./m')[:, forcemask], dims=2)
    d = dot(outward, meanforce)

    mflig = meanforce = mean((reshape(F, 3, :)./m')[:, .!forcemask], dims=2)
    bilig = vec(-1 .* dot(outward, mflig) .* outward .* .!forcemask' .* m')
    #if d < 0
    #    d = 0
    #end
    bias = vec(-1 .* d .* outward .* forcemask' .* m')
    (bias .+ bilig) .* STRENGTH
end

function combinedbias(q; F, kwargs...)
    adaptivebias(q; F, kwargs...) .+ constacc(q; F, kwargs...)
end

function simbiased(; bias=combinedbias, strength=1.0, push=0.3, nx=1000, saveevery=10)
    global STRENGTH = strength
    global PUSH = push
    sim = OpenMMSimulation(; simargs..., bias=combinedbias, minimize=true, step=0.002)
    traj = laggedtrajectory(sim, nx)
    path = "worms $(string(sim.bias)) s$(strength) p$(push).pdb"
    savecoords(path, sim, traj.values[:, 1:saveevery:end])
    println(path)
    return traj.values
end



function protocol()
    # obtained "pushout" trajectory by biased sampling
    xs0 = simbiased(push=5, strength=0, nx=30, saveevery=1)

    sim = OpenMMSimulation(; simargs...)

    f1 = FeaturesPairs(sim; maxdist=1, atomfilter=a -> a.residue.chain.id == "X", maxfeatures=100)
    f2 = FeaturesPairs(sim; maxdist=1, atomfilter=a -> a.name == "CA", maxfeatures=100)

    a = atoms(sim)
    ca = [i for (i, a) in enumerate(a) if a.name == "CA"]
    lig = [i for (i, a) in enumerate(a) if a.residue.chain.index == 1]

    cross = StatsBase.sample(([(i, j) for i in ca, j in lig]), 1000, replace=false)

    featurizer = FeaturesPairs(vcat(f1.pairs, f2.pairs, cross))


    prop = propagate(sim, xs0, 10) |> ISOKANN.flattenlast

    data = SimulationData(sim, prop, 3; featurizer)

    # instantiate simulationdata and iso object with 3 koopman sample
    iso = Iso(data, autoplot=3)
end
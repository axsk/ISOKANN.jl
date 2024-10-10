@time "loading packages" begin
    using ISOKANN, ISOKANN.OpenMM
    using StatsBase: sample
    using CUDA
end

@time "loading" sim = OpenMMSimulation(py="data/atefe/mor.py", steps=10_000)

@time "minimizing" OpenMM.minimize!(sim)


NFEATURES = 3000
NX = 1_000
NK = 1
NTRAIN = 30_000


#xs = laggedtrajectory(sim, 1000)
xs2 = laggedtrajectory(sim, NX, resample_velocities=false)
JLD2.save("data/atefe/laggedtraj_$(NX)_momentum.jld2", "xs", xs2)

featurizer = OpenMM.FeaturesPairs(sample(OpenMM.calpha_pairs(sim.pysim), NFEATURES, replace=false))


data = @time "sampling" trajectorydata_bursts(sim, NX, NK; featurizer)
iso = Iso(data, opt=NesterovRegularized())

ISOKANN.save("atefe.iso", iso)

@time "training" run!(iso, NTRAIN)


ISOKANN.save("atefe.iso", iso)


## scrapboard

function lastsoluteindex(sim)
    map(sim.pysim.topology.atoms()) do a
        !(a.residue.name in ["POPC", "HOH", "SOD", "CLA"])
    end |> findlast
end
@time "loading packages" begin
    using ISOKANN, ISOKANN.OpenMM
    using StatsBase: sample
    using CUDA
end

@time "loading" sim = OpenMMScript("data/atafe/fe/script.py", steps=10_000)

NFEATURES = 3000
NX = 1_000
NK = 1
NTRAIN = 30_000
TESTRUN = false

if TESTRUN
    NX = 2
    NK = 1
end

begin
    @time xs = laggedtrajectory(sim, 1_000)
    @time JLD2.save("data/atafe/traj.jld2", "xs", xs)
end

featurizer = FeaturesPairs(sample(OpenMM.calpha_pairs(sim.pysim), NFEATURES, replace=false))
data = @time "sampling" trajectorydata_bursts(sim, NX, NK; featurizer)
iso = Iso(data, opt=NesterovRegularized(), gpu=CUDA.has_cuda())

@time "training" run!(iso, 100)
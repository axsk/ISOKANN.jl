using ISOKANN

xs = ISOKANN.data_from_trajectories([JLD2.load("data/villin/villin long $i.jld2", "xs")[:, 1:800:end] for i in 1:4], reverse=true, stride=1)

featurizer = OpenMM.FeaturesPairs([(rand(1:591), rand(1:591)) for i in 1:1000])
data = SimulationData(ISOKANN.ExternalSimulation(), xs; featurizer)

transform = ISOKANN.TransformPinvHist(4, 4) |> gpu
model=pairnet(n=1000, nout=3)
opt=NesterovRegularized(1e-3, 1e-2)

iso = Iso(data; model, opt, transform)

L = chis(iso) |> cpu
R = ISOKANN.expectation(iso.model, propfeatures(iso.data)) |> cpu
kinv = L * pinv(R)
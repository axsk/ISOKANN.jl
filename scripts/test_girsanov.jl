using ISOKANN
using CUDA
using PyCall
pdbfile = "data/joram/6O0K_fixed_receptor.pdb"
ligand = "data/joram/ligand_hydrogens1.sdf"


# gpu= true am besten
sim = OpenMMSimulation(
    steps=10000,
    pdb=pdbfile,
    ligand=ligand,
    temp=25 + 272.15,
    features=0,
    minimize=false,
    gpu=CUDA.functional()
)

@pyinclude("scripts/center_of_mass.py")
lig_vec_py, lig_indexes_py, avg_force_py, avg_force_ligand_py = @pycall py"get_ligand_idx_and_distance"(sim.pysim)::PyObject
lig_indexes = Array{Int16, 1}(lig_indexes_py);
lig_vec = Matrix{Float64}(lig_vec_py);
avg_force_ligand = Array{Float64, 1}(avg_force_ligand_py);
avg_force = Array{Float64, 1}(avg_force_py);
mean_ligand = sum(avg_force_ligand)/3;

function pull_ligand(coords, mu)
    force = zeros(size(reshape(coords, 3, :)))
    force[:, lig_indexes] = force[:, lig_indexes] .+ lig_vec'*mu;
    return vec(force)
end 

g(i) = (coords) -> pull_ligand(coords, i)
const iter = 1000;
const images = 1000;
results = Array{Float64, 2}(undef, 7167, images+1)
k = 10000
sim = OpenMMSimulation(
    steps=10000,
    pdb=pdbfile,
    ligand=ligand,
    temp=25 + 272.15,
    features=0,
    minimize=false,
    gpu=CUDA.functional(); 
    F_ext=g(mean_ligand*1/k)
)
using JLD2
xs = JLD2.load("resultsNeu4.jld2")["results"][:,1:800]
println("Loaded xs")
result = [(a, b) for a in 1:2389 for b in 2280:2389]
close_ids = ISOKANN.localpdistinds(xs[:,1], 0.7)
calpha_inds_temp = ISOKANN.OpenMM.calpha_pairs(sim.pysim)
feature_idxes = vcat(result, calpha_inds_temp)
inersection = intersect(feature_idxes, close_ids)
featurizer(x) = ISOKANN.pdists(x, inersection)
model = pairnet(n=size(featurizer(xs),1))
data = ISOKANN.X(sim, xs, 1; featurizer=featurizer)
JLD2.save("ysResultNeu.jld2","data", data)
opt = ISOKANN.NesterovRegularized(1e-3, 1e-3)
iso = Iso(data; opt=opt, model=model, gpu=true, autoplot=10)
run!(iso, 200)
runadaptive!(iso, generations=100, cutoff=2000)

xyz = ISOKANN.reactionpath_minimum(iso)
ISOKANN.savecoords("output/girsanov/minimumpath6.pdb",iso, xyz)
# TODO: this is currently loaded from openmm.jl - decouple it from that

featurizer(sim::OpenMMSimulation) = featurizer(sim, get(sim.constructor, :features, nothing))

featurizer(sim, ::Nothing) =
    if natoms(sim) < 100
        FeaturesAll()
    else
        maxfeatures = 100
        @warn("No default featurizer specified. Falling back to $maxfeatures random pairs")
        FeaturesPairs(sim; maxdist=0, maxfeatures)
    end
featurizer(sim, atoms::Vector{Int}) = FeaturesAtoms(atoms)
featurizer(sim, pairs::Vector{Tuple{Int,Int}}) = FeaturesPairs(pairs)
featurizer(sim, features::Function) = features
featurizer(sim, radius::Number) = FeaturesPairs([calpha_pairs(sim.pysim); local_atom_pairs(sim.pysim, radius)] |> unique)

struct FeaturesCoords end
(f::FeaturesCoords)(coords) = coords

struct FeaturesAll end
(f::FeaturesAll)(coords) = ISOKANN.flatpairdists(coords)

""" Pairwise distances between all provided atoms """
struct FeaturesAtoms
    atominds::Vector{Int}
end
(f::FeaturesAtoms)(coords) = ISOKANN.flatpairdists(coords, f.atominds)

struct FeaturesPairs
    pairs::Vector{Tuple{Int,Int}}
end
(f::FeaturesPairs)(coords) = ISOKANN.pdists(coords, f.pairs)
Base.show(io::IO, f::FeaturesPairs) = print(io, "FeaturesPairs() with $(length(f.pairs)) features")


"""
    FeaturesPairs(pairs::Vector{Tuple{Int,Int}})
    FeaturesPairs(system; selector="all", maxdist=Inf, maxfeatures=Inf)

Creates a FeaturesPairs object from either:
- a list of index pairs (`Vector{Tuple{Int,Int}}`) passed directly.
- an `OpenMMSimulation` or PDB file path (`String`), selecting atom pairs using MDTraj selector syntax (`selector`),
  optionally filtered by `maxdist` (in nm) and limited to `maxfeatures` randomly sampled pairs.
"""
FeaturesPairs(sim::OpenMMSimulation; kwargs...) = FeaturesPairs(pdbfile(sim); kwargs...)
function FeaturesPairs(pdb::String; selector="all", maxdist=Inf, maxfeatures=Inf)
    #mdtraj = pyimport("mdtraj", "mdtraj", "conda-forge")
    mdtraj = PythonCall.pyimport("mdtraj")
    m = mdtraj.load(pdb)
    inds = m.top.select(selector) .+ 1
    if maxdist < Inf
        c = permutedims(m.xyz, (3, 2, 1))
        c = reshape(c, :, size(c, 3))
        inds = ISOKANN.restricted_localpdistinds(c, maxdist, inds)
    else
        inds = [(inds[i], inds[j]) for i in 1:length(inds) for j in i+1:length(inds)]
    end
    if length(inds) > maxfeatures
        inds = StatsBase.sample(inds, maxfeatures, replace=false) |> sort
    end
    return FeaturesPairs(inds)
end


"""
    featurepairs(d::ISOKANN.SimulationData)

returns pairs of atom indices corresponding to the pairwise distance features 
"""
function featurepairs(d::ISOKANN.SimulationData)
    if d.featurizer isa OpenMM.FeaturesPairs
        return d.featurizer.pairs
    elseif d.featurizer isa OpenMM.FeaturesAll
        return Tuple.(halfinds(OpenMM.natoms(d.sim)))
    else
        @error "featurepairs not defined for this featurizer"
    end
end

import BioStructures
struct FeaturesAngles
    struc
end

function FeaturesAngles(sim::OpenMMSimulation)
    return FeaturesAngles(read(sim.constructor.pdb, BioStructures.PDBFormat))
end

function (f::FeaturesAngles)(coords::AbstractVector)
    coords = reshape(coords, 3, :)
    atoms = collectatoms(f.struc)
    for (a, c) in zip(atoms, eachcol(coords))
        coords!(a, c)
    end
    filter(!isnan, vcat(phiangles(f.struc), psiangles(f.struc)))
end

function (f::FeaturesAngles)(coords)
    mapslices(f, coords, dims=1)
end


function remove_H_H2O_NACL(atom)
    !(
        atom.element.symbol == "H" ||
        atom.residue.name in ["HOH", "NA", "CL"]
    )
end

atoms(sim::OpenMMSimulation) = collect(sim.pysim.topology.atoms())

function local_atom_pairs(pysim::PyObject, radius; atomfilter=remove_H_H2O_NACL)
    xs = reshape(coords(pysim), 3, :)
    atoms = filter(atomfilter, pysim.topology.atoms() |> collect)
    inds = map(atom -> atom.index + 1, atoms)

    pairs = Tuple{Int,Int}[]
    for i in 1:length(inds)
        for j in i+1:length(inds)
            if norm(xs[:, i] - xs[:, j]) <= radius
                push!(pairs, (inds[i], inds[j]))
            end
        end
    end
    return pairs
end

function calpha_pairs(pysim::PyObject)
    local_atom_pairs(pysim, Inf; atomfilter=x -> x.name == "CA")
end

calpha_inds(sim::OpenMMSimulation) = calpha_inds(sim.pysim)
function calpha_inds(pysim::PyObject)
    map(filter(x -> string(x.name) == "CA", pysim.topology.atoms() |> collect)) do atom
        atom.index + 1
    end
end
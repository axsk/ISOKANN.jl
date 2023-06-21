## Interface to the Molly.System

using Molly: Molly, System
using Unitful
using StaticArrays

export PDB_5XER, PDB_6MRR, PDB_ACEMD

dim(sys::System) = length(sys.atoms) * 3
defaultmodel(sys::System) = pairnet(sys::System)

" Return the rotation handle indices for the standardform of a molecule"
function rotationhandles(sys::Molly.System)
    # TODO: fix this ugly if-elseif recognition
    if dim(sys) == 22 # PDB_ACEMD
        return (2,11,19)
    elseif dim(sys) == 1234 # some other poweriteration
        return (1,2,3)
    else
        error("No rotation handles known for this molecule")
    end
end

# These were supposed to be for convenience but created problems downstream

#Base.show(io::IO, ::MIME"text/plain", x::System) = print(io, "Molly.System ($(dim(x))D)")
#Base.show(io::IO, ::MIME"text/plain", x::Type{<:System}) = print(io, "Molly.System (TYPE)")


# Molly.System is not mutable. We provide a constructor creating a new instance with updated fields.
# (This was not possible programmatically since the System type is not inferrable from the fields alone.)
#=
" Constructor to update some fields of the (immutable) Molly.System "
function Molly.System(s::System; kwargs...)
    System(;
        atoms = s.atoms,
        atoms_data = s.atoms_data,
        pairwise_inters = s.pairwise_inters,
        specific_inter_lists = s.specific_inter_lists,
        general_inters = s.general_inters,
        constraints = s.constraints,
        coords = s.coords,
        velocities = s.velocities,
        boundary = s.boundary,
        neighbor_finder = s.neighbor_finder,
        loggers = s.loggers,
        force_units = s.force_units,
        energy_units = s.energy_units,
        k = s.k,
        kwargs...)
end
=#

""" extract the unitful SVector coords from `sys` and return as a normal vector """
getcoords(sys::System) = getcoords(sys.coords)
function getcoords(coords::AbstractArray)
    x0 = Molly.ustrip_vec(coords)
    x0 = reduce(vcat, x0)
    return x0 :: AbstractVector
end

""" convert normal vector of coords to SVectors of unitful system coords """
function vec_to_coords(x::AbstractArray, sys::System)
    xx = reshape(x, 3, :)
    coord = sys.coords[1]
    u = unit(coord[1])
    coords = typeof(coord)[c * u for c in eachcol(xx)]
    return coords
end

""" set the system to the given coordinates """
# TODO: the centercoords shift does not belong here, fix constant
# Note: Actually the Langevin integrator removes centercoords of mass motion, so we should be fine
setcoords(sys::System, coords) = setcoords(sys, vec_to_coords(centercoords(coords) .+ 1.36, sys))
setcoords(sys::System, coords::Array{<:SVector{3}}) = System(sys;
    coords=coords,
    velocities = copy(sys.velocities),
    #    neighbor_finder = deepcopy(sys.neighbor_finder),
)


## Save to pdb files

function savecoords(sys::System, coords::AbstractVector, path; append = false)
    append || rm(path, force=true)
    writer = Molly.StructureWriter(0, path)
    sys = setcoords(sys, coords)
    Molly.append_model!(writer, sys)
end

function savecoords(sys::System, data::AbstractMatrix, path; append = false)
    append || rm(path, force=true)
    for x in eachcol(data)
        savecoords(sys, x, path, append=true)
    end
end


## loaders for the molecules in /data

const molly_data_dir = joinpath(dirname(pathof(Molly)), "..", "data")

function PDB_6MRR()
    ff = MolecularForceField(
        joinpath(molly_data_dir, "force_fields", "ff99SBildn.xml"),
        joinpath(molly_data_dir, "force_fields", "tip3p_standard.xml"),
        joinpath(molly_data_dir, "force_fields", "his.xml"),
    )
    sys = System(joinpath(molly_data_dir, "6mrr_nowater.pdb"), ff)
    return sys
end

function PDB_5XER()
    sys = System(
        joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"),
        joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top");
    )
    #temp = 298.0u"K"
    ## attempt at removing water. not working, some internal data is referring to all atoms
    #nosol = map(sys.atoms_data) do a a.res_name != "SOL" end
    #sys.atoms = sys.atoms[nosol]
    #sys.atoms_data = sys.atoms_data[nosol]
    return sys
end

""" Peptide Dialanine """
function PDB_ACEMD(;kwargs...)
    ff = Molly.MolecularForceField(joinpath(molly_data_dir, "force_fields", "ff99SBildn.xml"))
    sys = System(joinpath(@__DIR__, "..", "data", "alanine-dipeptide-nowater av.pdb"), ff,
        rename_terminal_res = false, # this is important,
        #boundary = CubicBoundary(Inf*u"nm", Inf*u"nm", Inf*u"nm")  breaking neighbor search
        ; kwargs...
    )
    return sys
end

""" Peptide Dialanine """
function PDB_1UAO(;kwargs...)
    ff = Molly.MolecularForceField(joinpath(molly_data_dir, "force_fields", "ff99SBildn.xml"))
    sys = System(joinpath(@__DIR__, "..", "data", "1uao av.pdb"), ff,
        rename_terminal_res = false, # this is important,
        #boundary = CubicBoundary(Inf*u"nm", Inf*u"nm", Inf*u"nm")  breaking neighbor search
        ; kwargs...
    )
    return sys
end

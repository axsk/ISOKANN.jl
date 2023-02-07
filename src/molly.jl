using Molly
using Unitful
using StochasticDiffEq
using Accessors

export PDB_5XER, PDB_6MRR

function PDB_6MRR()
    data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
    ff = OpenMMForceField(
        joinpath(data_dir, "force_fields", "ff99SBildn.xml"),
        joinpath(data_dir, "force_fields", "tip3p_standard.xml"),
        joinpath(data_dir, "force_fields", "his.xml"),
    )

    sys = System(
        joinpath(data_dir, "6mrr_nowater.pdb"),
        ff;
        loggers=(
            energy=TotalEnergyLogger(10),
            writer=StructureWriter(10, "traj_6mrr_1ps.pdb", ["HOH"]),
        ),
    )
    return sys
end

function PDB_5XER()
    sys = System(
        joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"),
        joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top");
        loggers=(
            temp=TemperatureLogger(10),
            writer=StructureWriter(10, "traj_5XER_1ps.pdb"),
        ),
    )
    temp = 298.0u"K"
    return sys
end


export SDEProblem

function StochasticDiffEq.SDEProblem(sys::System, T=1e-3; dt=1e-5, alg=SROCK2(), kwargs...)

    kb = 8.314462618e−3 * u"kJ/mol/K" ./ 1u"Na"

    gamma = 10 / u"ps"
    temp = 298.0u"K"
    M = repeat(masses(sys), inner=3)

    sigma = sqrt(2*kb/gamma*temp) .* M.^ (-1/2)
    sigma = ustrip.(u"nm/ps^(1/2)", sigma)


    function driftf(x,p,t)
        (;neighborlist) = p
        sys = setcoords(sys, x)

        f = forces(sys, neighborlist, n_threads=1)

        drift = f ./ masses(sys) ./ gamma
        drift = reduce(vcat, drift) ./ 1u"Na"
        drift = ustrip.(u"nm/ps", drift)
        return drift
    end

    noise(x,p,t) = sigma
    x0 = coords_to_vec(sys)

    neighborlist = find_neighbors(sys)


    SDEProblem(driftf, noise, x0, T, (;neighborlist), alg=alg, dt=dt, kwargs...)
end

dim(sys::System) = length(sys.atoms) * 3
defaultmodel(dynamics::System, layers=[100,100]) = fluxnet([dim(dynamics); layers; 1])
function randx0(sys::System, nx; sigma=0.01)
    x0 = ustrip_vec(sys.coords)
    x0 = reduce(vcat, x0)
    hcat([x0 .* (1 .+ randn(length(x0))*sigma) for i in 1:nx]...)
end

## Utility functions

function vec_to_coords(x::AbstractArray, sys)
    xx = reshape(x, 3, :)
    coord = sys.coords[1]
    coords = typeof(coord)[c * unit(coord[1]) for c in eachcol(xx)]
    return coords
end

function coords_to_vec(sys)
    X = reduce(hcat, sys.coords)
    return map(x->x.val, X) |> vec
end

setcoords(sys, coords) = setcoords(sys, vec_to_coords(coords, sys))
setcoords(sys, coords::Array{<:SVector{3}}) = Molly.System(
    atoms = sys.atoms,
    atoms_data = sys.atoms_data,
    pairwise_inters=sys.pairwise_inters,
    specific_inter_lists = sys.specific_inter_lists,
    general_inters = sys.general_inters,
    constraints = sys.constraints,
    coords=coords,
    velocities = sys.velocities,
    boundary=sys.boundary,
    neighbor_finder = sys.neighbor_finder,
    loggers = sys.loggers,
    force_units = sys.force_units,
    energy_units = sys.energy_units,
    k=sys.k
)


cmask(sys) = map(x->x.element, sys.atoms_data) .== "C"

## Interface to the Molly.System and implementation of the Overdamped Langevin Equation
## in MollySDE

using Molly
using Unitful
using StochasticDiffEq
import StochasticDiffEq: SDEProblem, solve
export PDB_5XER, PDB_6MRR, PDB_ACEMD, SDEProblem, MollySDE, solve, exportdata


" Type composing the Molly.System with its integration parameters "
Base.@kwdef mutable struct MollySDE{S,A}
    sys::S
    temp::Float64 = 298. # 298 K = 25 °C
    gamma::Float64 = 10.
    dt::Float64 = 2e-6 * gamma  # in ps
    T::Float64 = 2e-3 * gamma # in ps   # tuned as to take ~.1 sec computation time
    alg::A = EM()
    nthreads::Int = 1  # number of threads for the force computations
end


dim(ms::MollySDE) = dim(ms.sys)
getcoords(ms::MollySDE) = getcoords(ms.sys)
defaultmodel(ms::MollySDE) = defaultmodel(ms.sys)

dim(sys::System) = length(sys.atoms) * 3

SDEProblem(m::MollySDE) = SDEProblem(m.sys, m.T;
       dt=m.dt, alg=m.alg, gamma=m.gamma, temp=m.temp, n_threads=m.nthreads)

function SDEProblem(sys::System, T=1e-3; dt, alg=SROCK2(),
    n_threads=1, gamma=10, temp=298, kwargs...)

    kb = 8.314462618e−3 * u"kJ/mol/K" ./ 1u"Na"

    gamma = gamma / u"ps"
    temp = temp * u"K"
    M = repeat(masses(sys), inner=3)

    sigma = sqrt(2*kb/gamma*temp) .* M.^ (-1/2)
    sigma = ustrip.(u"nm/ps^(1/2)", sigma)

    function driftf(x,p,t)
        (;neighborlist) = p
        sys = setcoords(sys, x)

        f = forces(sys, neighborlist, n_threads=n_threads)

        drift = f ./ masses(sys) ./ gamma
        drift = reduce(vcat, drift) ./ 1u"Na"
        drift = ustrip.(u"nm/ps", drift)
        return drift
    end

    noise(x,p,t) = sigma
    x0 = getcoords(sys)
    neighborlist = find_neighbors(sys)  # TODO: is 1 setting enough?

    SDEProblem(driftf, noise, x0, T, (;neighborlist), alg=alg, dt=dt, kwargs...)
end



#Base.show(io::IO, ::MIME"text/plain", x::System) = print(io, "Molly.System ($(dim(x))D)")
#Base.show(io::IO, ::MIME"text/plain", x::Type{<:System}) = print(io, "Molly.System (TYPE)")



# delegation for ::MollySDE

solve(m::MollySDE; kwargs...) = solve(SDEProblem(m); kwargs...)

savecoords(m::MollySDE, sol; kwargs...) = exportdata(m, reduce(hcat, sol.u); kwargs...)

function savecoords(m::MollySDE, traj::AbstractArray;
    path="out/$(m.gamma) $(m.T) $(m.dt).pdb", kwargs...)
    savecoords(m.sys, traj, path; kwargs...)
end

"""
propagting the dynamics `ms` starting at the given `x0` each `ny` times.
return the corresponding endpoints in an array of shape [dim, nx, ny]
"""
function propagate(ms::MollySDE, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, nx, ny)
    sde = SDEProblem(ms)
    @floop for i in 1:nx, j in 1:ny
        #  the tspan fixes u0 assignment (https://github.com/SciML/DiffEqBase.jl/issues/883)
        sol = solve(sde; u0=x0[:,i], tspan=(0,sde.tspan[2]))
        ys[:, i, j] = sol[end]
    end
    # TODO: at which point do we want to center the data?
    return ys
end


#defaultmodel(sys::System, layers=round.(Int, dim(sys).^[2/3, 1/3])) = fluxnet([dim(sys); layers; 1])

# Neural Network model for mol

import Flux

defaultmodel(sys::System) = pairnet(sys::System)

" Neural Network model for molecules, using pairwise distances as first layer "
function pairnet(sys)
    n = div(dim(sys), 3)
   #= l = [
        x->SliceMap.slicemap(x, dims=1) do col
            vec(pairwise(SqEuclidean(), reshape(col,3,:), dims=2))
        end,
        Flux.Dense(n*n, n, Flux.sigmoid),
        Flux.Dense(n, 1, Flux.sigmoid)
        ]
        =#
    nn = Flux.Chain(
        mythreadpairdists,
        Flux.Dense(n*n, n, Flux.sigmoid),
        Flux.Dense(n, 1, Flux.sigmoid))
    return nn
end


## utils for interaction with the ::Molly.System

""" extract the unitful SVector coords from `sys` and return as a normal vector """
function getcoords(sys::System)
    x0 = ustrip_vec(sys.coords)
    x0 = reduce(vcat, x0)
    return x0 :: AbstractVector
end

""" convert normal vector of coords to SVectors of unitful system coords """
function vec_to_coords(x::AbstractArray, sys::System)
    xx = reshape(x, 3, :)
    coord = sys.coords[1]
    coords = typeof(coord)[c * unit(coord[1]) for c in eachcol(xx)]
    return coords
end

""" set the system to the given coordinates """
setcoords(sys::System, coords) = setcoords(sys, vec_to_coords(coords, sys))
setcoords(sys::System, coords::Array{<:SVector{3}}) = Molly.System(
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

""" move the ::System to the GPU, mirroring behavior of Flux.gpu """
gpu(sys::System) = Molly.System(
    atoms = cu(sys.atoms),
    atoms_data = cu(sys.atoms_data),
    pairwise_inters=sys.pairwise_inters,
    specific_inter_lists = sys.specific_inter_lists,
    general_inters = sys.general_inters,
    constraints = sys.constraints,
    coords=cu(sys.coords),
    velocities = cu(sys.velocities),
    boundary=sys.boundary,
    neighbor_finder = sys.neighbor_finder,
    loggers = sys.loggers,
    force_units = sys.force_units,
    energy_units = sys.energy_units,
    k=sys.k
)


## loaders for the molecules in /data

const molly_data_dir = joinpath(dirname(pathof(Molly)), "..", "data")

function PDB_6MRR()
    ff = OpenMMForceField(
        joinpath(molly_data_dir, "force_fields", "ff99SBildn.xml"),
        joinpath(molly_data_dir, "force_fields", "tip3p_standard.xml"),
        joinpath(molly_data_dir, "force_fields", "his.xml"),
    )
    sys = System(joinpath(molly_data_dir, "6mrr_nowater.pdb"), ff)
    return MollySDE(;sys)
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
    return MollySDE(;sys)
end

""" Peptide Dialanine """
function PDB_ACEMD(;kwargs...)
    ff = OpenMMForceField(joinpath(molly_data_dir, "force_fields", "ff99SBildn.xml"))
    sys = System(joinpath("data", "alanine-dipeptide-nowater av.pdb"), ff,
        rename_terminal_res = false  # this is important
    )
    return MollySDE(;sys,kwargs...)
end

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



### dihedrals / Ramachandrann

using LinearAlgebra: cross

# https://naturegeorge.github.io/blog/2022/07/dihedral/
function dihedral(coord0, coord1, coord2, coord3)
    b = coord2 - coord1
    u = cross(b, coord1 - coord0)
    w = cross(b, coord2 - coord3)
    return atan(cross(u, w)'*b, u'*w * norm(b))
end

dihedral(x::AbstractMatrix) = @views dihedral(x[:,1], x[:,2], x[:,3], x[:,4])



function psi(x::AbstractVector)  # dihedral of the oxygens
    x = reshape(x, 3, :)
    @views ISOKANN.dihedral(x[:, [7,9,15,17]])
end

function phi(x::AbstractVector)
    x = reshape(x, 3, :)
    @views ISOKANN.dihedral(x[:, [5,7,9,15]])
end

phi(x::Matrix) = mapslices(phi, x, dims=1) |> vec
psi(x::Matrix) = mapslices(psi, x, dims=1) |> vec


### standardform

using LinearAlgebra
using StatsBase: mean

"""
center any given states by shifting their individual 3d mean to the origin
"""
function center(xs)
    mapslices(xs, dims=1) do x
        coords = reshape(x, 3, :)
        coords .-= mean(coords, dims=2)
        vec(coords)
    end
end

function rotationmatrix(e1, e2)
    e1 ./= norm(e1)
    e2 .-= dot(e1, e2) * e1
    e2 ./= norm(e2)
    e3 = cross(e1, e2)
    A = hcat(e1, e2, e3)
    R = A / I  #  A * R = I
end

" rotate vec representation of ACEMD "
function rotatevec(vec)
    x = reshape(vec, 3, :)
    e1 = x[:, 19] .- x[:, 2]
    e2 = x[:, 11] .- x[:, 2]
    R = rotationmatrix(e1, e2)
    return R' * x
end

standardform(x::AbstractArray) = mapslices(x, dims=1) do col
    x = rotatevec(col)
    x .-= mean(x, dims=2)
    vec(x)
end

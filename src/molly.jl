using Molly
using Unitful
using StochasticDiffEq
using Accessors
using Plots

import StochasticDiffEq: SDEProblem, solve

export PDB_5XER, PDB_6MRR, PDB_ACEMD, SDEProblem, MollySDE, solve, exportdata,
    inspecttrajectory, scatter_ramachandran

@with_kw mutable struct MollySDE{S,A}
    sys::S
    temp::Float64 = 298. # 298 K = 25 °C
    gamma::Float64 = 10.
    dt::Float64 = 2e-6 * gamma  # in ps
    T::Float64 = 2e-3 * gamma # in ps   # tuned as to take ~.1 sec computation time
    alg::A = EM()
    nthreads::Int = 1  # number of threads for the force computations
end

## loaders for the molecules in /data

function PDB_6MRR()
    data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
    ff = OpenMMForceField(
        joinpath(data_dir, "force_fields", "ff99SBildn.xml"),
        joinpath(data_dir, "force_fields", "tip3p_standard.xml"),
        joinpath(data_dir, "force_fields", "his.xml"),
    )
    sys = System(joinpath(data_dir, "6mrr_nowater.pdb"), ff)
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
    data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
    ff = OpenMMForceField(joinpath(data_dir, "force_fields", "ff99SBildn.xml"))
    sys = System(joinpath("data", "alanine-dipeptide-nowater av.pdb"), ff,
        rename_terminal_res = false  # this is important
    )
    return MollySDE(;sys,kwargs...)
end


solve(m::MollySDE; kwargs...) = solve(SDEProblem(m); kwargs...)


function exportdata(m::MollySDE, sol; kwargs...)
    exportdata(m, reduce(hcat, sol.u); kwargs...)
end

function exportdata(m::MollySDE, traj::AbstractArray;
    path="out/$(m.gamma) $(m.T) $(m.dt).pdb", kwargs...)
    exportdata(m.sys, path, traj; kwargs...)
end

"""
generate data for koopman evaluation by propagting the dynamics
starting at the given `x0` (or sampling `nx` random starting points)
returns `xs` the starting points and `ys` the corresponding endpoints
"""
function generatedata(ms::MollySDE, nkoop::Integer, x0::AbstractMatrix)

    dim, nx = size(x0)
    ys = zeros(dim, nx, nkoop)
    sde = SDEProblem(ms)

    @floop for i in 1:nx, j in 1:nkoop
        ys[:, i, j] = solve(sde; u0=x0[:,i])[end]
    end

    return center(x0), center(ys)
end

"""
sample `nx` initial starting points by propagating from the systems coordinate
"""
function bootstrapx0(ms::MollySDE, nx)
    x0 = reshape(getcoords(ms), :, 1)
    _, ys = generatedata(ms, nx, x0)
    reshape(ys, :, nx)
end

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

dim(ms::MollySDE) = dim(ms.sys)
getcoords(ms::MollySDE) = getcoords(ms.sys)
defaultmodel(ms::MollySDE) = defaultmodel(ms.sys)


### my ::System interface

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
    x0 = coords_to_vec(sys)
    neighborlist = find_neighbors(sys)  # TODO: is 1 setting enough?

    SDEProblem(driftf, noise, x0, T, (;neighborlist), alg=alg, dt=dt, kwargs...)
end

dim(sys::System) = length(sys.atoms) * 3
#defaultmodel(sys::System, layers=round.(Int, dim(sys).^[2/3, 1/3])) = fluxnet([dim(sys); layers; 1])

import Flux
using Distances
import SliceMap # provides slicemap == mapslices but with zygote gradients
import ChainRulesCore

defaultmodel(sys::System) = pairnet(sys::System)



function threadpairdists(x)
    ChainRulesCore.@ignore_derivatives begin
        d, s... = size(x)
        x = reshape(x, d, :)
        pd = SliceMap.ThreadMapCols(pairwisevec, x)
        pd = reshape(pd, :, s...)
        return pd
    end
end

function mythreadpairdists(x)
    d, s... = size(x)
    xx = reshape(x, d, :)
    cols = size(xx, 2)
    n = div(d, 3)
    out = similar(x, n, n, cols)
    @views Threads.@threads for i in 1:cols
        c = reshape(xx[:, i], 3, :)
        pairdistkernel(out[:, :, i], c)
    end
    reshape(out, n*n, s...)
end

function pairdistkernel(out, x)
    n, k = size(out)
    @views for i in 1:n, j in 1:k
        out[k,j] = out[j,k] = ((x[1,i]-x[1,j])^2 + (x[2,i]-x[2,j])^2 + (x[3,i]-x[3,j])^2)
    end
end

ChainRulesCore.@non_differentiable threadpairdists(x)

function pairnet(sys::System)
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
        x->threadpairdists(ChainRulesCore.ignore_derivatives(x)),
        Flux.Dense(n*n, n, Flux.sigmoid),
        Flux.Dense(n, 1, Flux.sigmoid))
    return nn
end


pairdists(x) = SliceMap.slicemap(pairwisevec, x, dims=1)

pairwisevec(col::AbstractVector) = vec(pairwise(SqEuclidean(), reshape(col,3,:), dims=2))


function getcoords(sys::System)
    x0 = ustrip_vec(sys.coords)
    x0 = reduce(vcat, x0)
    return x0
end



## Utility functions

function vec_to_coords(x::AbstractArray, sys::System)
    xx = reshape(x, 3, :)
    coord = sys.coords[1]
    coords = typeof(coord)[c * unit(coord[1]) for c in eachcol(xx)]
    return coords
end

function coords_to_vec(sys::System)
    X = reduce(hcat, sys.coords)
    return map(x->x.val, X) |> vec
end

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

# move the system to CUDA
cu(sys::System) = Molly.System(
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

function savecoords(sys::System, filepath, coords::AbstractVector)
    writer = Molly.StructureWriter(0, filepath)
    sys = setcoords(sys, coords)
    Molly.append_model!(writer, sys)
end

function exportdata(sys::System, filepath, data::AbstractMatrix; append = false)
    append || rm(filepath, force=true)
    write
    for x in eachcol(data)
        savecoords(sys, filepath, x)
    end
end

function inspecttrajectory(sys)
    @time sol = solve(SDEProblem(sys))
    x = reduce(hcat, sol.u)
    scatter_ramachandran(x) |> display
    exportdata(sys, x, path="out/inspect.pdb")
    return x
end


## Plotting

atommask(sys::System, atom="C") = map(x->x.element, sys.atoms_data) .== atom

function visualize!(sys::System, coords::AbstractVector{<:Number}; subinds = :, atomtype="C")
    coords = (reshape(coords, 3, :)')[atommask(sys, atomtype),:]
    coords = coords[subinds,:]
    scatter!(coords[:,1], coords[:,2], label=atomtype) |> display
end

function visualize(sys::System)
    plot()
    for a in ["C", "N", "O",]
        visualize!(sys, coords_to_vec(sys), atomtype=a)
    end
    plot!()
end

using LinearAlgebra: cross

# https://naturegeorge.github.io/blog/2022/07/dihedral/
function dihedral(coord0, coord1, coord2, coord3)
    b = coord2 - coord1
    u = cross(b, coord1 - coord0)
    w = cross(b, coord2 - coord3)
    return atan(cross(u, w)'*b, u'*w * norm(b))
end

dihedral(x::AbstractMatrix) = @views dihedral(x[:,1], x[:,2], x[:,3], x[:,4])

function scatter_ramachandran(x::Matrix, model=nothing)
    z = nothing
    !isnothing(model) && (z = model(x) |> vec)
    ph = phi(x)
    ps = psi(x)
    scatter(ph, ps, markersize=3, markerstrokewidth=0, markeralpha=0.5, marker_z=z, xlabel="\\phi", ylabel="\\psi", title="Ramachandran",
    xlims = [-pi, pi], ylims=[-pi, pi])
end

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

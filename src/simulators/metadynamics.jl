using LinearAlgebra: norm
using Interpolations

"""
    MetadynamicsSimulation(sim, rc, mdstate, dt, height, sigma)
    MetadynamicsSimulation(iso; height=1f0, sigma=0.1f0, dt=600f0)

Well-tempered metadynamics bias that can be used as a force in a Langevin simulation.

The bias potential is a sum of Gaussians deposited at visited reaction coordinate (RC)
values. When called as `md(x)`, it returns the negative gradient of the (well-tempered)
bias with respect to the configuration `x`, suitable for use as an additive force.

# Arguments
- `sim`: underlying `IsoSimulation` (provides temperature and propagation)
- `rc`: reaction coordinate function `x -> z`, mapping configuration to RC space
- `mdstate`: accumulated Gaussian centers — one of `MetadynamicsState`,
  `MetadynamicsStateMatrix` (GPU-optimized), or `MetadynamicsStateGridded`
- `height`: Gaussian height
- `sigma`: Gaussian width in RC space
- `dt`: well-tempered offset temperature (ΔT); `Inf` for classic (untempered) metadynamics

The convenience constructor builds `rc` from `chicoords(iso)` and initializes
`mdstate` from the current chi values of the `Iso`.

# See also
`deposit!`, `trajectory`, `wt_free_energy`, `plot_profile`
"""
mutable struct MetadynamicsSimulation{S, R, MS, T<:Real}
    sim::S
    rc::R
    mdstate::MS
    dt::T     ## welltempered offset (Inf = classic metadynamics)
    height::T ## Gaussian height
    sigma::T  ## Gaussian width
end

@functor MetadynamicsSimulation (mdstate,)

function (md::MetadynamicsSimulation)(x::AbstractVector; kwargs...)
    grad, = Zygote.gradient(x -> rescale_welltempered(bias_potential(md.mdstate, md.rc(x); md.height, md.sigma), md.dt), x)
    return -grad
end

# Convenience constructor: use chicoords of an Iso as the reaction coordinate
function MetadynamicsSimulation(iso::Iso; height=1f0, sigma=0.1f0, dt=600f0)
    isokann_rc(x) = project_onto_simplex_hyperplane(chicoords(iso, x))
    mdstate = MetadynamicsStateMatrix(chis(iso))
    return MetadynamicsSimulation(iso.data.sim, isokann_rc, mdstate, dt, height, sigma)
end

deposit!(md::MetadynamicsSimulation, z) = deposit!(md.mdstate, z)

trajectory(md::MetadynamicsSimulation; kwargs...) =
    OpenMM.langevin_girsanov!(md.sim; bias=md, sigmascaled=false, kwargs...)

rescale_welltempered(U, dt=1600f0) = isfinite(dt) ? dt * log(1 + U / dt) : U
project_onto_simplex_hyperplane(x) = length(x) > 1 ? x .- ((sum(x)-1) / length(x)) : x

# iso is passed separately since this is iso-specific visualization
function plot_profile(md::MetadynamicsSimulation, iso;)
    md = cpu(md)
    chilast = chicoords(iso, coords(iso)[:, end:end]) |> cpu
    if ISOKANN.outputdim(iso.model) == 1
        zs = reshape(collect(range(0f0, 1f0, length=100)), 1, :)
        F = wt_free_energy(md, zs)
        plot(vec(zs), F, xlabel="z", ylabel="V(z)", title="Metadynamics Free Energy Profile")
        Flast = wt_free_energy(md, chilast)
        scatter!(vec(chilast), Flast)
    else
        chivals = chis(iso) |> cpu
        F = wt_free_energy(md, chivals)
        scatter(eachrow(chivals)..., marker_z=F, camera=(135, 35), title="Metadynamics Free Energy Profile")
    end
end

function wt_free_energy(md::MetadynamicsSimulation, zs::AbstractMatrix)
    T = OpenMM.temp(md.sim)
    V = [bias_potential(md.mdstate, z; md.height, md.sigma) for z in eachcol(zs)]
    F = -(T + md.dt) / T * V
    return F
end


### State implementations (optional variants)

"""
    MetadynamicsState{T, V}

Vector-of-vectors storage for Gaussian centers.

**Performance:** CPU: OK | GPU: slow (~1000x) | Add center: O(1) push
**Best for:** CPU-only, frequent center additions
**Dynamics:** Exact

See also: `MetadynamicsStateMatrix`, `MetadynamicsStateGridded`
"""
struct MetadynamicsState{T<:Number,V<:AbstractVector{T}}
    centers::Vector{V}   # visited RC values s_i
end

MetadynamicsState(centers::AbstractMatrix{T}) where T =
    MetadynamicsState([view(centers, :, i) for i in 1:size(centers, 2)])

deposit!(s::MetadynamicsState, z::AbstractVector) = push!(s.centers, z)
deposit!(s::MetadynamicsState, z::AbstractMatrix) = append!(s.centers, eachcol(z))

# Bias potential in RC space: V(s) = sum_i h * exp(-|s-s_i|^2 / 2σ^2)
function bias_potential(mdstate::MetadynamicsState{T}, z::AbstractVector; height, sigma) where T
    isempty(mdstate.centers) && return zero(T)
    @assert length(z) == length(mdstate.centers[1])
    wN = sum(mdstate.centers) do sᵢ
        height * exp(-sum(abs2, z .- sᵢ) / (2 * sigma^2))
    end
    return wN
end

"""
    MetadynamicsStateMatrix{T}

Matrix storage (nrc × n_centers) for Gaussian centers — GPU-optimized.

**Performance:** CPU: fast | GPU: very fast | Add center: O(n) hcat
**Best for:** GPU production runs
**Dynamics:** Exact

See also: `MetadynamicsState`, `MetadynamicsStateGridded`
"""
mutable struct MetadynamicsStateMatrix{T<:Number, M<:AbstractMatrix{T}}
    centers::M
end

deposit!(s::MetadynamicsStateMatrix, z) = (s.centers = hcat(s.centers, z))

function bias_potential(mdstate::MetadynamicsStateMatrix{T}, z::AbstractVector; height, sigma) where T
    size(mdstate.centers, 2) == 0 && return zero(T)
    @assert length(z) == size(mdstate.centers, 1)
    dists_sq = sum(abs2, z .- mdstate.centers, dims=1)
    wN = sum(height .* exp.(-dists_sq ./ (2 * sigma^2)))
    return wN
end

"""
    MetadynamicsStateGridded{ITP}

Grid-based approximation of bias potential with cubic spline interpolation.

**Performance:** CPU: very fast | GPU: N/A | Add center: unsupported
**Best for:** 1D–2D rapid exploration (low-dim only)
**Dynamics:** Approximate (spline interpolation)

See also: `MetadynamicsState`, `MetadynamicsStateMatrix`
"""
struct MetadynamicsStateGridded{ITP}
    interpolant::ITP  # cubic spline interpolator (encodes grid structure and data)
end

function MetadynamicsStateGridded(xs::AbstractMatrix{T}, ranges::NTuple{N,<:AbstractRange}; height, sigma) where {T,N}
    grid = zeros(T, length.(ranges)...)
    for idx in CartesianIndices(grid)
        z = [ranges[i][idx[i]] for i in 1:N]
        for x in eachcol(xs)
            grid[idx] += height * exp(-sum(abs2, x .- z) / (2 * sigma^2))
        end
    end
    itp = cubic_spline_interpolation(ranges, grid)
    return MetadynamicsStateGridded(itp)
end

deposit!(::MetadynamicsStateGridded, _) = error("MetadynamicsStateGridded does not support online deposition. Use MetadynamicsState or MetadynamicsStateMatrix, or reconstruct the gridded state from accumulated centers.")

function bias_potential(mdstate::MetadynamicsStateGridded, z::AbstractVector)
    return mdstate.interpolant(z...)
end

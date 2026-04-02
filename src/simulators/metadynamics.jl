using LinearAlgebra: norm
using Interpolations

global HEIGHT = 1f0   # default Gaussian height
global SIGMA = 0.1f0  # default Gaussian width
global dT::Float32 = 600       # well-tempered offset (Inf = classic metadynamics)

const DT=600f0
# well tempered metadynamics reweighting
rescale_welltempered(U, dt::Float32=DT) = isfinite(dt) ? dt * log(1 + U / dt) : U

### MetadynamicsSimulator (main API)

mutable struct MetadynamicsSimulator
    sim
    force
    mdstate
end

@functor MetadynamicsSimulator (mdstate,)

function MetadynamicsSimulator(sim, rc::Function; height=HEIGHT, sigma=SIGMA, centers=Vector{Float32}[], mdstatetype=MetadynamicsStateMatrix)
    mdstate = mdstatetype(centers; height, sigma)
    function mdforce(x::AbstractVector; kwargs...)
        grad, = Zygote.gradient(x -> bias_potential(mdstate, rc(x)), x)
        return -grad
    end
    return MetadynamicsSimulator(sim, mdforce, mdstate)
end

# Convenience constructor: use chicoords of an Iso as the reaction coordinate
function MetadynamicsSimulator(iso::Iso; height=HEIGHT, sigma=SIGMA)
    isokann_rc(x) = project_onto_simplex_hyperplane(chicoords(iso, x))
    centers = chis(iso)
    return MetadynamicsSimulator(iso.data.sim, isokann_rc; height, sigma, centers)
end

project_onto_simplex_hyperplane(x) = length(x) > 1 ? x .- ((sum(x)-1) / length(x)) : x

trajectory(mdsim::MetadynamicsSimulator; kwargs...) =
    OpenMM.langevin_girsanov!(mdsim.sim; bias=mdsim.force, sigmascaled=false, kwargs...)

# TODO: OpenMM-style online metadynamics via step!(mdsim, nsteps)
# Add `frequency::Int` field to MetadynamicsSimulator (deposit every N steps).
# `langevin_girsanov!` is a tight inner loop with no mid-trajectory hook, so
# deposition requires calling it in chunks of `frequency` steps:
#
#   function step!(mdsim::MetadynamicsSimulator, nsteps; x0)
#       x = x0
#       samples = []
#       for _ in 1:(nsteps ÷ mdsim.frequency)
#           t = OpenMM.langevin_girsanov!(mdsim.sim; bias=mdsim.force,
#                   steps=mdsim.frequency, saveevery=mdsim.frequency,
#                   sigmascaled=false, x0=x)
#           x = t.values[:, end]
#           push!(mdsim.mdstate.centers, Vector{Float32}(mdsim.rc(x)))
#           push!(samples, t)
#       end
#       return reduce(hcat, samples)  # concatenated WeightedSamples
#   end
#
# Note: Girsanov weights reset at chunk boundaries (logw=0 per snippet),
# consistent with resample_velocities=true. Check whether this is acceptable
# for addcoords! before implementing.

# iso is passed separately since this is iso-specific visualization
function plot_profile(mdsim::MetadynamicsSimulator, iso; zs=-0.1:0.01:1.1)
    T = OpenMM.temp(mdsim.sim)
    global dT
    mdstate = cpu(mdsim.mdstate)   # plotting is always CPU
    if ISOKANN.outputdim(iso.model) == 1
        # plot over uniform grid
        V = [bias_potential(mdstate, [z]) for z in zs]
        F = -(T + dT) / T * V
        plot(zs, F, xlabel="z", ylabel="V(z)", title="Metadynamics Free Energy Profile")
        chilast = chicoords(iso, coords(iso)[:, end]) |> cpu
        Flast = -(T + dT) / T * bias_potential(mdstate, chilast)
        scatter!(chilast, [Flast])
    else
        # scatter over data points
        chi_vals = cpu(chis(iso))
        V = [bias_potential(mdstate, x) for x in eachcol(chi_vals)]
        F = -(T + dT) / T * V
        scatter(eachrow(chi_vals)..., marker_z=F, camera=(135, 35))
    end
end

function adaptive_metadynamics(iso; deposit=OpenMM.steps(iso.data.sim), height=HEIGHT, sigma=SIGMA, x0=coords(iso)[:, end])
    md = MetadynamicsSimulator(iso; height, sigma)
    @time "Girsanov trajectory" t = trajectory(md; x0, saveevery=deposit)
    @assert norm(t.values[:, end]) < 100
    xnew = values(t)
    addcoords!(iso, xnew)
    return (; t, md, xnew)
end

function run_metadynamics!(iso; generations=100, height=HEIGHT, sigma=SIGMA, iter=100, plots=[])
    for _ in 1:generations
        @time adaptive_metadynamics(iso; height, sigma)
        @time run!(iso, iter)
        if plots != false
            l = @layout [[a; b] c{0.3w}]
            global p1 = scatter_ramachandran(iso)
            scatter!(p1, [ISOKANN.phi(coords(iso)[:, end])], [ISOKANN.psi(coords(iso)[:, end])])
            global p2 = plot_training(iso)
            global p3 = plot_profile(MetadynamicsSimulator(iso; height, sigma), iso)
            p = plot(p1, p3, p2, layout=l, size=(800, 800))
            display(p)
            push!(plots, p)
        end
    end
    return (; iso, plots)
end

function run_kde_dash!(iso; generations=1, plots=[], kwargs...)
    for _ in 1:generations
        ISOKANN.run_kde!(iso; generations=1, kwargs...)
        global p1 = scatter_ramachandran(iso)
        global p2 = plot_training(iso)
        p = plot(p1, p2, layout=(1, 2), size=(800, 800))
        push!(plots, p)
    end
    return plots
end

function run_both!(iso; generations=100, samples_kde=1, iter=100, plots=[])
    for _ in 1:generations
        run_kde!(iso; generations=1, kde=samples_kde, iter)
        run_metadynamics!(iso; generations=1, iter, plots)
    end
end

function reactivepath_save(iso; kwargs...)
    i = hash(kwargs)
    @show i
    ix = save_reactive_path(iso, sigma=0.03, minjump=0.02, maxjump=0.1, out="experiments/260311 metadyn adp/path$i.pdb"; kwargs...)
    OpenMM.potential(iso.data.sim, coords(iso)[:, ix]) |> vec |> plot
    savefig("experiments/260311 metadyn adp/energy$i.png")
    scatter_ramachandran(coords(iso)[:, ix], x -> chicoords(iso, x))
    savefig("experiments/260311 metadyn adp/rama$i.png")
end

function makeanim(ps, filename)
    a = Animation()
    for p in ps
        frame(a, p)
    end
    mp4(a, filename)
end

function benchmark_metadynamics(n=10000, d=3)
    centers = randn(Float32, d, n)
    z = rand(d)
    ms = [
        #MetadynamicsState(centers; height=HEIGHT, sigma=SIGMA),
        MetadynamicsStateMatrix(centers; height=HEIGHT, sigma=SIGMA),
        #MetadynamicsStateGridded(centers, tuple(repeat([range(0f0, 1f0, length=10)], d)...); height=HEIGHT, sigma=SIGMA)
    ]
    for m in ms
        @show typeof(m)
        @show @time bias_potential(m, z)
        m = gpu(m)
        z = gpu(z)
        @show @time bias_potential(m, z)
    end
end

### State implementations (optional variants)

"""
    MetadynamicsState{T, V}

Vector-of-vectors storage for Gaussian centers.

**Performance:** CPU: OK | GPU: slow (~1000x) | Add center: supported (push to vector)
**Best for:** CPU-only, frequent center changes
**Dynamics:** Exact

See also: `MetadynamicsStateMatrix`, `MetadynamicsStateGridded`
"""
struct MetadynamicsState{T<:Number,V<:AbstractVector{T}}
    centers::Vector{V}   # visited RC values s_i
    height::T       # h
    sigma::T        # σ
end

MetadynamicsState(centers::AbstractMatrix{T}; height=HEIGHT, sigma=SIGMA) where T =
    MetadynamicsState([view(centers, :, i) for i in 1:size(centers, 2)], T(height), T(sigma))


# Bias potential in RC space: V(s) = sum_i h * exp(-|s-s_i|^2 / 2σ^2)
function bias_potential(mdstate::MetadynamicsState{T}, z::AbstractVector) where T
    isempty(mdstate.centers) && return zero(T)
    @assert length(z) == length(mdstate.centers[1])
    wN = sum(mdstate.centers) do sᵢ
        mdstate.height * exp(-sum(abs2, z .- sᵢ) / (2 * mdstate.sigma^2))
    end
    return rescale_welltempered(wN)
end

"""
    MetadynamicsStateMatrix{T}

Matrix storage (nrc × n_centers) for Gaussian centers — GPU-optimized.

**Performance:** CPU: fast | GPU: very fast | Add center: unsupported (requires resizing matrix)
**Best for:** GPU production runs
**Dynamics:** Exact

See also: `MetadynamicsState`, `MetadynamicsStateGridded`
"""
struct MetadynamicsStateMatrix{T<:Number, M<:AbstractMatrix{T}}
    centers::M
    height::T
    sigma::T
end

function MetadynamicsStateMatrix(centers::AbstractMatrix{T}; height=HEIGHT, sigma=SIGMA) where T
    MetadynamicsStateMatrix(centers, T(height), T(sigma))
end

function bias_potential(mdstate::MetadynamicsStateMatrix{T}, z::AbstractVector) where T
    size(mdstate.centers, 2) == 0 && return zero(T)
    @assert length(z) == size(mdstate.centers, 1)
    dists_sq = sum(abs2, z .- mdstate.centers, dims=1)
    wN = sum(mdstate.height .* exp.(-dists_sq ./ (2 * mdstate.sigma^2)))
    return rescale_welltempered(wN)
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

function MetadynamicsStateGridded(xs::AbstractMatrix{T}, ranges::NTuple{N,<:AbstractRange}; height=HEIGHT, sigma=SIGMA) where {T,N}
    grid = zeros(T, length.(ranges)...)  # Create grid of correct size
    for idx in CartesianIndices(grid)
        z = [ranges[i][idx[i]] for i in 1:N]  # Get RC coords at grid point
        for x in eachcol(xs)
            # any(x .- z  .>= 3sigma) && continue # Only update nearby grid points for efficiency
            grid[idx] += height * exp(-sum(abs2, x .- z) / (2 * sigma^2))
        end
    end
    itp = cubic_spline_interpolation(ranges, grid)
    return MetadynamicsStateGridded(itp)
end

function bias_potential(mdstate::MetadynamicsStateGridded, z::AbstractVector)
    return rescale_welltempered(mdstate.interpolant(z...))
end
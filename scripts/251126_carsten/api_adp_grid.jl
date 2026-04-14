"""
Grid-based zero-variance committor on alanine dipeptide Ramachandran space.
Stores J (value function) on a 20x20 grid over (phi, psi).
Computes control u from interpolation gradients: u = -σ ∇J.
Uses OpenMM simulation + Girsanov importance sampling + moving average updates.
"""

using Plots
using Interpolations

using ISOKANN: coords, phi, psi, force, scatter_ramachandran
using ISOKANN.OpenMM: OpenMMSimulation, laggedtrajectory, langevin_girsanov!, stepsize
using LinearAlgebra: norm, dot, I
using ProgressMeter: @showprogress
using Zygote: gradient  
using StatsBase: mean
using JLD2: @save

abstract type Grid end


function (@main)(; nx=100_000, ngrid=20, niter=10)
    sim = OpenMMSimulation() # ADP by default
    xs = laggedtrajectory(sim, nx)
    grid = RamachandranGrid(n=ngrid)
    J0 = J0_from_trajectory(grid, xs) # committor estimation through tranfer matrix and linear system
    Js = api(J0, sim, niter; xs)

    @save "api_adp_grid.jld2" xs Js
    p = visualize_q(Js[end])
    savefig("api_adp_grid.png")
    display(p)

    return (; xs, Js)
end

test() = main(nx=1000, ngrid=10, niter=1)

### Grid-based control and simulation
function control_from_grid(grid::Grid, x::AbstractVector)
    """Compute control u = ∇J via autodiff through torsion angles."""
    dJ_dx = gradient(x) do x
        interpolate_J(grid, phi(x), psi(x))
    end
    return only(dJ_dx)
end

function create_bias_function(grid::Grid, dt::Real; max_force_norm::Real=1000.0)
    """Create bias function for langevin_girsanov! that stores u² per step."""
    u2_array = Float64[]
    function bias(q; t=0, sigma, F=nothing)
        dJ_dx = control_from_grid(grid, q)
        u_control = -sigma .* dJ_dx

        u_norm = norm(u_control)
        if u_norm > max_force_norm
            print("💥")
            u_control .*= max_force_norm / u_norm
        end
        
        u2_inc = dot(u_control, u_control) * dt
        push!(u2_array, u2_inc)

        return u_control
    end
    return bias, u2_array
end

function simulate(grid::Grid, sim::OpenMMSimulation, x0::AbstractVector;
                            steps::Int=100_000, max_force_norm::Real=1000.0, n=1)
    """Simulate trajectory with grid-based control, scan for A/B hits, return the integral of the value function expectation."""
    dt = stepsize(sim)
    bias, u2_array = create_bias_function(grid, dt; max_force_norm)

    should_stop(;q, kwargs...) = classify(q) in (:A, :B)

    xs = langevin_girsanov!(sim; saveevery=1, sigmascaled=true, steps, bias, x0, should_stop, showprogress=false) |> values
    scatter_ramachandran(xs[:, 1:10:end]) |> display

    xt = xs[:, end]
    class = classify(xt)
    u2 = sum(u2_array)

    q0, q1 = 0.01, 0.99 # TODO

    if class == :A
        v = u2 / 2 - log(q0)
    elseif class == :B
        v = u2 / 2 - log(q1)
    else
        # dynamic programming approach
        println("⚠ didnt terminate in A or B after $steps steps:")
        v = u2 / 2 - interpolate_J(grid, phi(xt), psi(xt))
    end

    q = exp(-v)
    println("Simulated $(size(xs,2)) steps, class: $class, q: $(round(q; digits=2)), u²: $(round(u2; digits=2))")

    return v
end

function api_step(grid::Grid, sim::OpenMMSimulation; alpha=1, n_mc=10, xs, kwargs...)
    bins = binned_samples(grid, xs)
    Jk = copy(grid.J)
    @showprogress for ij in CartesianIndices(bins)
        isempty(bins[ij]) && continue # no update
        x0s = rand(bins[ij], n_mc)
        E = mean(simulate(grid, sim, x; kwargs...) for x in x0s)
        Jk[ij] = (1 - alpha) * Jk[ij] + alpha * E
    end
    return similar(grid, Jk)
end

function binned_samples(grid, xs)
    bins = [typeof(xs[:, 1])[] for i in 1:size(grid, 1), j in 1:size(grid, 2)]
    for x in eachcol(xs)
        i, j = find_nearest_grid_point(grid, phi(x), psi(x))
        push!(bins[i, j], x)
    end
    return bins
end

function api(J0::Grid, sim::OpenMMSimulation, n; xs, kwargs...)
    Js = [J0]
    times = []
    for i in 1:n
        t = @elapsed push!(Js, api_step(Js[end], sim; xs, kwargs...))
        push!(times, t)
    end
    @show times
    return Js
end

function J0_from_trajectory(grid::Grid, xs; eps=1e-8, kwargs...)
    T = transfermatrix(xs, grid)
    q = committor(grid, T) .+ eps
    #return q
    J = similar(grid, -log.(q))
    return J
end

function visualize_q(grid::Grid)
    heatmap(exp.(-grid.J)', xlabel="phi", ylabel="psi", title="Committor ")
end


## Committor code

boxA = [-3 -2.6; 2.5 3]
boxB = [-1.5 -1; 0 1]
boxTrans = [-1.5 -1; 2 3]  # transition region

torsion(x::AbstractVector) = (phi(x), psi(x))
torsion(x::AbstractArray) = mapslices(torsion, x, dims=1)
inbox(x, box) = all(box[:, 1] .< x .< box[:, 2])

function classify(x::AbstractVector)
    torsion_angles = torsion(x)
    inbox(torsion_angles, boxA) && return :A
    inbox(torsion_angles, boxB) && return :B
    #inbox(torsion_angles, boxTrans) && return :trans
    return :interior
end

function transfermatrix(xs, grid=RamachandranGrid())
    n = length(grid.phi_knots) * length(grid.psi_knots)
    C = zeros(n, n) .+ 1e-8
    for k in 1:size(xs, 2) - 1
        x = xs[:, k]
        y = xs[:, k+1]
        i = linindex(grid, torsion(x))
        j = linindex(grid, torsion(y))
        C[i, j] += 1
    end
    T = C ./ sum(C, dims=2)
    return T
end

function committor(grid::Grid, T)
    A = T - I
    b = zeros(length(grid.itp))
    for (i,x) in enumerate(knots(grid.itp))
        if inbox(x, boxA)
            A[i, :] .= 0
            A[i, i] = 1
            b[i] = 0
        elseif inbox(x, boxB)
            A[i, :] .= 0
            A[i, i] = 1
            b[i] = 1
        end
    end
    q = A \ b
    q = reshape(q, size(grid.J))
    return q
end


## Grid utility

mutable struct RamachandranGrid <: Grid
    phi_knots::AbstractRange{Float64}
    psi_knots::AbstractRange{Float64}
    J::Matrix{Float64}
    itp::Any
end

function RamachandranGrid(;
    phi_range::Tuple{Float64,Float64}=(-pi * 1, pi * 1),
    psi_range::Tuple{Float64,Float64}=(-pi * 1, pi * 1),
    n=20,
    n_phi::Int=n, n_psi::Int=n,
    phi_knots=range(phi_range[1], phi_range[2], length=n_phi),
    psi_knots=range(psi_range[1], psi_range[2], length=n_psi),
    J=zeros(Float64, n_phi, n_psi),
    itp=build_interpolation(J, phi_knots, psi_knots))
    RamachandranGrid(phi_knots, psi_knots, J, itp)
end

function build_interpolation(J::Matrix{Float64}, phi_knots, psi_knots)
    """Build cubic spline interpolation with periodic boundary conditions."""
    itp_base = interpolate(J, BSpline(Cubic(Periodic(OnGrid()))))
    return scale(itp_base, phi_knots, psi_knots)
end

RamachandranGrid(J) = RamachandranGrid(; n_phi=size(J, 1), n_psi=size(J, 2), J=J)  # default constructor for loading from file

Base.size(grid::RamachandranGrid, args...) = size(grid.J, args...)
Base.similar(g::RamachandranGrid, J::Matrix) = RamachandranGrid(
    g.phi_knots, g.psi_knots, J, build_interpolation(J, g.phi_knots, g.psi_knots)
)

interpolate_J(grid::RamachandranGrid, phi_deg::Real, psi_deg::Real) =
    grid.itp(phi_deg, psi_deg)

function find_nearest_grid_point(grid::RamachandranGrid, phi_deg::Real, psi_deg::Real)
    i = argmin(abs.(grid.phi_knots .- phi_deg))
    j = argmin(abs.(grid.psi_knots .- psi_deg))
    return i, j
end

function linindex(grid::RamachandranGrid, x)
    i = argmin(abs.(grid.phi_knots .- x[1]))
    j = argmin(abs.(grid.psi_knots .- x[2]))
    return i + (j - 1) * length(grid.psi_knots)
end


## graveyard

#=
function interpolate_J_grad(grid::RamachandranGrid, phi_deg::Real, psi_deg::Real)
    grad = Interpolations.gradient(grid.itp, phi_deg, psi_deg)
    return (dJ_dphi=grad[1], dJ_dpsi=grad[2])
end

function update_grid_point!(grid::RamachandranGrid, phi_deg::Real, psi_deg::Real, J_target::Real; α::Float64=0.1)
    i, j = find_nearest_grid_point(grid, phi_deg, psi_deg)
    J_old = grid.J[i, j]
    grid.J[i, j] += α * (J_target - J_old)
    grid.itp = build_interpolation(grid.J, grid.phi_knots, grid.psi_knots)
    return J_target - J_old
end

function update_grid!(grid::RamachandranGrid, J_new::Matrix{<:Real}; α::Float64=0.1)
    """Update entire J matrix with moving average and rebuild interpolation."""
    @assert size(J_new) == size(grid.J) "J_new must match grid dimensions"
    grid.J .+= α .* (J_new .- grid.J)
    grid.itp = build_interpolation(grid.J, grid.phi_knots, grid.psi_knots)
end


initialize_grid(sim::OpenMMSimulation; n_grid=20, steps=1000) = initialize_grid(laggedtrajectory(sim, steps, resample_velocities=false); n_grid)
function initialize_grid(xs; n_grid=20)
    """Initialize J on grid from distance-based committor estimate."""
    grid = RamachandranGrid(n_phi=n_grid, n_psi=n_grid)
    grid.J .= -log(0.5)
    for x in eachcol(xs)
        i, j = find_nearest_grid_point(grid, phi(x), psi(x))
        q = committor_from_distance(x)
        grid.J[i, j] = -log(clamp(q, 1e-6, 1.0 - 1e-6))
    end
    grid.itp = build_interpolation(grid.J, grid.phi_knots, grid.psi_knots)
    return grid
end


function distance_to_box(ta::Union{AbstractVector,Tuple}, box)
    """Euclidean distance from torsion angles to box (0 if inside)."""
    inbox(ta, box) && return 0.0
    dx = max(box[1, 1] - ta[1], ta[1] - box[1, 2], 0.0)
    dy = max(box[2, 1] - ta[2], ta[2] - box[2, 2], 0.0)
    return sqrt(dx^2 + dy^2)
end

function committor_from_distance(x::AbstractVector)
    """Initialize committor value based on distance to A and B."""
    ta = torsion(x)
    dist_A = distance_to_box(ta, boxA)
    dist_B = distance_to_box(ta, boxB)
    if dist_A == 0.0
        return 0.01
    elseif dist_B == 0.0
        return 0.99
    else
        total = dist_A + dist_B
        return 0.01 * (dist_B / total) + 0.99 * (dist_A / total)
    end
end

=#
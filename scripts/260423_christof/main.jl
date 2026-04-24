using NPZ
using ISOKANN
using LinearAlgebra
using Interpolations
using Zygote
using ProgressMeter
using JLD2

### --- grid utilities (pure, cheap) ---

GRID = range(-pi, pi, 40 + 1)[1:end-1]

phipsi(x) = (ISOKANN.phi(x), ISOKANN.psi(x))

function gridindex(angle::Number)
    n = length(GRID)
    Δ = 2π / n
    mod(floor(Int, (angle - GRID[1] + Δ / 2) / Δ), n) + 1
end

function boxindex(x::Tuple) # tuple of angles
    @assert length(x) == 2
    n = length(GRID)
    inds = LinearIndices((n, n))
    inds[gridindex(x[1]), gridindex(x[2])]
end

boxindex(x::AbstractArray) = boxindex(phipsi(x)) # vector of coordinates

function neighborhood(φ, ψ, r=5)  # indices of boxes in a neighborhood of (φ, ψ)
    n = length(GRID)
    inds = LinearIndices((n, n))
    i0 = gridindex(φ)
    j0 = gridindex(ψ)
    [inds[mod(i0 + di - 1, n) + 1, mod(j0 + dj - 1, n) + 1] for di in -r:r, dj in -r:r] |> vec
end

interpolate_rama_cubic(X) = extrapolate(
    scale(interpolate(X, BSpline(Cubic(Periodic(OnCell())))), GRID, GRID),
    Periodic(),
)

interpolate_rama_linear(X) = extrapolate(
    scale(interpolate(X, BSpline(Linear())), GRID, GRID),
    Periodic(),
)

function rama_heat(x)  # plotting
    X = reshape(x, length(GRID), length(GRID))
    heatmap(GRID, GRID, X'; aspect_ratio=1, xlabel="phi", ylabel="psi",
        xlims=(-π, π), ylims=(-π, π), framestyle=:box)
end

### --- experiment steps ---

""" Load metadynamics samples. Returns x0 of shape (66, nsamples). """
load_x0(path="AMORE/X0_metad.npy") = npzread(path)'

""" Build an OpenMM simulation (5 ps lag time at 450 K). """
build_sim(; T = 5, steps=round(Int, T/0.002), temp=450) = OpenMMSimulation(; steps, temp)

""" Bin the starting points by their Ramachandran box. """
function bin_samples(x0)
    bins = [[] for _ in 1:length(GRID)^2]
    for x in eachcol(x0)
        push!(bins[boxindex(x)], x)
    end
    return bins
end

""" Starting in each box, compute nk τ-propagated samples.
Returns xt of shape (66, nk, nboxes). """
function compute_xt(sim, x0bins; nk=100, save_path="xt$i.jld2")
    xt = zeros(66, nk, length(x0bins))
    @showprogress for (i, bin) in enumerate(x0bins)
        isempty(bin) && (@warn("No starting point in bin $i"); continue)
        for k in 1:nk
            xt[:, k, i] = trajectory(sim; x0=rand(bin), saveevery=OpenMM.steps(sim))[:, end]
        end
    end
    save_path !== nothing && jldsave(save_path; xt)
    return xt
end

""" Count-based transfer matrix from (66, nk, nboxes) samples. """
function transfermatrix(xt)
    n = length(GRID)
    C = zeros(n^2, n^2)
    nkoop, nsamples = size(xt, 2), size(xt, 3)
    for i in 1:nsamples, k in 1:nkoop
        j = boxindex(xt[:, k, i])
        C[i, j] += 1
    end
    C ./ sum(C, dims=2)
end

""" Indicator vector over boxes for the target set around (φ, ψ). """
function build_indB(phi, psi; r=5)
    set = neighborhood(phi, psi, r)
    ind = falses(length(GRID)^2)
    ind[set] .= true
    return ind, set
end

""" Monte-Carlo estimate of P[X_T ∈ B | X_0 ∈ starting bin]. """
function estimate_pb_mc(sim, x0s, setB; n=100, Tmax=100.0)
    manysteps = round(Int, Tmax / OpenMM.stepsize(sim))
    countB = 0
    xts = []
    @showprogress for _ in 1:n
        x0 = rand(x0s)
        xt = trajectory(sim, manysteps; x0, saveevery=manysteps)[:, end]
        push!(xts, xt)
        boxindex(xt) in setB && (countB += 1)
    end
    return countB / n, xts
end

""" Stack of log(PB + ε) interpolants, PB_itp[k] ↔ remaining time (20−k+1)·τ. """
function build_logPB_stack(P, indB; horizon=20, ε_rel=1e-3, interp=interpolate_rama_linear)
    PB_k = [P^(horizon - k + 1) * indB for k in 1:horizon]
    ε = ε_rel * maximum(maximum.(PB_k))
    [interp(log.(reshape(p, length(GRID), length(GRID)) .+ ε)) for p in PB_k]
end

""" Build the time-dependent Girsanov bias u(x, t) = σ · ∇log PB(x, t). """
function build_bias(sim, logPB_itp)
    nk = length(logPB_itp)
    τ = OpenMM.stepsize(sim) * OpenMM.steps(sim)
    sigma_ODL = OpenMM.constants(sim, true).sigma
    function bias(x; t, kwargs...)
        k = clamp(floor(Int, t / τ) + 1, 1, nk)
        dlogp = Zygote.gradient(x) do x
            φ, ψ = phipsi(x)
            logPB_itp[k](φ, ψ)
        end |> only
        sigma_ODL .* dlogp
    end
end

""" Run one guided trajectory; returns (weight, endpoint). """
function guided_traj(sim, x0s, bias, nk)
    x0 = rand(x0s)
    steps = OpenMM.steps(sim) * nk
    ws = ISOKANN.OpenMM.langevin_girsanov!(sim; x0, bias, steps, sigmascaled=true)
    return ws.weights[end], ws.values[:, end]
end

""" Girsanov IS estimate of P[X_T ∈ B]. Returns (self-normalized, unnormalized). """
function estimate_pb_girsanov(sim, x0s, bias, setB; n=500)
    nk = length(setB)  # placeholder, caller passes horizon via bias's closure
    error("pass the horizon explicitly")
end

function estimate_pb_girsanov(sim, x0s, bias, setB, horizon; n=500)
    acc = 0.0
    den = 0.0
    hit_w = Float64[]
    @showprogress for _ in 1:n
        w, x = guided_traj(sim, x0s, bias, horizon)
        den += w
        if boxindex(x) in setB
            acc += w
            push!(hit_w, w)
        end
    end
    return (self_normalized=acc / den, unnormalized=acc / n, hits=hit_w)
end

### --- experiment state (mutable cache) ---

# Dependency graph (rebuild downstream after upstream changes):
#   sim ──────────────────────────┬─► bias ──► pb_g samples
#   x0 ─► x0bins ─► xt ─► P ──────┤
#   phiB/psiB/r ─► indB/setB ─────┴─► logPB_itp
#   phi0/psi0 ─► indx0 ─► x0s
#
# Each step!_ function writes one field. No auto-cascade; re-run downstream yourself.

Base.@kwdef mutable struct Experiment
    # heavy cache
    sim = nothing
    x0 = nothing
    x0bins = nothing
    xt = nothing
    P = nothing
    # target set B
    phiB::Float64 = deg2rad(-108 + 180)
    psiB::Float64 = deg2rad(157.5 - 180)
    r::Int = 5
    indB = nothing
    setB = nothing
    # starting point
    phi0::Float64 = deg2rad(103.5 - 180)
    psi0::Float64 = deg2rad(148.5 - 180)
    indx0::Union{Nothing,Int} = nothing
    x0s = nothing
    # bias stack
    horizon::Int = 20
    ε_rel::Float64 = 1e-3
    logPB_itp = nothing
    bias = nothing
    # results
    pb_transfer::Union{Nothing,Float64} = nothing
    pb_mc::Vector{Bool} = Bool[]                            # hit/miss per MC sample
    pb_g::Vector{Tuple{Float64,Bool}} = Tuple{Float64,Bool}[]  # (weight, hit)
end

function step_sim!(e::Experiment; kwargs...)
    e.sim = build_sim(; kwargs...)
    e
end

function step_x0!(e::Experiment; path="AMORE/X0_metad.npy")
    e.x0 = load_x0(path)
    e.x0bins = bin_samples(e.x0)
    e
end

function step_xt!(e::Experiment; nk=100, save_path="xt.jld2", reuse_if_exists=true)
    if reuse_if_exists && save_path !== nothing && isfile(save_path)
        e.xt = load(save_path, "xt")
        @info "loaded xt from $save_path" size=size(e.xt)
    else
        e.xt = compute_xt(e.sim, e.x0bins; nk, save_path)
    end
    e
end

function step_P!(e::Experiment)
    e.P = transfermatrix(e.xt)
    @info "top 3 eigenvalues" vals=reverse(eigen(e.P).values)[1:3]
    e
end

function step_target!(e::Experiment; phiB=e.phiB, psiB=e.psiB, r=e.r)
    e.phiB, e.psiB, e.r = phiB, psiB, r
    e.indB, e.setB = build_indB(phiB, psiB; r)
    e
end

function step_start!(e::Experiment; phi0=e.phi0, psi0=e.psi0)
    e.phi0, e.psi0 = phi0, psi0
    e.indx0 = boxindex((phi0, psi0))
    e.x0s = e.x0bins[e.indx0]
    e
end

function step_logPB!(e::Experiment; horizon=e.horizon, ε_rel=e.ε_rel)
    e.horizon, e.ε_rel = horizon, ε_rel
    e.logPB_itp = build_logPB_stack(e.P, e.indB; horizon, ε_rel)
    e
end

function step_bias!(e::Experiment)
    e.bias = build_bias(e.sim, e.logPB_itp)
    e
end

function step_pb_transfer!(e::Experiment)
    e.pb_transfer = (e.P^e.horizon * e.indB)[e.indx0]
    @info "transfer-matrix estimate" e.pb_transfer
    e
end

""" Append n Monte-Carlo samples to e.pb_mc. """
function step_mc!(e::Experiment; n=100, Tmax=100.0)
    manysteps = round(Int, Tmax / OpenMM.stepsize(e.sim))
    @showprogress for _ in 1:n
        x0 = rand(e.x0s)
        x = trajectory(e.sim, manysteps; x0, saveevery=manysteps)[:, end]
        push!(e.pb_mc, boxindex(x) in e.setB)
    end
    e
end

""" Append n Girsanov IS samples to e.pb_g. """
function step_girsanov!(e::Experiment; n=500)
    @showprogress for _ in 1:n
        w, x = guided_traj(e.sim, e.x0s, e.bias, e.horizon)
        push!(e.pb_g, (w, boxindex(x) in e.setB))
    end
    e
end

### --- reporters ---

pb_mc_estimate(e) = isempty(e.pb_mc) ? NaN : mean(e.pb_mc)

function pb_g_estimate(e; normalized=true)
    isempty(e.pb_g) && return NaN
    num = sum(w for (w, hit) in e.pb_g if hit; init=0.0)
    den = normalized ? sum(w for (w, _) in e.pb_g; init=0.0) : length(e.pb_g)
    num / den
end

function summary(e::Experiment)
    (; pb_transfer=e.pb_transfer,
       pb_mc=pb_mc_estimate(e), n_mc=length(e.pb_mc),
       pb_g_selfnorm=pb_g_estimate(e; normalized=true),
       pb_g_unnorm=pb_g_estimate(e; normalized=false),
       n_g=length(e.pb_g))
end

### --- persistence ---

""" Save the heavy cache (xt, P) + result accumulators. sim is not serialized. """
function save_experiment(e::Experiment, path)
    jldsave(path;
        x0=e.x0, xt=e.xt, P=e.P,
        phiB=e.phiB, psiB=e.psiB, r=e.r,
        phi0=e.phi0, psi0=e.psi0,
        horizon=e.horizon, ε_rel=e.ε_rel,
        pb_transfer=e.pb_transfer, pb_mc=e.pb_mc, pb_g=e.pb_g)
end

function load_experiment(path)
    d = load(path)
    e = Experiment(;
        x0=d["x0"], xt=d["xt"], P=d["P"],
        phiB=d["phiB"], psiB=d["psiB"], r=d["r"],
        phi0=d["phi0"], psi0=d["psi0"],
        horizon=d["horizon"], ε_rel=d["ε_rel"],
        pb_transfer=d["pb_transfer"], pb_mc=d["pb_mc"], pb_g=d["pb_g"],
    )
    e.x0bins = bin_samples(e.x0)
    e
end

### --- full pipeline (first run) ---

function experiment(; nk=100, n_mc=100, n_girsanov=500, Tmax_mc=100.0, kwargs...)
    e = Experiment(; kwargs...)
    step_sim!(e)
    step_x0!(e)
    step_xt!(e; nk)
    step_P!(e)
    step_target!(e)
    step_start!(e)
    step_pb_transfer!(e)
    step_logPB!(e)
    step_bias!(e)
    step_mc!(e; n=n_mc, Tmax=Tmax_mc)
    step_girsanov!(e; n=n_girsanov)
    @info "summary" summary(e)...
    e
end

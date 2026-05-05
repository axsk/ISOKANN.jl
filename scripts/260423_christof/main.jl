using NPZ
using ISOKANN
using LinearAlgebra
using Interpolations
using Zygote
using ProgressMeter
using JLD2
using Plots
using ImageFiltering

### --- grid utilities (pure, cheap) ---

# GRID: 41 bin edges from -π to π; CENTERS: 40 bin centers (used for interpolation).
const N_GRID = 40
GRID = range(-π, π, N_GRID + 1)
const Δ_GRID = step(GRID)
const CENTERS = range(-π + Δ_GRID/2, π - Δ_GRID/2, N_GRID)

phipsi(x) = (ISOKANN.phi(x), ISOKANN.psi(x))

gridindex(angle::Number) = mod(floor(Int, (angle - GRID[1]) / Δ_GRID), N_GRID) + 1

function boxindex(x::Tuple) # tuple of angles
    @assert length(x) == 2
    n = N_GRID
    inds = LinearIndices((n, n))
    inds[gridindex(x[1]), gridindex(x[2])]
end

boxindex(x::AbstractArray) = boxindex(phipsi(x)) # vector of coordinates

interpolate_rama_cubic(X) = extrapolate(
    scale(interpolate(X, BSpline(Cubic(Periodic(OnCell())))), CENTERS, CENTERS),
    Periodic(),
)

interpolate_rama_linear(X) = extrapolate(
    scale(interpolate(X, BSpline(Linear())), CENTERS, CENTERS),
    Periodic(),
)

function myheat(x; kwargs...)  # plotting
    X = reshape(x, N_GRID, N_GRID)
    Plots.heatmap(CENTERS, CENTERS, X'; aspect_ratio=1, xlabel="φ", ylabel="ψ",
        xlims=(-π, π), ylims=(-π, π), framestyle=:box, kwargs...)
end

### --- experiment steps ---

""" Load metadynamics samples. Returns x0 of shape (66, nsamples). """
load_x0(path="AMORE/X0_metad.npy") = npzread(path)'

""" Build an OpenMM simulation (5 ps lag time at 450 K). """
build_sim(; tau = 5, steps=round(Int, tau/0.002), temp=450, friction=1) =
    if friction != 1 # workaround for hash
        OpenMMSimulation(; steps, temp, friction)
    else
        OpenMMSimulation(; steps, temp)
    end


""" Bin the starting points by their Ramachandran box. """
function bin_samples(x0)
    bins = [[] for _ in 1:N_GRID^2]
    for x in eachcol(x0)
        push!(bins[boxindex(x)], x)
    end
    return bins
end

""" Starting in each box, compute nk τ-propagated samples.
Returns xt of shape (66, nk, nboxes). """
function compute_xt(sim, x0bins; nk=100)
    xt = zeros(66, nk, length(x0bins))
    @showprogress for (i, bin) in enumerate(x0bins)
        isempty(bin) && (@warn("No starting point in bin $i"); continue)
        for k in 1:nk
            xt[:, k, i] = trajectory(sim; x0=rand(bin), saveevery=OpenMM.steps(sim))[:, end]
        end
    end
    return xt
end

function transfermatrix_gridded(xt)
    dim, nk, nboxes = size(xt)
    C = zeros(N_GRID^2, N_GRID^2)
    for i in 1:N_GRID^2
        for k in 1:size(xt, 2)
            j = boxindex(xt[:, k, i])
            C[i, j] += 1
        end
    end
    C ./ sum(C, dims=2)
end

function transfermatrix_tau(tau, nk=500)
    sim = build_sim(; tau)
    xt = load("xt/xt_$(simhash(sim))_nk$nk.jld2", "xt")
    transfermatrix_gridded(xt)
end

""" Count-based transfer matrix """
function transfermatrix_matching(x0, xt)
    @assert size(x0) == size(xt)
    C = zeros(N_GRID^2, N_GRID^2)
    for (x,y) in zip(eachcol(reshape(x0, 66,:)), eachcol(reshape(xt, 66, :)))
        i = boxindex(x)
        j = boxindex(y)
        C[i, j] += 1
    end
    C ./ sum(C, dims=2)
end

""" Indicator vector over boxes for the target set around (φ, ψ). """
function build_B(phi, psi)
   @show  i0, j0 = Tuple(CartesianIndices((N_GRID, N_GRID))[boxindex((phi, psi))])
    ind = falses(N_GRID, N_GRID)
    for i in -5:4, j in -9:7
        ind[i0 + i, j0 + j] = true
    end
    ind = vec(ind)
    set = findall(ind)
    return ind, set
end

""" Stack of log(PB + ε) interpolants of length horizon+1.
Index k (1-based) corresponds to remaining time (horizon - k + 1)·τ; the last
entry (k = horizon+1) is P^0·indB = indB, so `logpbt` can interpolate up to t = horizon·τ.
Optional Gaussian smoothing with periodic BC to denoise gradients (set σ=0 to disable). """
function build_logPB_stack(P, indB; horizon=20, ε_rel=1e-3, σ=0.0, interp=interpolate_rama_linear)
    PB_k = [P^(horizon - k) * indB for k in 0:horizon]
    if σ > 0
        kernel = Kernel.gaussian((σ, σ))
        PB_k = [vec(imfilter(reshape(p, N_GRID, N_GRID), kernel, "circular")) for p in PB_k]
    end
    ε = ε_rel * maximum(maximum.(PB_k))
    [interp(log.(reshape(max.(p, 0) .+ ε, N_GRID, N_GRID))) for p in PB_k]
end

""" Spectral-truncated stack of log(PB + ε) interpolants. Keeps only the top
`n_modes` eigenmodes of P, which suppresses high-frequency noise while preserving
slow dynamical structure. """
function build_logPB_stack_spectral(P, indB; horizon=20, ε_rel=1e-3, n_modes=10, interp=interpolate_rama_linear)
    F = eigen(P)
    idx = sortperm(abs.(F.values), rev=true)[1:n_modes]
    λ = F.values[idx]
    V = F.vectors[:, idx]
    U = V \ I(size(V, 1))                # left eigenvectors (rows)
    coefs = U * indB                     # projection of indB onto each mode

    PB_k = [real.(V * (λ.^(horizon - k) .* coefs)) for k in 0:horizon]
    ε = ε_rel * maximum(maximum(abs, p) for p in PB_k)
    [interp(log.(reshape(max.(p, 0) .+ ε, N_GRID, N_GRID))) for p in PB_k]
end

U_add(phi, psi, phi0=deg2rad(103.5 - 180), psi0=deg2rad(148.5 - 180)) = (1 - cos(psi-psi0) + 1 + cos(phi-phi0))^4

""" Linear-in-time evaluation of the logPB stack at simulation time `t`.
Blends `logPB_itp[k]` and `logPB_itp[k+1]` with weight α = t/τ - (k-1).
The integer index arithmetic is wrapped in `Zygote.ignore` so the
`floor(Int, ...)` cast is not traced for AD (it lacks an adjoint and would
otherwise raise `UndefVarError(:j)` in Zygote). """
function logpbt(logPB_itp, t, τ, φ, ψ)
    nk = length(logPB_itp)
    k, k1, α = Zygote.ignore() do
        s = t / τ + 1
        k = floor(Int, s)
        (k, min(k + 1, nk), s - k)
    end
    (1 - α) * logPB_itp[k](φ, ψ) + α * logPB_itp[k1](φ, ψ)
end

LASTFNORMS=[]
LASTUNORMS=[]

""" Build the time-dependent Girsanov bias u(x, t) = σ · ∇log PB(x, t). """
function build_bias(sim, logPB_itp; uadd=1, alpha=1, maxnorm=Inf, minnorm=0)
    τ = OpenMM.stepsize(sim) * OpenMM.steps(sim)
    sigma_ODL = OpenMM.constants(sim, true).sigma
    #sigma_ODL = 1
    function bias(x; t, kwargs...)
        F = Zygote.gradient(x) do x
            φ, ψ = phipsi(x)
            logpbt(logPB_itp, t, τ, φ, ψ)
        end |> only
        F_U = Zygote.gradient(x) do x
            φ, ψ = phipsi(x)
            U_add(φ, ψ)
        end |> only

        push!(LASTFNORMS, norm(F))

        rand() > 0.995 && (@show norm(F), phipsi(x))
        if norm(F) < minnorm
            F .= 0
        end


        u = @. alpha * sigma_ODL * F - uadd * F_U
        normu = norm(u)
        if normu > maxnorm
            @info "capped bias force from $normu to $maxnorm", phipsi(x)
            u = u / normu * maxnorm
        end

        push!(LASTUNORMS, norm(u))

        return u
    end
end

""" Run one guided trajectory; returns (weight, endpoint). """
function guided_traj(sim, x0s, bias, horizon)
    x0 = rand(x0s)
    steps = OpenMM.steps(sim) * horizon

    global LASTFNORMS = []
    global LASTUNORMS = []

    xs, logws = ISOKANN.OpenMM.langevin_girsanov!(sim; x0, bias, steps, sigmascaled=true, showprogress=true)

    global LASTGIRSANOV = xs
    global LASTLOGWS = logws |> vec

    plot_lastgirsanov() |> display

    return xs, logws
end

function plot_lastgirsanov()
    p1 = scatter_ramachandran(LASTGIRSANOV, LASTLOGWS, markersize=1, cbar=true)
    Plots.title!("Guided Trajectory, logw = $(LASTLOGWS[end])")

    p2 = scatter_ramachandran(LASTGIRSANOV, LASTUNORMS, markersize=1, cbar=true)
    Plots.title!("Guided Trajectory, logw = $(LASTLOGWS[end])")

    p3 = plot(LASTLOGWS)
    p4 = plot(phipsi(LASTGIRSANOV)[1])

    plot(p3, p1, p4, p2, layout=(2,2), size=(800, 800))
end


""" Run one overdamped-Langevin (Euler-Maruyama) guided trajectory. Returns (xs, logw). """
function guided_traj_od(sim, x0s, bias, horizon)
    x0 = rand(x0s)
    steps = OpenMM.steps(sim) * horizon
    _, logw, xs = ISOKANN.OpenMM.integrate_girsanov(sim; x0, bias, steps)
    p = scatter_ramachandran(xs[:, 1:10:end])
    title!("OD Guided Trajectory, logw = $(logw)") |> display
    return xs, logw
end

#=
""" Girsanov IS estimate of P[X_T ∈ B]. Returns (self-normalized, unnormalized). """
function estimate_pb_girsanov(sim, x0s, bias, setB, horizon; n=500)
    acc = 0.0
    den = 0.0
    hit_w = Float64[]
    @showprogress for _ in 1:n
        x, logw = guided_traj(sim, x0s, bias, horizon)
        den += w
        if boxindex(x) in setB
            acc += w
            push!(hit_w, x)
        end
    end
    return (self_normalized=acc / den, unnormalized=acc / n, hits=hit_w)
end
=#

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
    indB = nothing
    setB = nothing
    # starting point
    phi0::Float64 = deg2rad(103.5 - 180)
    psi0::Float64 = deg2rad(148.5 - 180)
    indx0::Union{Nothing,Int} = nothing
    x0s = nothing
    # bias stack
    horizon::Int = 20  # horizon * tau = total simulation time for Girsanov, MC and transfer-matrix estimates. <=> `N` from the article
    logPB_itp = nothing
    bias = nothing
    u_gain::Float64 = 0
    # results
    pb_transfer::Union{Nothing,Float64} = nothing
    pb_mc::Vector{Bool} = Bool[]                            # hit/miss per MC sample
    pb_g::Vector{Tuple{Float64,Bool, Bool}} = Tuple{Float64,Bool, Bool}[]  # (logweight, hit, final)
end

Experiment1() = experiment()
Experiment1_500() = experiment(horizon=100)
Experiment2() = experiment(tau=2)

function article()
    e1 = Experiment1(n_mc=0, n_girsanov=500)
    e1_500 = experiment(n_mc=500, n_girsanov=0, horizon=100)
    e2 = Experiment2(n_mc=500, n_girsanov=500)
end

function experiment(; nk=500, n_mc=1, n_girsanov=1, tau=5, friction=1, kwargs...)
    e = Experiment(; kwargs...)
    global E = e
    e.sim = build_sim(; tau, friction)
    e.x0 = load_x0()
    e.x0bins = bin_samples(e.x0)

    @time step_xt!(e; nk)
    e.P = transfermatrix_gridded(e.xt)
    @info "top 3 eigenvalues" vals = reverse(eigen(e.P).values)[1:3]

    e.indB, e.setB = build_B(e.phiB, e.psiB)
    e.indx0 = boxindex((e.phi0, e.psi0))
    e.x0s = e.x0bins[e.indx0]
    e.pb_transfer = (e.P^e.horizon*e.indB)[e.indx0]
    @info "transfer-matrix estimate" e.pb_transfer

    @time step_mc!(e; n=n_mc)
    @info "MC estimate" pb_mc_estimate(e) n_mc=length(e.pb_mc)

    e.logPB_itp = build_logPB_stack(e.P, e.indB; e.horizon,)
    e.bias = build_bias(e.sim, e.logPB_itp, uadd = e.u_gain)
    @time step_girsanov!(e; n=n_girsanov)
    @info "Girsanov IS estimate" pb_g_estimate(e; normalized=false) n_g=length(e.pb_g)

    @info "summary" summary(e)...
    e
end

""" OD variant of `experiment`: uses Euler-Maruyama overdamped Langevin (`integrate_girsanov`)
for the IS samples. The OD optimal control `u = σ_OD·∇log P_B` is genuinely optimal here,
so logw should be much smaller in magnitude and ESS much larger than the UDL variant.
Note: OD and UDL hitting probabilities differ at finite friction; pb_transfer/pb_mc still
refer to the UDL dynamics used to build P. """
function experiment_od(; nk=500, n_mc=500, n_girsanov=500, tau=5, friction_od=1000, kwargs...)
    e = Experiment(; kwargs...)
    global E = e
    e.sim = build_sim(; tau)                                     # γ=1, UDL — for xt/P/MC
    sim_od = build_sim(; tau, friction=friction_od)              # γ=γ_od, for EM stability
    e.x0 = load_x0()
    e.x0bins = bin_samples(e.x0)

    @time step_xt!(e; nk)
    e.P = transfermatrix_gridded(e.xt)
    @info "top 3 eigenvalues" vals = reverse(eigen(e.P).values)[1:3]

    e.indB, e.setB = build_B(e.phiB, e.psiB)
    e.indx0 = boxindex((e.phi0, e.psi0))
    e.x0s = e.x0bins[e.indx0]
    e.pb_transfer = (e.P^e.horizon*e.indB)[e.indx0]
    @info "transfer-matrix estimate" e.pb_transfer

    @time step_mc!(e; n=n_mc)
    @info "MC estimate" pb_mc_estimate(e) n_mc=length(e.pb_mc)

    e.logPB_itp = build_logPB_stack(e.P, e.indB; e.horizon, )
    e.bias = build_bias(sim_od, e.logPB_itp; uadd =e.u_gain)            # σ_OD from sim_od
    @time step_girsanov_od!(e; n=n_girsanov, sim=sim_od)
    @info "OD Girsanov IS estimate" pb_g_estimate(e; normalized=false) n_g=length(e.pb_g) ess=pb_g_ess(e)

    @info "summary" summary(e)...
    e
end

function from_jakob(;n_mc=500, n_girsanov=1)
    e = Experiment()
    e.sim = build_sim(; tau=5)
    e.x0 = npzread("AMORE/X0_metad.npy")'
    e.xt = npzread("AMORE/Xtau_metad.npy")'
    e.x0bins = bin_samples(e.x0)
    e.P = transfermatrix_matching(e.x0, e.xt)
    @info "top 3 eigenvalues" vals = reverse(eigen(e.P).values)[1:3]

    e.indB, e.setB = build_B(e.phiB, e.psiB)
    e.indx0 = boxindex((e.phi0, e.psi0))
    e.x0s = e.x0bins[e.indx0]
    e.pb_transfer = (e.P^e.horizon*e.indB)[e.indx0]
    @info "transfer-matrix estimate" e.pb_transfer

    @time step_mc!(e; n=n_mc)
    @info "MC estimate" pb_mc_estimate(e) n_mc=length(e.pb_mc)

    e.logPB_itp = build_logPB_stack(e.P, e.indB; e.horizon, )
    e.bias = build_bias(e.sim, e.logPB_itp)
    @time step_girsanov!(e; n=n_girsanov)
    @info "Girsanov IS estimate" pb_g_estimate(e; normalized=false) n_g=length(e.pb_g)

    @info "summary" summary(e)...
    e
end



simhash(sim) = string(hash(sim.constructor); base=16)

function step_xt!(e; nk=100, save_dir="xt", reuse_if_exists=true)
    tag = "$(simhash(e.sim))_nk$(nk)"
    save_path = joinpath(save_dir, "xt_$tag.jld2")
    if reuse_if_exists && isfile(save_path)
        @info "loading xt from $save_path"
        e.xt = load(save_path, "xt")
        return e
    end

    @info "computing xt with nk=$nk, save_path=$save_path"
    e.xt = compute_xt(e.sim, e.x0bins; nk)
    jldsave(save_path; xt=e.xt, sim=e.sim)

    return e
end

""" Append n Monte-Carlo samples to e.pb_mc. Runs for horizon·τ, matching the Girsanov and transfer-matrix horizons. """
function step_mc!(e; n=100)
    manysteps = OpenMM.steps(e.sim) * e.horizon
    @showprogress for _ in 1:n
        x0 = rand(e.x0s)
        x = trajectory(e.sim, manysteps; x0, saveevery=manysteps)[:, end]
        hit = boxindex(x) in e.setB
        #hit && println("hit at iteration $i")
        push!(e.pb_mc, hit)
    end
    e
end

ERRS = []
LASTGIRSANOV = nothing

""" Append n Girsanov IS samples to e.pb_g. """
function step_girsanov!(e; n=1)
    @showprogress for _ in 1:n
        #try


            xs, logws = guided_traj(e.sim, e.x0s, e.bias, e.horizon)

            hit_any = any(boxindex(x) in e.setB for x in eachcol(xs))
            hit_end = boxindex(xs[:, end]) in e.setB
            push!(e.pb_g, (logws[end], hit_end, hit_any))
        #=catch err
            err isa InterruptException && break
            @show err
            push!(ERRS, err)
            push!(e.pb_g, (NaN, false, false))
        end=#
    end
    e
end

""" Append n overdamped-Langevin Girsanov IS samples to e.pb_g (uses integrate_girsanov).
Pass `sim=sim_od` to use a high-friction sim for EM stability while keeping `e.sim` as the
UDL sim used for the transfer matrix and MC. """
function step_girsanov_od!(e::Experiment; n=1, sim=e.sim)
    @showprogress for _ in 1:n
        try
            xs, logw = guided_traj_od(sim, e.x0s, e.bias, e.horizon)
            hit_any = any(boxindex(x) in e.setB for x in eachcol(xs))
            hit_end = boxindex(xs[:, end]) in e.setB
            push!(e.pb_g, (logw, hit_end, hit_any))
        catch err
            err isa InterruptException && break
            push!(e.pb_g, (NaN, false, false))
        end
    end
    e
end

### --- reporters ---

pb_mc_estimate(e) = isempty(e.pb_mc) ? NaN : mean(e.pb_mc)

function pb_g_estimate(e; normalized=true)
    isempty(e.pb_g) && return NaN
    lws = [lw for (lw, _, _) in e.pb_g]
    m = maximum(lws)  # logsumexp shift for stability
    num = sum(exp(lw - m) for (lw, hit_end, _) in e.pb_g if hit_end; init=0.0)
    den = normalized ? sum(exp(lw - m) for lw in lws) : length(e.pb_g) * exp(-m)
    num / den
end

pb_g_hitrate(e) = isempty(e.pb_g) ? NaN : mean(hit_any for (_, _, hit_any) in e.pb_g)

""" Effective sample size of the Girsanov weights in `e.pb_g`. ESS = (Σwᵢ)² / Σwᵢ².
ESS ≪ length(pb_g) signals that IS variance is dominated by a few outlier weights. """
function pb_g_ess(e::Experiment)
    isempty(e.pb_g) && return NaN
    lws = Float64[lw for (lw, _, _) in e.pb_g if isfinite(lw)]
    isempty(lws) && return NaN
    m = maximum(lws)
    s1 = sum(exp(lw - m)       for lw in lws)
    s2 = sum(exp(2 * (lw - m)) for lw in lws)
    return s1^2 / s2
end

function summary(e::Experiment)
    (; pb_transfer=e.pb_transfer,
       pb_mc=pb_mc_estimate(e), n_mc=length(e.pb_mc),
       pb_g_selfnorm=pb_g_estimate(e; normalized=true),
       pb_g_unnorm=pb_g_estimate(e; normalized=false),
       pb_g_hitrate=pb_g_hitrate(e),
       pb_g_ess=pb_g_ess(e),
       n_g=length(e.pb_g))
end

function plot_eigenmodes(e)
    # figure 14
     eig = eigen(e.P')
    @show eig.values[end-2:end]
    myheat(real(eig.vectors[:, end]), title="Eigenvector to λ=$(round(real(eig.values[end]), digits=3)) of P'", cmap=:viridis) |> display
    myheat(real(eig.vectors[:, end-1]), title="Eigenvector to λ=$(round(real(eig.values[end-1]),digits=3)) of P'", cmap=:diverging_bwr_55_98_c37_n256, clims=(-0.25, 0.25)) |> display
end



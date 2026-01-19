using ISOKANN
using ISOKANN: coords, phi, psi, isotarget, dchidx
using ISOKANN.OpenMM: friction, stepsize, masses, temp, force, steps


using LinearAlgebra: norm, dot
using StatsBase: mean, mean_and_std, sample, std
using ProgressBars
using ProgressMeter
using Plots
using StructArrays

using Base: @deprecate

scatr(x;kw...) = scatter!(eachrow(x)...;kw...)
signedsqrt(x) = sign.(x) .* sqrt.(abs.(x))
function scatter_api(iso; xs=coords(iso))
    
    z = chicoords(iso, xs) |> vec
    support = iso.data.sim.support
    bias = ustar(cpu(iso))
    f = mapreduce(bias, hcat, eachcol(xs))
    f = signedsqrt.(f./100)
    scatter(eachrow(xs)..., marker_z=z, hover=z, quiver=tuple(eachrow(f)...))
    quiver!(eachrow(xs)..., 
        quiver = tuple(eachrow(f)...), 
        #xlims=support[1,:], 
        #ylims=support[2,:], 
        color=:grey, alpha=0.3)
end

function contour_api(ia)
    xs = Float32.(range(-2, 1, length=200))
    ys = Float32.(range(-.5, 2, length=200))
    f(x, y) = ia.model([x, y]) |> only
    contour(xs,ys, f, clims=(0,10), levels=20) 
    #beta = 2 / sim.sigma^2
    contour!(xs, ys, (x,y)->ISOKANN.potential(ia.data.sim, [x,y]) ./ 10, clims=(-10,10), levels=10)
end

boxA = [-3 -2.6; 2.5 3] # phi lims; psi lims
boxB = [-1.5 -1; 0 1]
boxTrans = [-1.5 -1; 2 3]

torsion(x::AbstractVector) = [ISOKANN.phi(x), ISOKANN.psi(x)]
torsion(x::AbstractArray) = mapslices(torsion, x, dims=1)
inbox(x, box) = all(box[:,1] .< x .< box[:,2])

classify(X::AbstractMatrix) = classify.(eachcol(X))
function classify(x::AbstractVector)
    a = torsion(x)
    inbox(a, boxA) && return :A
    inbox(a, boxB) && return :B
    inbox(a, boxTrans) && return :trans #transition region
    return :interior # not classified
end

function plot_boxes!()
    box = boxA
    x = [box[1], box[3], box[3], box[1]]
    y = [box[2], box[2], box[4], box[4]]
    plot!(x, y, seriestype=:shape, c=:red, alpha=0.3, label="A")

    box = boxB
    x = [box[1], box[3], box[3], box[1]]
    y = [box[2], box[2], box[4], box[4]]
    plot!(x, y, seriestype=:shape, c=:blue, alpha=0.3, label="B")
end

# monte carlo computation of the committor
function empirical_committor(x0::AbstractVector; sim, K=1, tmax=1_000_000, trajectory=false)
    hits = Int[]
    ts = Int[]
    ys = Matrix{Float32}[]
    for k in 1:K
        OpenMM.setcoords(sim, x0)
        OpenMM.set_random_velocities!(sim)
        t = 0
        x = x0
        xs = copy(x0)
        for j in 1:tmax
            c = classify(x)
            if c > 0
                push!(hits, c)
                push!(ts, t)
                trajectory && push!(ys, reshape(xs, length(x), :))
                break
            else
                t += 1
                sim.pysim.step(1)
                x = coords(sim)
                trajectory && append!(xs, x)
            end
        end
        @show t
    end
    length(hits) < K && @warn("Not all trajectories hit in time")
    q = sum(==(2), hits) / length(hits)
    return (;q, hits, ts, x0, ys)
end

empirical_committor(X::AbstractMatrix; kwargs...) = StructArray(empirical_committor(x0;kwargs...) for x0 in eachcol(X))
empirical_committor(data::SimulationData; kwargs...) = empirical_committor(coords(data); kwargs...)
function refine_committor(Qs; kwargs...)
    map(Qs) do q1
        (;q, hits, ts, x0, ys) = Q
        q2 = empirical_committor(x0; kwargs...)
        hits = vcat(q1.hits, q2.hits)
        ts = vcat(q1.ts, q2.ts)
        ys = hcat(q1.ys, q2.ys)
        q = sum(==(2), hits) / length(hits)
        (;q, hits, ts, x0, ys )
    end
end



function plot_committor(Q=Q; kwargs...)
    plot(legend=false)
    plot_boxes!()
    for x in Q
        for y in x.ys
            plot!(ISOKANN.phi(y), ISOKANN.psi(y); kwargs...)
        end
    end
    savefig("committortraj.png")
end

# target for supervised "assignment"
struct SupervisedTarget
    values::Vector{Float32}
end
ISOKANN.isotarget(iso, ct::SupervisedTarget) = ct.values'

### ISOKANN - Committor variant

# compute committor using Kq = q with boundary cond.
mutable struct CommittorTarget{T}
    maskA::T
    maskB::T
end

function CommittorTarget(data, classify=classify)
    x = coords(data)
    y = ISOKANN.propcoords(data)
    d, K, N = size(y)
    maskA = zeros(Bool, 1, K, N)
    maskB = zeros(Bool, 1, K, N)
    for n in 1:N
        c = classify(x[:, n])
        if c == :A
            maskA[1, :, n] .= 1
        elseif c == :B
            maskB[1, :, n] .= 1
        else
            for k in 1:K
                c = classify(y[:,k,n])
                maskA[1, k, n] = c == :A
                maskB[1, k, n] = c == :B
            end
        end
    end

    return CommittorTarget(maskA, maskB)
end

# if the data is changing, update committor targets (used in `runadaptive!`)
function maybeupdate!(t::CommittorTarget, data)
    if length(data) != size(t.maskA, 3)
        tt = CommittorTarget(data)
        t.maskA = tt.maskA
        t.maskB = tt.maskB
    end
end

function ISOKANN.isotarget(iso, t::CommittorTarget)
    maybeupdate!(t, iso.data)
    y = iso.model(propfeatures(iso.data))
    y[t.maskA] .= 0
    y[t.maskB] .= 1
    dropdims(sum(y; dims=2); dims=2) ./ size(y, 2)
end

#####


# generate long traj. data `DATA` and "ground truth" ISOKANN committor `ISO`
function gendata(; steps=100, nx0 = 10_000, nx=1000, k=8, iter=2000)

    global SIM, SIM0, X, DATA, ISO

    SIM = OpenMMSimulation(steps=steps, integrator="brownian", step=1e-5,
        constraints="HBonds",
        )

    # no constraint, needed for correct force evals but requires smaller stepsize
    SIM0 = OpenMMSimulation(steps=steps*2, integrator="brownian", step=5e-6)

    X = OpenMM.laggedtrajectory(SIM, nx0)#, step=0.1, minimize=true)
    featurizer = OpenMM.FeaturesAtoms(findall(a.element.symbol != "H" for a in OpenMM.atoms(SIM)))
    ix = picking(featurizer(X), nx).is
    DATA = SimulationData(SIM, X[:, ix], k; featurizer)

    #committors = empirical_committor(data;K)  # monte carlo committor estimation
    #iso = Iso(data, transform=CommittorTarget(committors))

    ISO = Iso(DATA, transform=CommittorTarget(DATA))
    run!(ISO, iter)
end

function comm_error(ISO)
    (sqrt(mean(abs2,chicoords(ISO, reduce(hcat, filter(x->classify(x)==:A, eachcol(coords(ISO))))))),
    sqrt(mean(abs2,chicoords(ISO, reduce(hcat, filter(x->classify(x)==:B, eachcol(coords(ISO))))).-1)))
end


function validationdata(;nx0=10_000, nx=20, K=100)
    X = OpenMM.laggedtrajectory(SIM, nx0)
    featurizer = OpenMM.FeaturesAtoms(findall(a.element.symbol != "H" for a in OpenMM.atoms(SIM)))
    #X = X[:, classify(X) .== 0]
    xs = X[:, picking(featurizer(X), nx).is]
    global Q = empirical_committor(xs; sim=SIM, K=K)
end

Q_loss(iso) = sum(abs2, only(chicoords(iso, q.x0)) - q.q for q in Q)

function production()
    @time gendata()
    @time validationdata()
    @time iso = iso_api()
    @time run!(iso, 1_000, 10)
end


### Optimal Control and Girsanov

# zero control
const Z = zeros(66)
uzero(x) = Z

# optimal control
function ustar(iso::Iso; eps=1e-4, maxnorm=1000)
    σ = constants(iso.data.sim).σ
    function ustar_iso(x)
        x = Float32.(x)
        phi = ISOKANN.chicoords(iso, x) |> cpu |> only
        dphi = ISOKANN.dchidx(iso, x)
        c = dphi ./ max(phi,eps)
        if !all(isfinite.(c))
            @show x
            @error "infinite control"
        end
        u = σ .* c  
        if norm(u) > maxnorm
            u = u / norm(u) * maxnorm
        end

        return u
    end
    return ustar_iso
end

function constants(sim::OpenMMSimulation)
    kB = 0.008314463
    γ = friction(sim)

    M = repeat(masses(sim), inner=3)
    T = temp(sim)
    σ = @. sqrt(2 * kB * T / (γ * M))

    (;kB, γ, M, T, σ)
end

girsanov(sim, x0::AbstractVector;kwargs...) = _girsanov(sim, x0;kwargs...)

# dispatch for arrays
function girsanov(sim, x0::AbstractArray; kwargs...)
    dims = Tuple(2:ndims(x0))
    x = @showprogress "Girsanov sampling" map(eachslice(x0; dims)) do x0
        _girsanov(sim, x0; kwargs...)
    end
    StructArray(x)
end


global Ts = Int[]
global HIST = []
global DEBUG::Bool = false


## simulate the overdamped langevin dynamics
## dX^u_t = (F/(γ*M) + σ*u) dt + σ dB_t
## where u = bias(X_t)

function _girsanov(sim, x0=coords(sim); steps=steps(sim), dt=stepsize(sim), bias, trajectory=true, classify=classify)
    (; kB, γ, M, T, σ) = constants(sim)

    x = copy(x0)
    g = 0.0
    u2 = 0.0

    z = trajectory ? similar(x, length(x), steps) : similar(x, 0, 0)

    T = 0
    while T < steps

        if classify(x) in (:A, :B)
            trajectory && (z = z[:, 1:T])
            break
        end

        T+=1

        F = force(sim, x, reclaim=false)
        ux = bias(x)
        if norm(F) > 10_000
            @show F, x
        end
        if norm(ux) > 1_000
            @show ux, x, x0
            ux ./= norm(ux) * 1_000
        end
        #println(" F", norm(F), " u", norm(ux), " x", x)
        @assert all(isfinite.(x))
        @assert all(isfinite.(F))
        dg, du2 = girsanov_step!(x, F, M, σ, γ, dt, ux)
        g += dg
        u2 += du2
        @assert isfinite(u2)
        trajectory && ( z[:, T] = x)
    end

    DEBUG && push!(Ts, T) # keep track of how long we simulate

    return (;x, g, z, u2, T)
end


function girsanov_step!(x, F, M, σ, γ, dt, u)
    dB = randn(length(x)) * sqrt(dt)
    @. x += (1 / (γ * M) * F + (σ * u)) * dt + σ * dB
    @assert all(isfinite.(x))
    dg = dot(u, u) / 2 * dt + dot(u, dB)
    du2 = dot(u, u) * dt
    @assert isfinite(du2)
    return dg, du2
end


### approximate policy iteration

@kwdef mutable struct ApproxPolicyIter3{T}
    Js::T
    K::Int # number of new samples per target evaluation
    α::Float32
    classifier
    q0::Float64 = 0
    q1::Float64 = 1
end

logclamped(x, eps=typeof(x)(1e-8)) = x < eps ? log(eps) : log(x)

global RESAMPLE_MAXSTEPS::Int =1000

function resample!(t::ApproxPolicyIter3, iso;
        α=t.α, 
        K=t.K, 
        is=sort(sample(1:length(t.Js), K, replace=false)),
        maxsteps=RESAMPLE_MAXSTEPS,
        dt = stepsize(sim))

    bias = ustar(cpu(iso)) # cpu is faster for single calls
    #bias(x) = [0, 0]

    for i in is
        g = girsanov(iso.data.sim, coords(iso)[:, i]; steps=maxsteps, bias=bias, classify=t.classifier, dt)

        class = t.classifier(g.x)
        if class == :A
            j = g.u2 / 2 - logclamped(t.q0)
        elseif class == :B
            j = g.u2 / 2 - logclamped(t.q1)
        else
            j = g.u2 / 2 - logclamped(iso.model(g.x)|>only)
        end

        @assert isfinite(j)

        if DEBUG
            append!(QS[i], exp(-j))
            h = (;i, g.T, g.u2, q_j=exp(-j), q_J=exp(-t.Js[i]), g)
            #@show (;i, g.class, g.T, g.u2, q=exp(-j))
            push!(HIST, h)
        end

        if !isfinite(j)
            @show class, j, g.u2
            @error "got NaN for target"
        end
            
        
        t.Js[i] += α * (j - t.Js[i]) # moving average
    end
end

# sample a single J realization per point (from K randomly chosen) and update moving average, return q estimate for training
function ISOKANN.isotarget(iso, t::ApproxPolicyIter3)
    global Js
    if t.α > 0
        resample!(t, iso)
    end
    q = exp.(-t.Js)
    return q'
end

using ISOKANN.Flux

function iso_api(; nx=200, α=Float32(1 / 10), K=nx, iso_q=nothing, classifier, data=iso_q.data, model=iso_q.model, q0::Float64=0., q1::Float64=1., isoargs...)
    ix = picking(features(data), nx).is
    data = data[ix]
    if !isnothing(iso_q)
        J0 = -logclamped.(chicoords(iso_q, coords(data)) |> vec)
        transform = ApproxPolicyIter3(J0, K, α, classifier, q0, q1)
    else
        J0 = init_J(coords(data), classifier, q0, q1)
        transform = ApproxPolicyIter3(J0, K, α, classifier, q0, q1)
    end
    
    iso = Iso(data; transform, model=deepcopy(model), isoargs...)
    global QS = [Float32[] for i in 1:nx]
    return iso
end

function init_J(xs, classifier, q0, q1)
    c = classifier.(eachcol(xs))
    J0 = zeros(size(xs, 2))
    J0 .= -log(q0+q1 / 2)
    J0[c .== :A] .= -log(q0)
    J0[c .== :B] .= -log(q1)
    J0
end    

function benchmark_trajs()
    for gpu in [1, "gpu"]
        sim = OpenMMSimulation(integrator="brownian", step=5e-6, mmthreads=gpu)
        x0 = coords(sim)
        #=
        @time "$gpu traj no save" OpenMM.trajectory(sim, 1000; x0, saveevery=1000)
        @time "$gpu traj with save" OpenMM.trajectory(sim, 1000; x0, saveevery=1)
        @time "$gpu girsanov nosave uzero" girsanov(sim, x0=x0, bias=uzero, steps=1000)
        @time "$gpu girsanov save uzero" girsanov(sim, x0=x0, bias=uzero, steps=1000, trajectory=true)
        =#

        for dev in [cpu, Flux.gpu]
            u = ustar(dev(ISO))
            @time "openmm:$gpu girsanov nosave ustar flux: $dev" girsanov(sim, x0, bias=u, steps=1000)
        end
    end


end

function plot_girsanov_force(g, iso)
    t = g.z
    u = ustar(cpu(iso))
    us = [norm(u(x)) for x in eachcol(t)]
    scatter_ramachandran(t, us)
    plot_boxes!()
end


#=

archive

function x_transition(iso)
    xs = coords(iso)[:, 0.45 .< vec(chis(iso)) .< 0.55]
    i = argmin(vec(OpenMM.potential(iso.data.sim, xs)))
    xs[:, i]
end


scatcoords!(x::AbstractVector; kw...) = scatcoords!(reshape(x,:,1); kw...)
function scatcoords!(x; kwargs...)
    a = torsion(x)
    scatter!(a[1,:], a[2,:]; kwargs...)
end

function pickdata(data, n)
    is = ISOKANN.picking(features(data), n).is
    data[is]
end

function iso_torsion(data=DATA)
    featurizer = torsion
    data = SimulationData(data.sim, data.coords; featurizer)
    Iso(data,
        transform=CommittorTarget(data),
        model = ISOKANN.densenet(; layers=[2,16,16,16,1], layernorm=false, activation=Flux.sigmoid),
        opt=AdamRegularized(1e-2, 0)
    )
end

function api_data(data=DATA, n=100)
    is = picking(ISOKANN.flattenlast(data.features[2]), n).is
    xs = ISOKANN.flattenlast(data.coords[2])[:,is]
    SimulationData(SIM0, (xs,xs); featurizer=torsion)
end

=#
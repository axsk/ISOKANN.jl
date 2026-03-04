# think about the committor system as
# 1) dynamical system
# 2) classifier / sets A+B
# 3) committor representation, either a) sampled, b) ISO-committor, c) variational committor

# lets start with the sampled one

using LinearAlgebra:norm
using SqraCore

function classify_mb(x)
    norm(x - [-0.56, 1.44]) < 0.2 && return :A # note: had to increase radius so that sufficient points lie inside the core set s.t. bnd. conds. are enforced
    norm(x - [0.62, 0.03]) < 0.2 && return :B
    return :interior # not classified
end

function plot_cores!()
    centers = [[-0.56, 1.44], [0.62, 0.03]]
    r = 0.2
    θ = range(0, 2π, length=200)
    for c in centers
        plot!(c[1] .+ r*cos.(θ), c[2] .+ r*sin.(θ))
    end
    plot!()
end

struct Committor
    sim
    xs
    ys
    u
    Q
    classes
    q
end

Plots.heatmap(c::Committor) = Plots.heatmap(c.xs, c.ys, c.q')

function Committor(sim; h=0.01, maxiter = 10000)
    xs = sim.support[1,1]:h:sim.support[1,2]
    ys = sim.support[2,1]:h:sim.support[2,2]
    u = [sim.potential([x,y]) for x in xs, y in ys]
    Q = SqraCore.sqra_grid(u, beta=SqraCore.beta_from_sigma(sim.sigma))
    classes = map(vec([classify_mb([x,y]) for x in xs, y in ys])) do c
        c === :A && return 0.1 # replaced by 0 in `committor`
        c === :B && return 1
        return 0
    end
    q = committor(Q, classes;maxiter)
    q = reshape(q, length(xs), length(ys))

    return Committor(sim,xs,ys,u,Q,classes,q)
end

@inline function value(c::Committor, x)
    i = floor(Int, (x[1] - first(c.xs)) / step(c.xs) + 0.5) + 1
    j = floor(Int, (x[2] - first(c.ys)) / step(c.ys) + 0.5) + 1

    return (1 ≤ i ≤ length(c.xs) && 1 ≤ j ≤ length(c.ys)) ? c.q[i,j] : NaN
end

value(c::Committor, iso::ISOKANN.Iso) = [value(c, x) for x in eachcol(coords(iso))]

import ISOKANN.validationloss
# l2 difference over the Iso samples
function ISOKANN.validationloss(iso, c::Committor; plot=false)
    q1 = value(c, iso)
    q2 = (chis(iso) .- iso.transform.q0) ./ (iso.transform.q1 - iso.transform.q0) |> vec
    
    if plot
        p = sortperm(q1)
        Plots.plot([q1[p] q2[p]]) |> display
        Plots.scatter(eachrow(coords(iso))..., marker_z = q2 - q1) |> display
    end

    i = findall(!isnan, q1)
    i = findall(0.2 .< q1 .< 0.8) # focus on transition region only
    q1 = q1[i]
    q2 = q2[i]

    return mean(abs2.(q1-q2))
end




constants(sim::ISOKANN.Diffusion) = (kB=1, γ=1, M=1, T=1, σ=sim.sigma, beta=2/sim.sigma^2)
OpenMM.stepsize(sim::ISOKANN.Diffusion) = ISOKANN.dt(sim)


function run_mb()
    global sim, data, model, iso
    sim = ISOKANN.MuellerBrown(sigma=6, lagtime=0.01)
    data = trajectorydata_bursts(sim, 10_000, 10, x0=[0,0], featurizer=x->Float32.(x))
    model = ISOKANN.densenet(layers=[2, 20, 20, 1], activation=Flux.tanh, layernorm=false)
    iso_q=Iso(data, transform=CommittorTarget(data, classify_mb), model=deepcopy(model), opt=AdamRegularized())
    run!(iso_q, 10_000)
    ia = iso_api(;iso_q, classifier=classify_mb, opt=AdamRegularized(), model=deepcopy(model)) 
end

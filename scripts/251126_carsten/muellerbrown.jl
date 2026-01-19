# think about the committor system as
# 1) dynamical system
# 2) classifier / sets A+B
# 3) committor representation, either a) sampled, b) ISO-committor, c) variational committor

# lets start with the sampled one

using LinearAlgebra:norm

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
    


constants(sim::ISOKANN.Diffusion) = (kB=1, γ=1, M=1, T=1, σ=sim.sigma, beta=2/sim.sigma^2)
OpenMM.stepsize(sim::ISOKANN.Diffusion) = ISOKANN.dt(sim)


function run_mb()
    global sim, data, model, iso
    sim = ISOKANN.MuellerBrown(sigma=6, lagtime=0.01)
    data = trajectorydata_bursts(sim, 10_000, 10, x0=[0,0])
    model = ISOKANN.densenet(layers=[2, 20, 20, 1], activation=Flux.tanh, layernorm=false)
    iso_q=Iso(data, transform=CommittorTarget(data, classify_mb), model=deepcopy(model), opt=AdamRegularized())
    run!(iso_q, 10_000)
    ia = iso_api(;iso_q, classifier=classify_mb, opt=AdamRegularized(), model=deepcopy(model)) 
end

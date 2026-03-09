using Flux, Zygote
using ISOKANN.OpenMM: integrate_girsanov
using ISOKANN: coords

struct MetadynamicsState
    centers::Vector{Vector{Float32}}   # visited RC values s_i
    height::Float32                     # h
    sigma::Float32                      # σ
end

MetadynamicsState(; height=0.1f0, sigma=0.1f0) = 
    MetadynamicsState(Vector{Float32}[], height, sigma)

# Bias potential in RC space: V(s) = sum_i h * exp(-|s-s_i|^2 / 2σ^2)
function bias_potential(meta::MetadynamicsState, s::AbstractVector)
    isempty(meta.centers) && return 0f0
    sum(meta.centers) do sᵢ
        meta.height * exp(-sum(abs2, s .- sᵢ) / (2 * meta.sigma^2))
    end
end

# Bias force in configuration space: -dV/dx = -dV/ds * ds/dx
function make_bias(iso, meta::MetadynamicsState)
    function bias(x::AbstractVector; kwargs...)
        grad, = Zygote.gradient(x) do x_
            s = chicoords(iso, x_)
            bias_potential(meta, s)
        end
        return -grad   # negative gradient = force
    end
    return bias
end

mutable struct MetadynamicsSimulator
    sim
    iso
    force
    meta::MetadynamicsState
end

function plot_profile(mdsim::MetadynamicsSimulator, zs=0:0.01:1)
    V = [bias_potential(mdsim.meta, [z]) for z in zs]
    plot(zs, V, xlabel="z", ylabel="V(z)", title="Metadynamics Bias Potential")
end

function MetadynamicsSimulator(iso; height=0.1f0, sigma=0.1f0)
    sim = iso.data.sim
    meta = MetadynamicsState(eachcol(chis(iso)), height, sigma)
    bias = make_bias(iso, meta)
    return MetadynamicsSimulator(sim, iso, bias, meta)
end

function trajectory(mdsim::MetadynamicsSimulator; x0=coords(mdsim.iso)[:, end])
    x = x0
    traj = [x0]
    
    #for i in 1:n_rounds
    return OpenMM.langevin_girsanov!(mdsim.sim; x0=x, bias=mdsim.force)
    #    push!(traj, x)
        #s = chicoords(iso, x) |> vec
        #push!(sim.meta.centers, s)
    #end
    
    #return traj
end

# Main metadynamics loop
function run_metadynamics(iso;
        sim=iso.data.sim,
        x0=coords(iso.data)[:, end],
        n_rounds=100,
        #deposit_every=10,      # steps between Gaussian deposits
        height=0.1f0,
        sigma=0.1f0)
    
    meta = MetadynamicsState(; height, sigma)
    x = x0
    global trajectory = [x0]
    
    for i in 1:n_rounds
        # Build current bias function (closes over meta state)
        bias = make_bias(iso, meta)
        
        # Integrate for deposit_every steps
        x = integrate_girsanov(sim; x0=x, bias)
        push!(trajectory, x)
        
        # Deposit Gaussian at current RC value
        s = iso.model(x) |> vec
        push!(meta.centers, s)
        
        @info "Round $i, RC value: $s, N gaussians: $(length(meta.centers))"
    end
    
    return trajectory, meta
end
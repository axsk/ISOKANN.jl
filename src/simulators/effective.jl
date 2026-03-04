using NearestNeighbors
import Zygote
using LinearAlgebra: cholesky

import ISOKANN: coords, trajectory, dim, coords


struct EffectiveSimulation <: ISOKANN.IsoSimulation
    kde
    steps
    dt
end

dim(eff::EffectiveSimulation) = length(coords(eff.kde)[1])
coords(eff::EffectiveSimulation) = coords(eff.kde)[1]

# construct the KDEExpectation from an Iso object and store it in an EffectiveSimulation
function EffectiveSimulation(iso; 
        h=0.1, 
        dt=0.001, 
        steps=1000, 
        xi=x -> chicoords(iso, x),
        sim = iso.data.sim,
        xs = coords(iso)
        )
    
    (;sigma, M, gamma) = OpenMM.constants(sim)
    forcescale = 1 ./ (gamma .* M)

    bA = mapreduce(hcat, eachcol(xs)) do x
        x = Float32.(x)
        F = OpenMM.force(sim, x) .* forcescale # force == -∇V
        b, A = b_and_A(xi, x, F, sigma)
        vcat(b, vec(A))
    end

    zs = xi(xs)
    #zs = chis(iso)
    kde = KDEExpectation(zs, bA, h)

    EffectiveSimulation(kde, 1000, dt)
end

# eq (15,16) in Legoll, Lelievre, (8) in Sikorski, Donati, Weber, Schütte, (40) in Zhang, Hartmann, Schütte [2016]
# pointwise projection based on Itos formula
function b_and_A(xi, x::AbstractVector{Float32}, F, sigma)
    J = jac(xi, x)
    H = laplace_sigma2half(xi, x, sigma)

    b = J * F .+ H

    s = J .* sigma'
    A = s * s'
    return b, A
end

jac(xi, x) = Zygote.jacobian(xi, x) |> only

function laplace_sigma2half(xi, x, sigma; n=length(xi(x)))
    @assert sigma isa AbstractVector # diagonal noise in original dynamics
    ntuple(n) do i
        H = only(Zygote.diaghessian(x -> xi(x)[i], x))
        sum(@. sigma^2 / 2 * H)
    end
end

function trajectory(eff::EffectiveSimulation, steps=eff.steps; saveevery=1, x0=coords(eff), dt=eff.dt)
    # euler Maruyama
    x = copy(x0)
    xs = similar(x0, length(x0), div(steps, saveevery))
    for i in 1:steps
        b, sigtilde = b_and_sigma(eff, x)
        x += b * dt + sigtilde * sqrt(dt) * randn(length(x))
        if i % saveevery == 0
            xs[:, div(i, saveevery)] = x
        end
    end
    return xs
end

# decompose b and A stored in the KDE as vector and compute sigma via cholesky decomposition
function b_and_sigma(eff::EffectiveSimulation, z)
    bA = marginal(eff.kde, z)
    K = length(z)
    b = bA[1:K]
    A = reshape(@views(bA[K+1:end]), length(z), length(z))
    sigtilde = cholesky(A).U
    return b, sigtilde
end

# TODO: make mutable (h)
struct KDEExpectation{T,S<:NearestNeighbors.NNTree}
    tree::S
    fs::Matrix{T}
    h::T
end

function KDEExpectation(zs::AbstractMatrix, fs::AbstractMatrix, h::Number)
    KDEExpectation(KDTree(zs, reorder=false), fs, eltype(fs)(h))
end

### Conditional expectation using KD-Tree based KDE

function coords(E::KDEExpectation)
    @assert !E.tree.reordered
    return E.tree.data
end

function marginal_and_weight(E::KDEExpectation, z)
    idxs = inrange(E.tree, z, E.h)

    @assert length(idxs) > 0 "No neighbors found within bandwidth. Consider increasing h."

    acc = zero(E.fs[:,1])
    weight = 0.0

    @inbounds for i in idxs
        u = (coords(E)[i] .- z) ./ E.h
        w = epanechnikov(u)
        acc += w * E.fs[:, i]
        weight += w
    end

    return acc, weight
end

function marginal(E::KDEExpectation, z) 
    acc, weight = marginal_and_weight(E, z)
    acc / weight
end

weight(E::KDEExpectation, z) = marginal_and_weight(E, z)[2]

@inline function epanechnikov(u)
    w = 1.0
    @inbounds for j in eachindex(u)
        aj = abs(u[j])
        aj > 1 && return 0.0
        w *= (1 - aj^2)
    end
    return w
end


#### convenience    

function plot_b(eff::EffectiveSimulation; pullback=true, kwargs...)
    if length(coords(eff.kde)[1]) == 2
        return plot_b2(eff; kwargs...)
    end
    bs = []
    as = []
    for x in coords(eff.kde)
        b, A = b_and_sigma(eff, x)
        push!(bs, b |> only)
        push!(as, only(A)^2)
    end

    # collect all entries of the Vector{SVector} to a single Vector
    zs = mapreduce(only, vcat, coords(eff.kde))
    ix = sortperm(zs)


    b = plot(label="btilde(z)", xlabel="\\xi")
    pullback && plot!(zs[ix], eff.kde.fs[1, ix], label="\\xi*b")
    plot!(zs[ix], bs[ix], label="b~")

    A = plot(label="Atilde(z)", xlabel="\\xi")
    pullback && plot!(zs[ix], eff.kde.fs[2, ix], label="\\xi*A")
    plot!(zs[ix], as[ix], label="A~")
    plot(b, A)
end

function plot_b2(eff::EffectiveSimulation; scale=1e-5, kwargs...)
    zs = coords(eff.kde)

    n = length(zs)
    #n = 100
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    us = Vector{Float64}(undef, n)
    vs = Vector{Float64}(undef, n)

    for i in 1:n
        z = zs[i]
        b, _ = b_and_sigma(eff, z)
        b .*= scale

        xs[i] = z[1]
        ys[i] = z[2]
        us[i] = b[1]
        vs[i] = b[2]
    end

    #quiver(xs + us, ys + vs, quiver=(-us, -vs), aspect_ratio=:equal, arrow=(-0.1); kwargs...)
    quiver(xs, ys, quiver=(us, vs), aspect_ratio=:equal; kwargs...)
end

function plot_b_raw(eff::EffectiveSimulation; kwargs...)
    zs = coords(eff.kde)

    n = length(zs)
    #n = 100
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    us = Vector{Float64}(undef, n)
    vs = Vector{Float64}(undef, n)

    for i in 1:n
        z = zs[i]
        #b, _ = b_and_sigma(eff, z)
        b = eff.kde.fs[1:2, i]
        b ./= 200000

        xs[i] = z[1]
        ys[i] = z[2]
        us[i] = b[1]
        vs[i] = b[2]
    end

    quiver(xs, ys, quiver=(us, vs), aspect_ratio=:equal; kwargs...)
end

function plot_density(eff::EffectiveSimulation)
    zs = coords(eff.kde)
    ws = [weight(eff.kde, z) for z in zs]
    zs = mapreduce(only, vcat, zs)
    scatter(zs, ws, xlabel="\\xi", title="KDE")
end

function sortdata!(iso)
    idxs = sortperm(vec(chis(iso)))
    iso.data = iso.data[idxs]
    return iso
end

function test_effective(; ndata=1000, niter=1000)
    sim = OpenMMSimulation(integrator="brownian", friction=1000)
    data = SimulationData(sim, ndata, 1)
    iso = Iso(data)
    run!(iso, niter)
    eff = EffectiveSimulation(iso)

    plot_b(eff) |> display
    trajectory(eff) |> vec |> plot |> display

    (; iso, eff)
end

function script()
    data = SimulationData(sim, coords(iso.data), 10, featurizer=iso.data.featurizer)
    iso = Iso(data, model=ISOKANN.densenet(layers=[45,16,16,16,1], layernorm=true, activation=Flux.swish), opt=ISOKANN.AdamRegularized(), minibatch=500)
    run!(iso, 100_000)
    eff = EffectiveSimulation(iso, h=0.01)

end
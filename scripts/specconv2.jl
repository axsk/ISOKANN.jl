using LinearAlgebra
using SqraCore
using ISOKANN
using Plots
using StatsBase

function K_matrix(h=0.1, tau=0.1, beta=0.02)
    xs = -1.5:h:1
    ys = -0.5:h:2
    u = [ISOKANN.mueller_brown(x, y) for x in xs, y in ys]

    heatmap(@.(exp(-beta * u)))

    Q = sqra_grid(u, beta=beta)
    K = exp(collect(Q) .* tau)
    evals, evecs = eigen(K, sortby=x -> real(-x))
    evals, evecs = e.values, e.vectors
    return K, evals, evecs
end

const global K, evals, Evecs = K_matrix()

abstract type Model end

@kwdef mutable struct VecModel4 <: Model
    u::AbstractMatrix
    K::AbstractMatrix
    noise_store
    noise_koop
    eta  # learnrate in (0,1)
end

function VecModel(; m=1, noise_store=0, noise_koop=0, eta=1)
    u = randn(size(K, 1), m)
    VecModel4(u, K, noise_store, noise_koop, eta)
end

value(m::VecModel4) = copy(m.u)

function update!(m::Model, u)
    m.u = (1 - m.eta) * m.u + m.eta * u  # convex combination update
    m.u += randn(size(m.u)) * m.noise_store  # additive noise
end

#import ISOKANN: expectation
# NOTE: here we use cols for chi values, ISOKANN expects rows
expectation(model::VecModel4) = model.K * model.u + randn(size(model.u)) * model.noise_koop




function krylovspace(m, iter; update=nothing, maxcols=Inf)
    N, M = size(value(m))
    X = zeros(N, 0)
    Y = zeros(N, 0)
    for i in 1:iter
        x = value(m)
        y = expectation(m)
        X = lastcols([X x], maxcols)
        Y = lastcols([Y y], maxcols)
        if !isnothing(update)
            y = update(X, Y).vecs[:, 1:M] |> real
        end
        
        update!(m, y)
    end
    return X, Y
end

function lastcols(X, i)
    n, m = size(X)
    m <= i && return X
    return X[:, end-i+1:end]
end

## Ritz Methods

function rr_svd(X, Y)
    U, S, V = svd(X)
    Kh = U' * Y * V * inv(Diagonal(S))
    @show cond(Kh)
    vals, vecs = eigen(Kh, sortby=x -> -real(x))
    vecs = U * vecs
    return (; vals, vecs)
end

function rr_svd_i(X, Y)
    vals, vecs = rr_svd(Y, X)
    vals = 1 ./ vals[end:-1:1]
    vecs = vecs[:, end:-1:1]
    return (; vals, vecs)
end

function rr_svd_si(X, Y)
    vals, vecs = rr_svd(X - Y, X)
    vals = 1 .- 1 ./ vals
    return (; vals, vecs)
end

function rr_gev(X, Y)
    C = X' * X
    M = X' * Y
    vals, vecs = eigen(M, C, sortby=x -> -real(x))
    vecs = Y * vecs
    return (; vals, vecs)
end

function rr_cross(X, Y)
    Q, R = qr(Y)
    C = X' * X
    M = X' * Matrix(Q)
    T = R * (C \ M)
    vals, vecs = eigen(T, sortby=x -> -real(x))
    vecs = Q * vecs
    return (; vals, vecs)
end

function rr_qr(X, Y)
    Q, R = qr(X)
    Kh = Q[:, 1:size(R, 1)]' * Y * inv(R)  # apparatenly the line below is more stable
    Kh = (R' \ (Q[:, 1:size(R, 1)]' * Y)')'
    @show cond(Kh)
    vals, vecs = eigen(Kh, sortby=x -> -real(x))
    vecs = Q * e.vectors

    return (; vals, vecs)
end

## Tests


function compare_update(; iter=20, maxcols=20, reps=10, modelargs=(m=5, noise_store=0.01, noise_koop=0.0, eta=0.8))
    plot()
    for (nu, update) in enumerate([nothing, rr_cross, rr_gev, rr_svd, rr_svd_si,])
        for i in 1:reps
            X, Y = krylovspace(VecModel(; modelargs...), iter; update, maxcols=maxcols)
            plot!(real(rr_svd(X, Y).vals[1:5]), color=nu, label=i == 1 ? string(update) : nothing)
        end
    end
    plot!(real(evals[1:5]), color=:black, linewidth=1, label="truth")
end

function compare_rr_svd_si(; m=3, iter=10, compdim=3)
    m = VecModel(m=3)
    X, Y = krylovspace(m, iter)
    e1, v1 = rr_svd(X, Y)
    e2, v2 = rr_svd_si(X, Y)
    e0 = evals[1:compdim]
    e1 = e1[1:compdim]
    e2 = [1; e2][1:compdim]

    plot([e0 e1 e2] |> real, labels=["K" "svd" "svd_si"], ylims=(0.6, 1)) |> display

    v0 = evecs[:, 1:compdim]
    v1 = v1[:, 1:compdim]
    v2 = v2[:, 1:compdim]

    @show subspacedist(v0, v1)
    @show subspacedist(v0, v2)

end


# not symmetric, here x represents ground truth
function subspacedist(x, y)
    D = [1 - abs(dot(normalize(i), normalize(j))) for i in eachcol(x), j in eachcol(y)]
    d = minimum(D, dims=2)
    @show d
    return norm(d)
end

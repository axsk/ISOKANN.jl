using FLoops: @floop
using StaticArrays

# b + σu
function controlled_drift(D, xx, p, t, ::Val{n}, f::F, g::G, u::U) where {n,F,G,U}
    x = SVector{n}(@view xx[1:n])
    ux = u(x,t)
    gx = g(x,p,t)
    if isa(gx, Matrix)
        D[1:end-1] .= f(x,p,t) .+ gx * ux
    else
        D[1:end-1] .= f(x,p,t) .+ gx .* ux
    end
    D[end] = sum(abs2, ux) / 2
end

function controlled_noise(D, xx, p, t, ::Val{n}, g::G, u::U) where {n,G,U}
    x = SVector{n}(@view xx[1:n])
    gx = g(x,p,t)
    D .= 0
    if isa(gx, AbstractMatrix)
        D[1:n, 1:n] .= gx
    elseif isa(gx, AbstractVector)
        for i in 1:n
            D[i,i] = gx[i]
        end
    else
        for i in 1:n
            D[i,i] = gx
        end
    end
    D[end, 1:end-1] .= u(x,t)
end

# specialize on the dimension of the problem for SVector
GirsanovSDE(sde, u::F) where F = GirsanovSDE(sde, u, Val(length(sde.u0)))

""" Construct the SDE problem for Girsanov with control u """
function GirsanovSDE(sde, u::U, ::Val{n}) where {n, U}
    nrp = zeros(n+1, n+1)  # we could do with (n+1,n) but SROCK2 only takes square noise
    u0 = vcat(sde.u0, 0)   # append the girsanov dimension

    try
        u(sde.u0, 0)
    catch
        error("control `u` has wrong signature")
    end

    drift(D,x,p,t) = controlled_drift(D,x,p,t, Val(n), sde.f, sde.g, u)
    noise(D,x,p,t) = controlled_noise(D,x,p,t, Val(n), sde.g, u)

    return StochasticDiffEq.SDEProblem(drift, noise, u0, sde.tspan, sde.p; noise=sde.noise,
        noise_rate_prototype = nrp, sde.kwargs...)
end

## TODO: where do we use this?
function CompoundSDE(sde, u::U, v::Val{n} = Val(length(sde.u0))) where {n, U}
    nrp = zeros(n+1, n+1)
    u0 = vcat(sde.u0, 1.)

    f = sde.f
    g = sde.g

    function drift(D, xx, p, t)
        x = SVector{n}(@view xx[1:n])
        ux = u(x,t)
        gx = g(x,p,t)
        D[1:end-1] .= f(x,p,t) .+ gx .* ux
        D[end] = - xx[end] * sum(abs2, ux) / 2
    end

    function noise(D,xx,p,t)
        x = SVector{n}(@view xx[1:n])
        ux = u(x,t)
        gx = g(x,p,t)
        D .= 0
        for i in 1:n
            D[i,i] = gx
        end
        D[end, 1:end-1] .= -xx[end] * ux
    end

    return StochasticDiffEq.SDEProblem(drift, noise, u0, sde.tspan, sde.p;
        noise=sde.noise, noise_rate_prototype = nrp, sde.kwargs...)
end

nocontrol(x, t) = zero(x)

" convenience wrapper for obtaining X[end] and the Girsanov Weight"
function girsanovsample(cde, x0)
    u0 = vcat(x0, 0)
    sol=solve(cde; u0=u0)
    x = sol[end][1:end-1]
    w = exp(-sol[end][end])
    return x::Vector{Float64}, w::Float64
end

# TODO: maybe use DiffEq MC interface
function girsanovbatch(cde, xs, n)
    dim, nx = size(xs)
    ys ::Array{Float64, 3} = zeros(dim, nx, n)
    ws ::Array{Float64, 2} = zeros(nx, n)
    @floop for i in 1:nx, j in 1:n  # using @floop allows threaded iteration over i AND j
            ys[:, i, j], ws[i, j] = girsanovsample(cde, xs[:, i])
    end
    return ys, ws
end

" optcontrol(chis, Q, T, sigma, i)

optimal control u(x,t) = -∇log(Z)
for Z = Kχᵢ if Kχ = exp(Qt) χ.
Given it terms of the known generator Q"
function optcontrol(chis, Q, T, sigma, i)
    function u(x,t)
        dlogz = Zygote.gradient(x) do x  # this should prob. be ForwardDiff
            Z = exp(Q*(T-t)) * chis(x)
            log(Z[i])
        end
        return sigma' * dlogz
    end
    return u
end

""" K on {v₁, v₂} acts like a shift-scale, represented by `Shiftscale` """
struct Shiftscale
    a::Float64
    q::Float64
end

function Shiftscale(data::AbstractArray, T=1)
    a, b = extrema(data)
    lambda = b-a
    a = a/(1-lambda)
    q = log(lambda) / T
    return Shiftscale(a, q)
end

function (s::Shiftscale)(data, T=1)
    lambda = exp(T * s.q)
    return data .* lambda .+ s.a * (1-lambda)
end

function invert(s, data, T=1)
    lambda = exp(T*s.q)
    return (data .- s.a * (1-lambda)) ./ lambda
end

# TODO: check if this gives the same results as ociso
" optcontrol(chi, kchi::Array, T, sigma)

assume χ = a1 + bϕ with Kᵀϕ = λϕ = exp(qT)
then Kᵀχ = λχ + a(1-λ)1
given minima and maxima of Kᵀχ we can estimate λ and a
and therefore compute the optimal control for Kχ = E[χ]
u* = -σᵀ∇Φ = σᵀ∇log(Kχ) "

function optcontrol(chi::F, S::Shiftscale, T, sigma) where F
    function u(x,t)
        #x = SVector{length(x)}(x)
        dlogz = ForwardDiff.gradient(x) do x
            lambda = exp(S.q*(T-t))
            Z = lambda * first(chi(x)) + S.a*(1-lambda)
            if Z < 0
                @warn("negative log in control encountered")
                return 0.
            end
            log(Z)
        end #:: Vector{Float64}  # TODO: this should be inferred!
        return sigma' * dlogz
    end
    return u
end

# convenience wrapper using the original sde to extract noise and T
function optcontrol(model, S::Shiftscale, sde)
    sigma = sde.g(nothing, nothing, nothing)
    T = sde.tspan[end]
    optcontrol(model, S, T, sigma)
end

### Tests

function test_GirsanovSDE()
    sde = SDEProblem(Doublewell())
    cde = GirsanovSDE(sde, nocontrol)
    ys, ws = girsanovbatch(cde, rand(1,2), 3)
end

function test_optcontrol()
    sde = SDEProblem(Doublewell())
    model = fluxnet([1,3,3,1])
    u = optcontrol(model, Shiftscale(1,0), 1, 1)
    cde = GirsanovSDE(sde, u)
    ys, ws = girsanovbatch(cde, rand(1,2), 3)
end

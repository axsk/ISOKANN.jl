using Parameters
using StochasticDiffEq
import StochasticDiffEq: SDEProblem
import ForwardDiff

abstract type AbstractLangevin end
# interface methods: potential(l), sigma(l), dim(l)

function SDEProblem(l::AbstractLangevin, x0=randx0(l), T=1; dt=.01, alg=SROCK2(), kwargs...)
    drift(x,p,t) = force(l, x)
    noise(x,p,t) = sigma(l, x)
    StochasticDiffEq.SDEProblem(drift, noise, x0, T, alg=alg, dt=dt; kwargs...)
end

function force(l::AbstractLangevin, x)
    - ForwardDiff.gradient(x->potential(l, x), x)
end

##  Generic Diffusion in a potential
@with_kw struct Diffusion{T} <: AbstractLangevin
    potential::T
    dim::Int64=1
    σ::Float64=1.
end

potential(d::Diffusion, x) = d.potential(x)
sigma(l::Diffusion, x) = l.σ
dim(l::Diffusion) = l.dim
support(l::Diffusion) = repeat([-1.5 1.5], outer=[dim(l)]) :: Matrix{Float64}

randx0(l::Diffusion, n) = reduce(hcat, [randx0(l) for i in 1:n])
function randx0(l::Diffusion)
    s = support(l)
    x0 = rand(size(s, 1)) .* (s[:,2] .- s[:,1]) .+ s[:,1]
    return x0
end

doublewell(x) = ((x[1])^2 - 1) ^ 2

Doublewell(;kwargs...) = Diffusion(;potential=doublewell, kwargs...)

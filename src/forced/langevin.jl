using Parameters
import StochasticDiffEq  # does this incur invalidations?
import ForwardDiff

abstract type AbstractLangevin end
# interface methods: potential(l), sigma(l), dim(l)

function SDEProblem(l::AbstractLangevin, x0=randx0(l), T=tmax(l); dt=dt(l), alg=integrator(l), kwargs...)
    drift(x,p,t) = force(l, x)
    noise(x,p,t) = sigma(l, x)
    StochasticDiffEq.SDEProblem(drift, noise, x0, T, alg=alg, dt=dt; kwargs...)
end

function force(l::AbstractLangevin, x)
    - ForwardDiff.gradient(x->potential(l, x), x)
end

function propagate(l::AbstractLangevin, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, nx, ny)
    Threads.@threads for (i, j) in [(i, j) for i in 1:nx, j in 1:ny]
        ys[:, i, j] = solve_end(l; u0=x0[:, i])
    end
    return ys
end

function solve_end(l::AbstractLangevin; u0)
    StochasticDiffEq.solve(SDEProblem(l, u0))[end]
end

##  Generic Diffusion in a potential
@with_kw struct Diffusion{T} <: AbstractLangevin
    potential::T
    dim::Int64=1
    sigma::Vector{Float64} = [1.0]
end

tmax(d::AbstractLangevin) = tmax(d.potential)
tmax(d) = 1.0

integrator(d::AbstractLangevin) = StochasticDiffEq.SROCK2()

dt(d::AbstractLangevin) = dt(d.potential)
dt(d) = 0.01

potential(d::Diffusion, x) = d.potential(x)
sigma(l::Diffusion, x) = l.sigma
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

triplewell(x,y) = (3/4 * exp(-x^2 - (y-1/3)^2)
            - 3/4 * exp(-x^2 - (y-5/3)^2)
            - 5/4 * exp(-(x-1)^2 - y^2)
            - 5/4 * exp(-(x+1)^2 - y^2)
            + 1/20 * x^4 + 1/20 * (y-1/3)^4)

triplewell(x) = triplewell(x...)


Triplewell() = Diffusion(; potential=triplewell, dim=2, sigma=[1.0, 1.0])

defaultmodel(sim::AbstractLangevin; nout) = smallnet(dim(sim), nout)
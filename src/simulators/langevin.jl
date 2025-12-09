#using Parameters

# Abstract type defining the Overdamped Langevin dynamics
# mandatory interface methods: potential, sigma, dim, lagtime, dt
abstract type AbstractLangevin <: IsoSimulation end

featurizer(::AbstractLangevin) = identity
integrator(::AbstractLangevin) = StochasticDiffEq.EM()
defaultmodel(l::AbstractLangevin; n=dim(l), kwargs...) = smallnet(n; kwargs...)

function SDEProblem(l::AbstractLangevin, x0=randx0(l), T=lagtime(l); dt=dt(l), alg=integrator(l), kwargs...)
    drift(x,p,t) = force(l, x)
    noise(x,p,t) = sigma(l, x)
    StochasticDiffEq.SDEProblem(drift, noise, x0, T, alg=alg, dt=dt; kwargs...)
end

function force(l::AbstractLangevin, x)
    - ForwardDiff.gradient(x->potential(l, x), x)
end

function propagate(l::AbstractLangevin, x0::AbstractMatrix, ny)
    dim, nx = size(x0)
    ys = zeros(dim, ny, nx)
    Threads.@threads for (i, j) in [(i, j) for j in 1:ny, i in 1:nx]
        ys[:, j, i] = trajectory(l; x0=x0[:, i], saveat=lagtime(l))[:, end]
    end
    return ys
end

propagate(l::AbstractLangevin, x0::CuArray, ny) = gpu(propagate(l, cpu(x0), ny))

"""  trajectory(l::AbstractLangevin; T=lagtime(l), x0=randx0(l), save_start=false, saveat=lagtime(l), dt=dt(l))
generate a trajectory of length `T`, starting at x0 with stepsize `dt`, saving the output every `saveat` time. """
function trajectory(l::AbstractLangevin; T=lagtime(l), x0=randx0(l), save_start=false, saveat=dt(l), dt=dt(l))
    sde = SDEProblem(l, x0, T)
    sol = StochasticDiffEq.solve(sde; saveat, save_start, dt)
    xs = reduce(hcat, sol.u)
    return xs::Matrix
end

laggedtrajectory(l::AbstractLangevin, lags; lagtime=lagtime(l), T=lags*lagtime, kwargs...) = trajectory(l; T, saveat=lagtime, kwargs...)

# helper functions to generate random initial data

randx0(l::AbstractLangevin) = randx0(l, 1) |> vec

function randx0(l::AbstractLangevin, n)
    s = support(l)
    x0 = rand(size(s, 1), n) .* (s[:, 2] .- s[:, 1]) .+ s[:, 1]
    return x0
end

supportbox(l, x::Number) = supportbox(l, [-x, x])
supportbox(l, x::Vector) = repeat(x', outer=dim(l))
supportbox(l, x::Matrix) = x

# dispatch potential for higher dimensional arrays
potential(l::AbstractLangevin, x::AbstractArray) = mapslices(x -> potential(l, x), x; dims=1)

### Concrete implementations of the AbstractLangevin type

##  Generic Diffusion in a potential
@kwdef struct Diffusion{T} <: AbstractLangevin
    potential::T
    dim::Int64=1
    sigma::Union{Number,Vector,Matrix} = 1.0
    dt::Float64 = 0.01
    lagtime::Float64 = 1.0
    support::Union{Number,Vector,Matrix} = 1.0
end

potential(d::Diffusion, x::Vector) = d.potential(x)
dim(l::Diffusion) = l.dim
sigma(l::Diffusion, x) = l.sigma
dt(d::Diffusion) = d.dt
lagtime(d::Diffusion) = d.lagtime
support(l::Diffusion) = supportbox(l, l.support)

## Double- and Triplewell potentials implemented using the generic `Diffusion`

Doublewell(; kwargs...) = Diffusion(;
    potential=doublewell,
    support=1.5,
    kwargs...)

doublewell(x) = ((x[1])^2 - 1)^2


Triplewell(; kwargs...) = Diffusion(;
    potential=triplewell,
    dim=2,
    sigma=[1.0, 1.0],
    support=[-2 2; -1.5 2.5],
    kwargs...)


# as per 2006 - Philipp Metzner, Christof Schütte, and Eric Vanden-Eijnden
triplewell(x) = triplewell(x...)
triplewell(x,y) = (3 * exp(-x^2 - (y-1/3)^2)
            - 3 * exp(-x^2 - (y-5/3)^2)
            - 5 * exp(-(x-1)^2 - y^2)
            - 5 * exp(-(x+1)^2 - y^2)
            + 1/5 * x^4 + 1/5 * (y-1/3)^4
)


MuellerBrown(; kwargs...) = Diffusion(;
    potential=mueller_brown,
    dim=2,
    sigma=7.0,
    support=[-1.5 1; -0.5 2],
    dt=0.0001,
    lagtime=0.001,
    kwargs...)

mueller_brown(x::AbstractVector) = mueller_brown(x...)
function mueller_brown(x, y)
    -200 * exp(-1 * (x - 1)^2 + -10 * (y)^2) -
    100 * exp(-1 * (x)^2 + -10 * (y - 0.5)^2) -
    170 * exp(-6.5 * (x + 0.5)^2 + 11 * (x + 0.5) * (y - 1.5) + -6.5 * (y - 1.5)^2) +
    15 * exp(0.7 * (x + 1)^2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1)^2)
end

#=
## Example of a custom type for a specific process

struct CustomMuellerBrown <: AbstractLangevin end

integrator(m::MuellerBrown) = StochasticDiffEq.EM()
lagtime(m::MuellerBrown) = 0.01
dt(m::MuellerBrown) = 0.0001
dim(::MuellerBrown) = 2
potential(::MuellerBrown, x) = mueller_brown(x)
sigma(l::MuellerBrown, x) = 2.0
support(l::MuellerBrown) = [-1.5 1; -0.5 2]
=#

using StaticArrays

function mueller_brown_2d_vec(xy)
    A = @SVector [-200, -100, -170, 15]
    α = @SVector [-1, -1, -6.5, 0.7]
    β = @SVector[0, 0, 11, 0.6]
    γ = @SVector[-10, -10, -6.5, 0.7]
    a = @SVector[1, 0, -0.5, -1]
    b = @SVector[0, 0.5, 1.5, 1]

    x, y = xy

    potential = sum(1:4) do i
        @. A[i] * exp(α[i] * (x - a[i])^2 + β[i] * (x - a[i]) * (y - b[i]) + γ[i] * (y - b[i])^2)
    end

    return potential
end

# 10-25% few percent faster
function mueller_brown_2d_man(x, y)
    @. -200 * exp(-1 * (x - 1)^2 + -10 * (y)^2) -
       100 * exp(-1 * (x)^2 + -10 * (y - 0.5)^2) -
       170 * exp(-6.5 * (x + 0.5)^2 + 11 * (x + 0.5) * (y - 1.5) + -6.5 * (y - 1.5)^2) +
       15 * exp(0.7 * (x + 1)^2 + 0.6 * (x + 1) * (y - 1) + 0.7 * (y - 1)^2)
end

mueller_brown_2d_man(x::AbstractVector) = mueller_brown_2d_man(x[1], x[2])
mueller_brown_2d_man(x::AbstractMatrix) = mueller_brown_2d_man.(eachcol(x))
mueller_brown_2d_man(x::AbstractArray) = mueller_brown_2d_man(eachslice(x, dims=1)...)

mueller_brown = mueller_brown_2d_man

# Extends AbstractLangevin from forced/langevin.jl
@kwdef struct MuellerBrown <: AbstractLangevin
    tmax::Float64 = 0.01
    sigma::Float64 = 2.0
end

integrator(m::MuellerBrown) = StochasticDiffEq.EM()
tmax(m::MuellerBrown) = m.tmax
dt(m::MuellerBrown) = 0.0001
dim(::MuellerBrown) = 2
potential(d::MuellerBrown, x) = mueller_brown(x)
sigma(l::MuellerBrown, x) = l.sigma
support(l::MuellerBrown) = repeat([-1.5 1.5], outer=[dim(l)])::Matrix{Float64}
randx0(l::MuellerBrown, n) = reduce(hcat, [randx0(l) for i in 1:n])
function randx0(l::MuellerBrown)
    s = support(l)
    x0 = rand(size(s, 1)) .* (s[:, 2] .- s[:, 1]) .+ s[:, 1]
    return x0
end
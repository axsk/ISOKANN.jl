# the idea: take the derivative of the chi function as force field for an
# "average transition path"

using LinearAlgebra: normalize

function reactionforce(sim, x, chi, direction=1, orth=0.01)
    # setcoords
    f = force(sim, x)
    #@show norm(f_sys)

    f_chi = Zygote.gradient(x -> first(chi(x)), x)[1]
    magn = norm(f_chi)
    f_chi = normalize(direction * f_chi)

    ratio = dot(f_sys, f_chi)
    f_orth = (f_sys - f_chi * ratio)

    #@show norm(f_chi)
    #@show norm.([f_sys, f_orth])
    #@show dot(f_orth, f_chi)

    f = f_chi / magn + orth * f_orth
    #f_orth = f_sys - f_chi * dot(f_sys, f_chi) / dot(f_chi, f_chi)

    #f = -fc #+ fso * 0.001
    #f = direction * fc / norm(fc)^2 + fso * orth

    #@show norm(f)
    collect(f)
end



function energyminforce(sim, x)
    f = force(sim, x)
    return f / norm(f)
end

using OrdinaryDiffEq: ODEProblem

function energyminimization(sim, x0; solver=OrdinaryDiffEq.Tsit5(), t=0.1, dt=0.0001, kwargs...)
    s = OrdinaryDiffEq.solve(ODEProblem((x, p, t) -> energyminforce(sim, x), x0, t; dt, kwargs...), solver)
    return s.u[end]
end

""" compute the reactionpath for the simulation `sim` starting in `x0` along the gradient of the function provided by `chi`

Optional arguments:
extrapolate: walk beyond the interval limits
orth: factor for the force orthogonal to the `chi` gradient
solver: the ODE solver to use
dt: the timestep size
kwargs...: Keyword arguments passed to the `solve` method
"""
function reactionpath(sim, x0, chi; extrapolate=0.00, orth=0.01, solver=OrdinaryDiffEq.Euler(), dt=0.0001, kwargs...)

    t0 = chi(x0) |> first
    @show t0

    bw = OrdinaryDiffEq.solve(ODEProblem((x, p, t) -> reactionforce(sim, x, chi, -1, orth), x0, (0 - extrapolate, t0)), solver; dt, kwargs...)
    #@show bw.stats

    fw = OrdinaryDiffEq.solve(ODEProblem((x, p, t) -> reactionforce(sim, x, chi, 1, orth), x0, (t0, 1 + extrapolate)), solver; dt, kwargs...)
    #@show fw.stats

    fw = reduce(hcat, fw.u)
    bw = reduce(hcat, bw.u)

    u = hcat(bw[:, end:-1:1], fw)
end

function reactionpath(iso::IsoRun; minimize=true, kwargs...)
    xs, ys = iso.data
    x0 = xs[:, rand(1:size(xs, 2))]
    #println("minimizing energy")
    #x0 = energyminimization(iso.sim, x0)
    println("computing reaction path")
    reactionpath(iso.sim, x0, iso.model; kwargs...)
end


## now come my attempts to fit the orthogonal speed to the "average χ-speed",
## which however does not extist .. :D



# mean of the folded normal distribution
# https://en.wikipedia.org/wiki/Folded_normal_distribution
function folded_normal_mean(mu, sigma)
    sigma * sqrt(2 / pi) * exp(-mu^2 / (2 * sigma^2)) + mu * erf(mu / sqrt(2 * sigma^2))
end



""" what is the mean speed for a given force """
function mean_speed(velocity, sys, temp)
    mu = norm(velocity)
    v = velocity ./ mu
    sig = sigma(sys, temp)
    sig = norm(sig .* v)
    return folded_normal_mean(mu, sigma)
end

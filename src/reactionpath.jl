# the idea: take the derivative of the chi function as force field for an
# "average transition path"

using Zygote
using OrdinaryDiffEq

function reactionforce(sim, x, chi, direction=1, orth=0.01)
    # setcoords
    sys = setcoords(sim.sys, x)
    f_sys = Molly.accelerations(sys, find_neighbors(sys))  # TODO: scale with gamma?
    f_sys = ustrip(reduce(vcat, f_sys))

    #@show norm(f_sys)

    f_chi = Zygote.gradient(x->first(chi(x)), x)[1]
    magn = norm(f_chi)
    f_chi =  normalize(direction * f_chi)

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


function transitionpath(sim, x0, chi; orth = 0.01, solver=Rosenbrock23(), tmax = 1, kwargs...)
    direction = (first(chi(x0))) > 0.5 ? -1 : 1
    prob = ODEProblem((x,p,t)->velocity(sim, x, chi, direction, orth), x0, (0,tmax))
    sol = solve(prob, solver; kwargs...)
    #return reduce(hcat, sol.u)
end

export reactionpath
function reactionpath(sim, x0, chi; extrapolate=0.01, orth=0.01, solver=Euler(), kwargs...)

    t0 = chi(x0) |> first
    @show t0

    bw = solve(ODEProblem((x,p,t)->reactionforce(sim, x, chi, -1, orth), x0, (0-extrapolate, t0)), solver; kwargs...)
    #@show bw.stats

    fw = solve(ODEProblem((x,p,t)->reactionforce(sim, x, chi, 1, orth), x0, (t0, 1+extrapolate)), solver; kwargs...)
    #@show fw.stats

    fw = reduce(hcat, fw.u)
    bw = reduce(hcat, bw.u)

    u = hcat(bw[:, end:-1:1], fw)
end

using SpecialFunctions: erf

# mean of the folded normal distribution
# https://en.wikipedia.org/wiki/Folded_normal_distribution
function folded_normal_mean(mu, sigma)
    sigma * sqrt(2/pi) * exp(-mu^2/(2*sigma^2)) + mu * erf(mu/sqrt(2*sigma^2))
end

""" return the diagonal of the noise factor sigma of the brownian dynamics """
function sigma(sys::Molly.System, temp, gamma = 1/u"s")
    k = Unitful.k
    mass = Molly.masses(sys)
    σ = sqrt.(k * temp ./ mass *  gamma)
    repeat(σ, inner=3)
end

""" what is the mean speed for a given force """
function mean_speed(velocity, sys, temp)
    mu = norm(velocity)
    v = velocity ./ mu
    sig = sigma(sys, temp)
    sig = norm(sig.*v)
    return folded_normal_mean(mu, sigma)
end

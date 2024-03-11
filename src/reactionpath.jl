# the idea: take the derivative of the chi function as force field for an
# "average transition path"

using LinearAlgebra: normalize

function reactionforce(sim, x, chi, featurizer, direction=1, orth=0.01)
    # setcoords
    f_sys = force(sim, x)
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

function dchidx(iso, x=getcoords(iso.data)[1][:, 1])
    Zygote.gradient(x) do x
        iso.model(features(iso.data, x)) |> first
    end[1]
end

function test_dchidx()
    data = SimulationData(OpenMMSimulation(), 10, 2, ISOKANN.flatpairdists)
    iso = Iso2(data)
    dchidx(iso)
end


import Optim



"""
    reactionpath_minimum(iso::Iso2, x0, resolution=100)

Compute the reaction path by integrating ∇χ with orthogonal energy minimization.

# Arguments
- `iso::Iso2`: The isomer for which the reaction path minimum is to be computed.
- `x0`: The starting point for the reaction path computation.
- `resolution=100`: The number of steps to take along the reaction path.

# Returns
The reaction path minimum as an array of coordinates.
"""
function reactionpath_minimum(iso::Iso2, x0, resolution=100)
    @show chi = iso.model(iso.data.featurizer(x0)) |> only

    steps2 = floor(Int, resolution * (1 - chi))
    steps1 = floor(Int, resolution * chi)

    x1 = reactionintegrator(iso, steps=steps1, stepsize=1 / resolution, x0=x0, direction=-1)[:, end:-1:1]
    x2 = reactionintegrator(iso, steps=steps2, stepsize=1 / resolution, x0=x0, direction=1)

    hcat(x1, x2)
end

function reactionintegrator(iso::Iso2; x0, steps=10, stepsize=0.01, direction=1)

    U(x) = OpenMM.potential(iso.data.sim, x)

    # energy minimization direction, orthogonal to chi
    function dir(x)
        du = -OpenMM.force(iso.data.sim, x)
        dchi = dchidx(iso, x) |> normalize
        return du - dot(du, dchi) * dchi
    end

    x = x0
    xs = similar(x0, length(x0), steps)

    iter = 0
    p = Progress(steps)

    for i in 1:steps
        o = Optim.optimize(U, dir, x, Optim.LBFGS(; alphaguess=1e-8), Optim.Options(; x_tol=1e-8); inplace=false)
        x = Optim.minimizer(o)
        xs[:, i] .= x

        dchi = dchidx(iso, x) * direction
        x += stepsize / norm(dchi)^2 .* dchi  # adjust stepsize s.t. χ changes by `stepsize`

        chi = iso.model(iso.data.featurizer(x)) |> only

        iter += o.iterations
        ProgressMeter.next!(p; showvalues=[(:iter, iter), (:V, o.minimum), (:chi, chi)])
    end

    return xs
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

function reactionpath_ode(iso::Iso2; x0, minimize=false, kwargs...)

    if minimize
        #println("minimizing energy")
        #x0 = energyminimization(iso.sim, x0)
    end
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

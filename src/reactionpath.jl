import Optim
using LinearAlgebra: normalize

function dchidx(iso, x=getcoords(iso.data)[1][:, 1])
    Zygote.gradient(x) do x
        iso.model(features(iso.data, x)) |> first
    end[1]
end

"""
    reactionpath_minimum(iso::Iso2, x0; steps=100)

Compute the reaction path by integrating ∇χ with orthogonal energy minimization.

# Arguments
- `iso::Iso2`: The isomer for which the reaction path minimum is to be computed.
- `x0`: The starting point for the reaction path computation.
- `steps=100`: The number of steps to take along the reaction path.
"""
function reactionpath_minimum(iso::Iso2, x0=getcoords(iso.data)[1][:, 1]; steps=101)

    x = energyminimization_hyperplane(iso, x0)
    chi = chicoords(iso, x) |> only

    steps2 = floor(Int, steps * (1 - chi))
    steps1 = floor(Int, steps * chi)
    stepsize = 1 / steps

    x1 = reactionintegrator(iso, x0; steps=steps1, stepsize, direction=-1)[:, end:-1:1]
    x2 = reactionintegrator(iso, x0; steps=steps2, stepsize, direction=1)

    hcat(x1, x, x2)
end

function reactionintegrator(iso::Iso2, x0; steps=10, stepsize=0.01, direction=1)
    x = x0
    xs = similar(x0, length(x0), steps)
    @showprogress for i in 1:steps
        dchi = dchidx(iso, x)
        dchi .*= direction / norm(dchi)^2
        x += dchi .* stepsize
        x = energyminimization_hyperplane(iso, x)
        xs[:, i] .= x
    end
    return xs
end

function energyminimization_hyperplane(iso, x; alpha=1e-5, x_tol=1e-6)
    U(x) = OpenMM.potential(iso.data.sim, x)

    # energy minimization direction, orthogonal to chi
    function dir(x)
        du = -OpenMM.force(iso.data.sim, x)
        dchi = normalize(dchidx(iso, x))
        return du - dot(du, dchi) * dchi
    end

    o = Optim.optimize(U, dir, x, Optim.LBFGS(; alphaguess=alpha), Optim.Options(; x_tol); inplace=false)
    return o.minimizer
end


"""
    reactionpath_ode(iso, x0; steps=101, extrapolate=0, orth=0.01, solver=OrdinaryDiffEq.Tsit5(), dt=1e-3, kwargs...)

Compute the reaction path by integrating ∇χ as well as `orth` * F orthogonal to ∇χ where F is the original force field.


# Arguments
- `iso::Iso2`: The isomer for which the reaction path minimum is to be computed.
- `x0`: The starting point for the reaction path computation.
- `steps=100`: The number of steps to take along the reaction path.
- `minimize=false`: Whether to minimize the orthogonal to ∇χ before integration.
- `extrapolate=0`: How fast to extrapolate beyond χ 0 and 1.
- `orth=0.01`: The weight of the orthogonal force field.
- `solver=OrdinaryDiffEq.Tsit5()`: The solver to use for the ODE integration.
- `dt=1e-3`: The initial time step for the ODE integration.
"""
function reactionpath_ode(iso, x0; steps=101, minimize=false, extrapolate=0, orth=0.01, solver=OrdinaryDiffEq.Tsit5(), dt=1e-3, kwargs...)

    x0 = minimize ? energyminimization_hyperplane(iso, x0) : x0

    sim = iso.data.sim
    saveat = range(start=-extrapolate, stop=1 + extrapolate, length=steps)

    t0 = chicoords(iso, x0) |> only

    bw = OrdinaryDiffEq.solve(
        OrdinaryDiffEq.ODEProblem((x, p, t) -> reactionforce(iso, sim, x, -1, orth),
            x0, (0 - extrapolate, t0)), solver; saveat, dt, kwargs...)
    fw = OrdinaryDiffEq.solve(
        OrdinaryDiffEq.ODEProblem((x, p, t) -> reactionforce(iso, sim, x, 1, orth),
            x0, (t0, 1 + extrapolate)), solver; saveat, dt, kwargs...)

    return hcat(reduce.(hcat, (bw.u[end:-1:1], fw.u))...)
end

function reactionforce(iso, sim, x, direction, orth=1)
    f = force(sim, x)
    dchi = dchidx(iso, x)
    n2 = norm(dchi)^2

    f .-= dchi .* (dot(f, dchi) / n2)
    dchi ./= n2

    return direction * dchi + f * orth
end

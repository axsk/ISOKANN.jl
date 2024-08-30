using LinearAlgebra: normalize

function dchidx(iso, x)
    Zygote.gradient(x) do x
        chicoords(iso, x) |> myonly
    end |> only
end


## works on gpu as well
myonly(x) = only(x)
function myonly(x::CuArray)
    @assert length(x) == 1
    return sum(x)
end

"""
    reactionpath_minimum(iso::Iso, x0; steps=100)

Compute the reaction path by integrating ∇χ with orthogonal energy minimization.

# Arguments
- `iso::Iso`: The isomer for which the reaction path minimum is to be computed.
- `x0`: The starting point for the reaction path computation.
- `steps=100`: The number of steps to take along the reaction path.
"""
function reactionpath_minimum(iso::Iso, x0=randomcoords(cpu(iso)); steps=101, xtol=1e-3)

    iso = cpu(iso) # TODO: find another solution

    xs = energyminimization_projected(iso, x0; xtol)
    chi = chicoords(iso, xs) |> only

    steps2 = floor(Int, steps * (1 - chi))
    steps1 = floor(Int, steps * chi)
    stepsize = 1 / steps

    x1 = reactionintegrator(iso, xs; steps=steps1, stepsize, direction=-1, xtol)[:, end:-1:1]
    x2 = reactionintegrator(iso, xs; steps=steps2, stepsize, direction=1, xtol)

    hcat(x1, xs, x2)
end

function reactionintegrator(iso::Iso, x0; steps=10, stepsize=0.01, direction=1, xtol)
    x = x0
    xs = similar(x0, length(x0), steps)
    @showprogress for i in 1:steps
        dchi = dchidx(iso, x)
        dchi .*= direction / norm(dchi)^2
        x += dchi .* stepsize
        x = energyminimization_projected(iso, x; xtol)
        xs[:, i] .= x
    end
    return xs
end

energyminimization_projected(iso, x; kwargs...) = energyminimization_chilevel(iso, x; kwargs...)

function energyminimization_hyperplane(iso, x; alpha=1e-5, xtol=1e-6)
    U(x) = OpenMM.potential(iso.data.sim, x)

    # energy minimization direction, orthogonal to chi
    function dir(x)
        du = -OpenMM.force(iso.data.sim, x)
        dchi = normalize(dchidx(iso, x))
        return du - dot(du, dchi) * dchi
    end

    o = Optim.optimize(U, dir, x, Optim.LBFGS(; alphaguess=alpha), Optim.Options(; x_tol=xtol); inplace=false)
    return o.minimizer
end


"""
    reactionpath_ode(iso, x0; steps=101, extrapolate=0, orth=0.01, solver=OrdinaryDiffEq.Tsit5(), dt=1e-3, kwargs...)

Compute the reaction path by integrating ∇χ as well as `orth` * F orthogonal to ∇χ where F is the original force field.


# Arguments
- `iso::Iso`: The isomer for which the reaction path minimum is to be computed.
- `x0`: The starting point for the reaction path computation.
- `steps=100`: The number of steps to take along the reaction path.
- `minimize=false`: Whether to minimize the orthogonal to ∇χ before integration.
- `extrapolate=0`: How fast to extrapolate beyond χ 0 and 1.
- `orth=0.01`: The weight of the orthogonal force field.
- `solver=OrdinaryDiffEq.Tsit5()`: The solver to use for the ODE integration.
- `dt=1e-3`: The initial time step for the ODE integration.
"""
function reactionpath_ode(iso, x0; steps=101, minimize=false, extrapolate=0, orth=0.01, solver=OrdinaryDiffEq.Tsit5(), dt=1e-3, kwargs...)

    iso = cpu(iso) # TODO: find another solution

    x0 = minimize ? energyminimization_hyperplane(iso, x0, xtol=1e-4) : x0

    sim = iso.data.sim
    saveat = range(start=-extrapolate, stop=1 + extrapolate, length=steps)

    t0 = chicoords(iso, x0) |> only

    bw = OrdinaryDiffEq.solve(
        OrdinaryDiffEq.ODEProblem((x, p, t) -> reactionforce(iso, sim, x, -1, orth),
            x0, (0 - extrapolate, t0)), solver; saveat, dt, kwargs...)
    fw = OrdinaryDiffEq.solve(
        OrdinaryDiffEq.ODEProblem((x, p, t) -> reactionforce(iso, sim, x, 1, orth),
            x0, (t0, 1 + extrapolate)), solver; saveat, dt, kwargs...)

    #return bw, fw

    return hcat(reduce.(hcat, (bw.u[end:-1:1], fw.u))...)
end

function randomcoords(iso)
    c = getcoords(iso.data)
    n = size(c, 2)
    c[:, rand(1:n)]
end


"""
    reactionforce(iso, sim, x, direction, orth=1)


Compute the vector `f` with colinear component to dchi/dx such that dchi/dx * f = 1
and orth*forcefield in the orthogonal space
"""
function reactionforce(iso, sim, x, direction, orth=1)
    f = force(sim, x)
    #@show f[1]
    dchi = dchidx(iso, x)
    n2 = norm(dchi)^2

    # remove component of f pointing into dchi
    f .-= dchi .* (dot(f, dchi) / n2)

    @. f = f * orth + (direction / n2) * dchi
    return f
end

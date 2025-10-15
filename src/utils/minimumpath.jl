using LinearAlgebra: normalize

function dchidx(iso, x)
    Zygote.gradient(x) do x
        chicoords(iso, x) |> myonly
    end |> only
end

function dchidfeat(iso, feat)
    Zygote.gradient(feat) do x
        iso.model(x) |> myonly
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
function reactionpath_minimum(iso::Iso, x0=randomcoords(iso); steps=101, xtol=1e-3, extrasteps=0)

    #iso = cpu(iso) # TODO: find another solution

    xs = energyminimization_projected(iso, x0; xtol) #|> gpu
    chi = chicoords(iso, xs) |> myonly

    steps2 = max(floor(Int, steps * (1 - chi)) + extrasteps, 0)
    steps1 = max(floor(Int, steps * chi) + extrasteps, 0)
    stepsize = 1 / steps

    x1 = reactionintegrator(iso, xs; steps=steps1, stepsize, direction=-1, xtol)[:, end:-1:1]
    x2 = reactionintegrator(iso, xs; steps=steps2, stepsize, direction=1, xtol)

    hcat(x1, xs, x2)
end

function reactionintegrator(iso::Iso, x0; steps=10, stepsize=0.01, direction=1, xtol)
    x = copy(x0)
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
    c = coords(iso.data)
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

""" 
    energyminimization_chilevel(iso, x0; f_reltol=1e-3, alphaguess=1e-5, iterations=20, show_trace=false, skipwater=false, algorithm=Optim.GradientDescent, xtol=nothing)

Local energy minimization on the current levelset of the chi function
"""
function energyminimization_chilevel(iso, x0; skipwater=false, kwargs...)
    sim = iso.data.sim
    x = copy(x0) .|> Float64

    chi(x) = chicoords(iso, reshape(x, :, 1)) |> myonly

    U(x) = OpenMM.potential(sim, x)

    function dU(x)
        f = -OpenMM.force(sim, x)
        (skipwater && zerowater!(sim, f))
        f
    end

    return minimize_levelset(x, chi, U, dU; kwargs...)
end

function minimize_levelset(x0, f, U, dU; f_reltol=1e-3, alphaguess=1e-5, iterations=20, show_trace=false, algorithm=Optim.GradientDescent, xtol=nothing)
    manifold = Levelset(f, f(x0))
    linesearch = Optim.LineSearches.HagerZhang(alphamax=alphaguess)
    alg = algorithm(; linesearch, alphaguess, manifold)


    o = Optim.optimize(U, dU, x0, alg, Optim.Options(; iterations, f_reltol, show_trace,); inplace=false)
    return o.minimizer
end

function zerowater!(sim, x)
    inds = map(sim.pysim.topology.atoms()) do a
        a.residue.name == "HOH"
    end
    x = reshape(x, 3, :)
    x[:, inds] .= 0
    vec(x)
end

struct Levelset{F,T} <: Optim.Manifold
    f::F
    target::T
end

function Optim.project_tangent!(M::Levelset, g, x)
    @assert !any(isnan.(g))
    @assert !any(isnan.(x))
    #replace!(g, NaN => 0)
    u = Zygote.gradient(M.f, x) |> myonly
    u ./= norm(u)
    g .-= dot(g, u) * u
end

function Optim.retract!(M::Levelset, x)
    @assert !any(isnan.(x))
    g = Zygote.withgradient(M.f, x)
    u = g.grad |> myonly
    h = M.target - g.val
    x .+= h .* u ./ (norm(u)^2)
end

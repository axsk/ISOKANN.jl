
"""
    addextrapolates!(iso, n, stepsize=0.01, steps=10)

Sample new data starting points obtained by extrapolating the chi function beyond
the current extrema and attach it to the `iso` objects data.

Samples `n` points at the lower and upper end each, resulting in 2n new points.
`step`` is the magnitude of chi-value-change per step and `steps`` is the number of steps to take.
E.g. 10 steps of stepsize 0.01 result in a change in chi of about 0.1.

The obtained data is filtered such that unstable simulations should be removed,
which may result in less then 2n points being added.
"""
function addextrapolates!(iso, n; stepsize=0.01, steps=1)
    xs = extrapolate(iso, n, stepsize, steps)
    nd = SimulationData(iso.data.sim, xs, nk(iso.data))
    iso.data = merge(iso.data, nd)
    return
end

"""
    extrapolate(iso, n, stepsize=0.1, steps=1, minimize=true)

Take the `n` most extreme points of the chi-function of the `iso` object and
extrapolate them by `stepsize` for `steps` steps beyond their extrema,
resulting in 2n new points.
If `minimize` is true, the new points are energy minimized.
"""
function extrapolate(iso, n, stepsize=0.1, steps=1, minimize=true)
    data = iso.data
    model = iso.model
    coords = flatend(data.coords[2])
    features = flatend(data.features[2])
    xs = Vector{eltype(coords)}[]
    skips = 0

    p = sortperm(model(features) |> vec) |> cpu

    for (p, dir, N) in [(p, -1, n), (reverse(p), 1, 2 * n)]
        for i in p
            try
                x = extrapolate(data, model, coords[:, i], dir * stepsize, steps)
                minimize && (x = energyminimization_chilevel(iso, x))
                push!(xs, x)
            catch e
                if isa(e, PyCall.PyError) || isa(e, DomainError)
                    skips += 1
                    continue
                end
                rethrow(e)
            end
            length(xs) == N && break
        end
    end

    skips > 0 && @warn("extrapolate: skipped $skips extrapolates due to instabilities")
    xs = reduce(hcat, xs)
    return xs
end

function extrapolate(d, model, x::AbstractVector, step, steps)
    x = copy(x)
    for _ in 1:steps
        grad = dchidx(d, model, x)
        x .+= grad ./ norm(grad)^2 .* step
        #@show model(features(d,x))
    end
    return x
end

function energyminimization_chilevel(iso, x0; f_tol=1e-1, alphaguess=1e-6, iterations=100, show_trace=true)
    sim = iso.data.sim
    x = copy(x0)

    chi(x) = myonly(chicoords(iso, x))
    chilevel = Levelset(chi, chi(x0))

    U(x) = OpenMM.potential(sim, x)
    dU(x) = -OpenMM.force(sim, x)

    o = Optim.optimize(U, dU, x, Optim.LBFGS(; alphaguess, manifold=chilevel), Optim.Options(; iterations, f_tol, show_trace); inplace=false)
    return o.minimizer
end

struct Levelset{F,T} <: Optim.Manifold
    f::F
    target::T
end

function Optim.project_tangent!(M::Levelset, g, x)
    u = Zygote.gradient(M.f, x) |> only
    u ./= norm(u)
    g .-= dot(g, u) * u
end

function Optim.retract!(M::Levelset, x)
    g = Zygote.withgradient(M.f, x)
    u = g.grad |> only
    h = M.target - g.val
    x .+= h .* u ./ (norm(u)^2)
end

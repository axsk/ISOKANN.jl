
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
    n == 0 && return
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
function extrapolate(iso, n::Integer, stepsize=0.1, steps=1, minimize=true, maxskips=10)
    data = iso.data
    model = iso.model
    coords = flattenlast(data.coords[2])
    features = flattenlast(data.features[2])
    xs = Vector{eltype(coords)}[]
    skips = 0

    p = sortperm(model(features) |> vec) |> cpu

    for (p, dir, N) in [(p, -1, n), (reverse(p), 1, 2 * n)]
        for i in p
            skips > maxskips && break
            try
                x = extrapolate(iso, coords[:, i], dir * stepsize, steps)
                minimize && (x = energyminimization_chilevel(iso, x))
                #=
                if hasfield(typeof(data.sim), :momenta) && data.sim.momenta
                    x = reshape(x, :, 2)
                    #x[:, 2] .= 0
                    x = vec(x)
                end
                =#
                #&& ISOKANN.OpenMM.set_random_velocities!(data.sim, x)
                push!(xs, x)
            catch e
                if isa(e, PyCall.PyError) || isa(e, DomainError) || isa(e, AssertionError)
                    @show e
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

function extrapolate(iso, x::AbstractVector, step, steps)
    x = copy(x)
    for _ in 1:steps
        grad = dchidx(iso, x)
        x .+= grad ./ norm(grad)^2 .* step
    end
    return x
end

global trace = []

function energyminimization_chilevel(iso, x0; f_tol=1e-3, alphaguess=1e-5, iterations=20, show_trace=false, skipwater=false, algorithm=Optim.GradientDescent)
    sim = iso.data.sim
    x = copy(x0) .|> Float64

    chi(x) = myonly(chicoords(iso, x))
    manifold = Levelset(chi, chi(x0))


    global trace = [x0]
    U(x) = begin
        push!(trace, x)
    end
    dU(x) = begin
        push!(trace, x)
        f = -OpenMM.force(sim, x)
        (skipwater && zerowater!(sim, f))
        f
    end


    linesearch = Optim.LineSearches.HagerZhang(alphamax=alphaguess)
    alg = algorithm(; linesearch, alphaguess, manifold)


    o = Optim.optimize(U, dU, x, alg, Optim.Options(; iterations, f_tol, show_trace,); inplace=false)
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
    u = Zygote.gradient(M.f, x) |> only
    u ./= norm(u)
    g .-= dot(g, u) * u
end

function Optim.retract!(M::Levelset, x)
    @assert !any(isnan.(x))
    g = Zygote.withgradient(M.f, x)
    u = g.grad |> only
    h = M.target - g.val
    x .+= h .* u ./ (norm(u)^2)
end

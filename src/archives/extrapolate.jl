
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
    addcoords!(iso, xs)
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

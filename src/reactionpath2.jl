# maximum likelihood path on given data

function reactive_path(xi::AbstractVector, coords::AbstractMatrix; sigma, maxjump=1, method=QuantilePath(0.05), normalize=false, sortincreasing=true)
    from, to = fromto(method, xi)

    nco = normalize ? coords ./ norm(coords, Inf) : coords

    ids = shortestchain(nco, xi, from, to; sigma, maxjump)
    if sortincreasing && !isincreasing(ids)
        ids = reverse(ids)
    end

    path = coords[:, ids]
    return ids, path
end

reactive_path(iso::Iso; kwargs...) = reactive_path(chis(iso) |> vec, getcoords(iso.data); kwargs...)

# heuristic whether a sequence is increasing
isincreasing(x) = sum(diff(x) .> 0) > length(x) / 2

# find path from s1 to s2
struct FromToPath
    s1::Int
    s2::Int
end

# find a path between the top and bottom q percentile of xi
struct QuantilePath
    q::Float64
end

# find a path between the first and last frame
struct FullPath end

# find the path between minimal and maximal chi value
struct MaxPath end

function fromto(q::QuantilePath, xi)
    q = q.q
    from = findall(xi .< quantile(xi, q))
    to = findall(xi .> quantile(xi, 1 - q))
    return from, to
end

fromto(f::FromToPath, xi) = (f.s1, f.s2)
fromto(::FullPath, xi) = (1, length(xi))
fromto(::MaxPath, xi) = (argmin(xi), argmax(xi))

using Distances: pairwise, Euclidean

# compute the shortest chain through the samples xs with reaction coordinate xi
function shortestchain(xs, xi, from, to; sigma, maxjump)
    dxs = pairdist(xs)
    logp = finite_dimensional_distribution(dxs, xi, sigma, size(xs, 1), maxjump)
    ids = shortestpath(-logp, from, to)
    return ids
end

# compute the shortest through the matrix A from ind s1 to s2
function shortestpath(A::AbstractMatrix, s1::AbstractVector{<:Integer}, s2::AbstractVector{<:Integer})
    A = sparse(replace(A, Inf => 0))
    g = SimpleWeightedDiGraph(A)
    bf = Graphs.bellman_ford_shortest_paths(g, s1, A)
    j = s2[bf.dists[s2]|>argmin]
    return Graphs.enumerate_paths(bf, j)
end

shortestpath(A::AbstractMatrix, s1::Integer, s2::Integer) = shortestpath(A, [s1], [s2])

function shortestpath(A::CuArray, s1::Integer, s2::Integer)
    _, par = bellmanford_parallel(A, s1)
    enumerate_path(par, s2)
end

# path probabilities c.f. https://en.wikipedia.org/wiki/Onsager-Machlup_function
# this is the dense "vectorized" implementation which is slightly faster on cpu but works and much faster on gpu
function finite_dimensional_distribution(dxs, xi, sigma, dim, maxjump)
    dt = xi' .- xi
    map(dxs, dt) do dx, dt
        0 < dt < maxjump || return -Inf
        v = dx / dt
        L = 1 / 2 * (v / sigma)^2
        s = (-dim / 2) * Base.log(2 * pi * sigma^2 * dt)
        logp = (s - L * dt)
    end
end

### VISUALIZATION

function plot_reactive_path(ids, xi)
    xi = Flux.cpu(xi)
    plot(xi)
    scatter!(ids, xi[ids])
    plot!(ids, xi[ids])
    plot(plot!(), plot(xi[ids]))
end

# visualization of what shortestchain does in 1d with random data
function visualize_shortestpath(; n=1000, sigma=0.1)
    xs = rand(1, n)
    xi = rand(n)
    xi[1] = 0
    xi[end] = 1
    ids = shortestchain(xs, xi, 1, n; sigma)
    scatter(xi, xs[1, :])
    plot!(xi[ids], xs[1, ids], xlabel="t", ylabel="x", label="reaction path")
    plot!(yticks=[0, 1], xticks=[0, 1], legend=false)
end


"""
    save_reactive_path(iso::Iso, coords::AbstractMatrix;
        sigma=1,
        out="out/reactive_path.pdb",
        source,
        kwargs...)

Extract and save the reactive path of a given `iso`.

Computes the maximum likelihood path with parameter `sigma` along the given data points,
aligns it and saves it to the `out` path.

# Arguments
- `iso::Iso`: The isomer for which the reactive path is computed.
- `coords::AbstractMatrix`: The coordinates corresponding to the samples in `iso`
- `sigma=1`: The standard deviation used for the reactive path calculation.
- `out="out/reactive_path.pdb"`: The output file path for saving the reactive path.
- `source`: The source .pdb file
= `kwargs...`: additional parameters passed to `reactive_path`.

# Returns
- `ids`: The IDs of the reactive path.

"""
function save_reactive_path(iso::Iso, coords::AbstractMatrix=getcoords(iso.data) |> cpu;
    sigma=1,
    maxjump=1,
    out="out/reactive_path.pdb",
    source=pdb(iso.data),
    kwargs...)

    chi = chis(iso) |> vec |> cpu
    ids, path = reactive_path(chi, coords; sigma, maxjump, kwargs...)
    if length(ids) == 0
        @warn "The computed reactive path is empty. Try adjusting the `sigma` parameter."
        return ids
    end
    plot_reactive_path(ids, chi) |> display
    path = aligntrajectory(path)
    path = centercoords(path)
    println("saving reactive path of length $(length(ids)) to $out")
    mkpath(dirname(out))
    save_trajectory(out, path, top=source)
    return ids
end

### GPU Bellman Ford

function enumerate_path(par, s2)
    par = cpu(par)
    c = s2
    path = [s2]
    while true
        c = par[c]
        c == 0 && break
        push!(path, c)
    end
    return reverse(path)
end

function bellmanford_parallel(A, source)
    n = size(A, 1)
    A = A + I * Inf
    d = similar(A, n) .= typemax(eltype(A))
    CUDA.@allowscalar d[source] = 0
    par = similar(d, Int) .= 0
    new = similar(d, Bool) .= false
    next = similar(A)
    for _ in 1:n
        next .= d .+ A
        dd, pp = findmin(next, dims=1)
        new .= vec(dd) .+ 1e-8 .< d
        any(new) > 0 || break
        d[new] = dd[new]
        par[new] = map(x -> x[1], pp[new])
    end
    return d, par
end

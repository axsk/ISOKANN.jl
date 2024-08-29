# maximum likelihood path on given data

function reactive_path(xi::AbstractVector, coords::AbstractMatrix; sigma, maxjump=1, method=QuantilePath(0.05), normalize=false, sortincreasing=true)
    xi = cpu(xi)
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
    dxs = pairwise(Euclidean(), xs, dims=2) ## TODO: gpu support
    logp = finite_dimensional_distribution(dxs, xi, sigma, size(xs, 1), maxjump)
    ids = shortestpath(-logp, from, to)
    return ids
end

# compute the shortest through the matrix A from ind s1 to s2
function shortestpath(A::AbstractMatrix, s1::AbstractVector{<:Integer}, s2::AbstractVector{<:Integer})
    g = SimpleWeightedDiGraph(A)
    bf = Graphs.bellman_ford_shortest_paths(g, s1, A)
    j = s2[bf.dists[s2]|>argmin]
    return Graphs.enumerate_paths(bf, j)
end

shortestpath(A::AbstractMatrix, s1::Integer, s2::Integer) = shortestpath(A, [s1], [s2])

# path probabilities c.f. https://en.wikipedia.org/wiki/Onsager-Machlup_function
function finite_dimensional_distribution(dxs, xi, sigma, dim, maxjump)
    logp = zero(dxs)
    for c in CartesianIndices(dxs)
        i, j = Tuple(c)
        dt = xi[j] - xi[i]
        if 0 < dt < maxjump
            v = dxs[i, j] / dt
            L = 1 / 2 * (v / sigma)^2
            s = (-dim / 2) * Base.log(2 * pi * sigma^2 * dt)
            logp[c] = (s - L * dt)
        else
            logp[c] = -Inf
        end
    end
    return logp
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


function SimpleDiGraph_fast(A::SparseMatrixCSC{U}, T = typeof(A).parameters[2]) where {U<:Real}
    dima, dimb = size(A)
    isequal(dima, dimb) ||
        throw(ArgumentError("Adjacency / distance matrices must be square"))

    badjlist = Vector{T}[]
    for i in 1:dima
        ni = A.rowval[A.colptr[i]:A.colptr[i+1]-1]
        push!(badjlist, ni)
    end

    At = sparse(A')
    fadjlist = Vector{T}[]
    for i in 1:dima
        ni = At.rowval[At.colptr[i]:At.colptr[i+1]-1]
        push!(fadjlist, ni)
    end

    return SimpleDiGraph(length(A.nzval), fadjlist, badjlist)
end


# implementation of own graph type using sparse arrays as representation
# about as fast as SimpleDiGraph, but does not require the compilation of it


using Graphs
using SparseArrays

struct SAGraph3{S<:SparseArrays.AbstractSparseMatrixCSC} <: Graphs.AbstractGraph{Int}
    A::S
    At::SparseMatrixCSC{Nothing,Int}

    function SAGraph3(a)
        A = sparse(a)
        i, j, _ = findnz(A)
        n, m = size(A)
        @assert n == m
        At = sparse(j, i, fill(nothing, length(i)), n, n)
        new{typeof(A)}(A, At)
    end
end

MySAGraph = SAGraph3

Graphs.ne(s::MySAGraph) = nnz(s.A)
Graphs.nv(s::MySAGraph) = s.A.n
Graphs.vertices(s::MySAGraph) = 1:Graphs.nv(s)
Graphs.is_directed(s::MySAGraph) = true
Graphs.outneighbors(s::MySAGraph, v) = @view s.At.rowval[s.At.colptr[v]:s.At.colptr[v+1]-1]
Graphs.inneighbors(s::MySAGraph, v) = @view s.A.rowval[s.A.colptr[v]:s.A.colptr[v+1]-1]
Graphs.has_vertex(s::MySAGraph, v) = v in Graphs.vertices(s)
Graphs.has_edge(s::MySAGraph, a, b) = s.A[a, b] != 0
Graphs.edgetype(s::MySAGraph) = typeof(Graphs.edges(s) |> first)
Graphs.edges(s::MySAGraph) = begin
    i, j, _ = findnz(s.At)
    zip(j, i)
end
Graphs.weights(s::MySAGraph) = s.A

function benchmark_bf_graphs(A=sprand(10_000, 10_000, 0.1))
    @time sim = SimpleDiGraph(A)
    @time sta = StaticDiGraph(sim)
    @time sag = MySAGraph(A)

    r1 = @time bellman_ford_shortest_paths(sim, 1, A)
    r2 = @time bellman_ford_shortest_paths(sta, 1, A)
    r3 = @time bellman_ford_shortest_paths(sag, 1, A)
    r1, r2, r3
end

=#
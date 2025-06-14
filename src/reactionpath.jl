"""
    save_reactive_path(iso::Iso,
        coords::AbstractMatrix=coords(iso.data) |> cpu;
        sigma=1,
        maxjump=1,
        out="out/reactive_path.pdb",
        source=pdbfile(iso.data),
        kwargs...)

Extract and save the reactive path of a given `iso`.

Computes the maximum likelihood path with parameter `sigma` along the given data points,
aligns it and saves it to the `out` path.

See also `reactive_path`.

# Arguments
- `iso::Iso`: The Iso for which the reactive path is computed.
- `out="out/reactive_path.pdb"`: The output file path for saving the reactive path.
- `source`: The source .pdb file providing the topology
- `kwargs...`: additional parameters passed to `reactive_path`.

# Returns
- `ids`: The IDs of the reactive path.

"""
function save_reactive_path(iso::Iso, coords::AbstractMatrix=coords(iso.data) |> cpu;
    sigma=1,
    maxjump=1,
    out="out/reactive_path.pdb",
    source=pdbfile(iso.data),
    chi=chicoords(iso, coords) |> vec |> cpu,
    weights=OpenMM.masses(iso.data.sim),
    kwargs...)

    ids = reactive_path(chi, coords; sigma, maxjump, weights, kwargs...)
    if length(ids) == 0
        @warn "The computed reactive path is empty. Try adjusting the `sigma` parameter."
        return ids
    end
    plot_reactive_path(ids, chi) |> display
    path = aligntrajectory(coords[:, ids]; weights)
    println("saving reactive path of length $(length(ids)) to $out")
    mkpath(dirname(out))
    save_trajectory(out, path, top=source)
    return ids
end

""" reactive_path(xi::AbstractVector, coords::AbstractMatrix; sigma, minjump=0, maxjump=1, method=QuantilePath(0.05), normalize=false, sortincreasing=true)

Find the maximum likelihood path (under the model of brownion motion with noise `sigma`) through `coords` with times `xi`.
Supports either CPU or GPU arrays.

# Arguments
- `coords`:  (ndim x npoints) matrix of coordinates.
- `xi`: time coordinate of the npoints points
- `sigma`: spatial noise strength of the model.
- `minjump`, `maxjump`: lower and upper bound to the jump in time `xi` along the path. Tighter bounds reduce the computational cost.
- `method`: either `FromToPath`,  `QuantilePath`, `FullPath` or `MaxPath`, specifying the end points of the path
- `normalize`: whether to normalize all `coords` first
- `sortincreasing`: return the path from lower to higher `xi` values
"""
function reactive_path(xi::AbstractVector, coords::AbstractMatrix; method=QuantilePath(0.05), normalize=false, sortincreasing=true, kwargs...)
    from, to = fromto(method, xi)
    nco = normalize ? coords ./ norm(coords, Inf) : coords
    ids = shortestchain(nco, xi, from, to; kwargs...)
    sortincreasing && !isincreasing(ids) && reverse!(ids)
    return ids
end

reactive_path(iso::Iso; kwargs...) = reactive_path(chis(iso) |> vec |> cpu, coords(iso.data); kwargs...)

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

# compute the shortest chain through the samples xs with reaction coordinate xi
function shortestchain(xs, xi, from, to; sigma, minjump, maxjump, weights, gpu=CUDA.has_cuda_gpu())
    @assert size(xs, 2) == length(xi)
    gpu && (xs = cu(xs); xi = cu(xi); weights = (cu(weights)))
    println("Computing pairwise distances")
    dt = xi' .- xi
    mask = minjump .<= dt .<= maxjump
    @time dxs = pairwise_aligned_rmsd(xs; mask, weights)
    dim = size(xs, 1)
    logp = fin_dim_loglikelihood.(dxs, dt, sigma, dim, minjump, maxjump)
    println("Computing shortest path")
    @time ids = shortestpath(-logp, from, to)
    return ids
end

# path probabilities c.f. https://en.wikipedia.org/wiki/Onsager-Machlup_function
function fin_dim_loglikelihood(dx, dt, sigma, dim, minjump, maxjump)
    minjump <= dt <= maxjump && dt > 0 || return -Inf
    v = dx / dt
    L = 1 / 2 * (v / sigma)^2
    s = (-dim / 2) * Base.log(2 * pi * sigma^2 * dt)
    return logp = (s - L * dt)
end


shortestpath(A::AbstractMatrix, s1::Integer, s2::Integer) = shortestpath(A, [s1], [s2])

# compute the shortest through the matrix A from ind s1 to s2
function shortestpath(A::AbstractMatrix, s1::AbstractVector{<:Integer}, s2::AbstractVector{<:Integer})
    A = sparse(replace(A, Inf => 0)) |> SparseArrays.dropzeros
    g = SimpleWeightedDiGraph(A)
    bf = Graphs.bellman_ford_shortest_paths(g, s1, A)
    j = s2[bf.dists[s2]|>argmin]
    return Graphs.enumerate_paths(bf, j)
end

function shortestpath(A::Union{CuArray,CUDA.CUSPARSE.CuSparseMatrixCSC}, s1::AbstractVector{<:Integer}, s2::AbstractVector{<:Integer})
    d, par = bellmanford(A, s1)
    j = s2[d[s2]|>argmin]
    enumerate_path(par, j)
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

bellmanford(s, source::Int) = bellmanford(s, [source])

function bellmanford(A::DenseMatrix, source::AbstractVector)
    n = size(A, 1)
    #A[diagind(A)] .= Inf
    d = similar(A, n) .= typemax(eltype(A))
    d[source] .= 0
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

using CUDA
import SparseArrays

# CUDA implementation of Bellman Ford for sparse matrices
# each thread corresponds to one column
function bellmanford(s::CUDA.CUSPARSE.AbstractCuSparseMatrix, source::AbstractVector)
    colptr = s.colPtr
    rowval = s.rowVal
    val = s.nzVal
    d = size(s, 1)
    p = similar(rowval, d) .= 0
    dists = similar(val, d) .= Inf
    changed = cu([false])
    dists[source] .= 0

    kernel = @cuda name = "bellman ford iteration" launch = false bf_cuda_sparse(dists, p, colptr, rowval, val, changed)
    config = launch_configuration(kernel.fun)
    threads = min(d, config.threads)
    blocks = cld(d, threads)

    for _ in 1:d
        kernel(dists, p, colptr, rowval, val, changed; threads=threads, blocks=blocks)
        any(changed) || break
        changed .= false
    end
    return dists, p

end

# iterate over all rows of a target column and find the shortest distance
function bf_cuda_sparse(dists, parent, colptr, rowval, val, changed)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col > length(dists) && return
    d = dists[col]
    p = parent[col]
    for i in colptr[col]:colptr[col+1]-1
        row = rowval[i]
        n = dists[row] + val[i]
        if n < d
            changed[1] = true
            d = n
            p = row
        end
    end
    parent[col] = p
    dists[col] = d
    return
end

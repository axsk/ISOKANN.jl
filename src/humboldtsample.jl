"
humboldtsample(xs, ocp, n, branch)

given a list of points `xs`, propagate each into `branch` trajectories according to the dynamics in `ocp`
and subsamble `n` of the resulting points uniformly over their chi value.
Returns a list of approximately equi-chi-distant start points"
function humboldtsample(xs, ocp, n, branch)
    # this is only used in old isokann.jl
    ocp = deepcopy(ocp)
    ocp.forcing = 0.0
    nxs = copy(xs)
    for x in xs
        for i in 1:branch
            s = msolve(ocp, x)[end][1:end-1]
            push!(nxs, s)
        end
    end

    ys = map(x -> ocp.chi(x)[1], nxs)
    is = subsample_uniformgrid(vec(ys), n)

    return nxs[is]
end

function humboldtsample(xs, ys::AbstractVector, n; keepedges=true)
    # this is used from isonew.jl
    i = subsample_uniformgrid(ys, n; keepedges)
    return xs[:, i]
end

" subsbample_uniformgrid(ys, n) -> is

given a list of values `ys`, return `n`` indices `is` such that `ys[is]` are approximately uniform by
picking the closest points to a randomly perturbed grid in [0,1]."
function subsample_uniformgrid(ys, n; keepedges=true)
    if n <= 2
        keepedges = false  # TODO: maybe we should warn here?
    end
    keepedges && (n = n - 2)
    needles = (rand(n) .+ (0:n-1)) ./ n
    keepedges && (needles = [0; needles; 1])
    pickclosest(ys, needles)::Vector{Int}
end

pickclosest(hs::AbstractVector, ns::AbstractVector) = pickclosestloop(hs, ns)

" pickclosest(haystack, needles)

Return the indices into haystack which lie closest to `needles` without duplicates
by removing haystack candidates after a match.
Note that this is not invariant under pertubations of needles

scales with n log(n) m where n=length(haystack), m=length(needles) "
function pickclosest_sort(haystack::AbstractVector, needles::AbstractVector)
    picks = Int[]
    for needle in needles
        inds = sortperm(norm.(haystack .- needle))
        for i in inds
            if i in picks
                continue
            else
                push!(picks, i)
                break
            end
        end
    end
    return picks
end

function pickclosestloop(hs::AbstractVector, ns::AbstractVector)
    ih = sortperm(hs)  # this is where all the time goes
    hs = hs[ih]
    ns = sort(ns)
    rs = _pickclosestloop(hs, ns)
    return ih[rs]
end


""" scales with n=length(hs) """
function _pickclosestloop(hs::AbstractVector, ns::AbstractVector)
    @assert issorted(hs) && issorted(ns)
    avl = fill(true, length(hs)) # available indices
    rs = Int[]  # picked indices
    i = 1
    for n in ns
        di = abs(hs[i] - n)
        while true
            j = findnext(avl, i + 1)
            if !isnothing(j) && ((dj = abs(hs[j] - n)) <= di)
                di = dj
                i = j
            else
                push!(rs, i)
                avl[i] = false
                i = findprev(avl, i)
                break
            end
        end
        if isnothing(i)
            i = findfirst(avl)
            isnothing(i) && break
        end
    end
    return rs
end

function pickclosest_test(hs, ns)
    hs = sort(hs)
    ns = sort(ns)
    i1 = pickclosest(hs, ns)
    i2 = pickclosestloop(hs, ns)
    @assert i1 == i2
    i1
end

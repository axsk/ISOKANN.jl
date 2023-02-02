using LinearAlgebra: norm
"
humboldtsample(xs, ocp, n, branch)

given a list of points `xs`, propagate each into `branch` trajectories according to the dynamics in `ocp`
and subsamble `n` of the resulting points uniformly over their chi value.
Returns a list of approximately equi-chi-distant start points"
function humboldtsample(xs, ocp, n, branch)
     # this is only used in old isokann.jl
    ocp = deepcopy(ocp)
    ocp.forcing = 0.
    nxs = copy(xs)
    for x in xs
        for i in 1:branch
            s = msolve(ocp, x)[end][1:end-1]
            push!(nxs, s)
        end
    end

    ys = map(x->ocp.chi(x)[1], nxs)
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
    keepedges && (n = n - 2)
    needles = (rand(n)  .+ (0:n-1)) ./ n
    keepedges && (needles = [[0,1]; needles])
    pickclosest(ys, needles)
end


" pickclosest(haystack, needles)

Return the indices into haystack which lie closest to `needles` without duplicates
by removing haystack candidates after a match.
Note that this is not invariant under pertubations of needles"
function pickclosest(haystack::AbstractVector, needles::AbstractVector)
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

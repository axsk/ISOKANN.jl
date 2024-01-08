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
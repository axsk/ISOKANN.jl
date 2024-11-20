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
pickclosest(haystack::CuArray, needles::AbstractVector) = pickclosest(collect(haystack), needles)

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

### Resampling according to the KDE of the data

"""
    resample_kde(xs, ys, n; kwargs...)

Return `n` indices of `ys` such that the corresponding points "fill the gaps" in the KDE of `xs`.
For possible `kwargs` see `kde_needles`.
"""
function resample_kde(xs, ys, n; kwargs...)
    needles = kde_needles(xs, n; kwargs...)
    iy = pickclosest(ys, needles)
    return iy
end

to_pdf(f::Function) = f
to_pdf(d::Distributions.Distribution) = x -> Distributions.pdf(d, x)

import AverageShiftedHistograms

function kde_needles(xs, n=10; bandwidth, target=Distributions.Uniform())
    xs = copy(xs)
    needles = similar(xs, 0)
    target = to_pdf(target)
    for i in 1:n
        k = KernelDensity.kde(xs; bandwidth)
        delta = @. k.density - target(k.x)
        #plot(k.x, [k.density delta]) |> display
        c = k.x[argmin(delta)]
        push!(needles, c)
        push!(xs, c)
    end
    return needles
end

function resample_kde_ash(xs, ys, n=10; m=50, target=Distributions.Uniform())
    iys = zeros(Int, n)
    rng = 0:0.001:1
    kde = AverageShiftedHistograms.ash(xs; rng, m)
    #display(kde)
    target = to_pdf(target)(rng)
    for i in 1:n
        @show chi = rng[argmax(target - kde.density)] # position of maximal difference to target pdf
        min = Inf
        local iy
        for j in 1:length(ys)
            if abs(ys[j] - chi) < min && !(j in iys)
                min = abs(ys[j] - chi)
                iy = j
            end
        end
        AverageShiftedHistograms.ash!(kde, ys[iy])
        iys[i] = iy
    end
    return iys
end

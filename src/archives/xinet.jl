# mock up implementation of the xinet loss

# goal: approximate
# 1/|Z| ∫Z ∫ΣZ ∫ΣZ ||p(x1,∘)-p(x2,∘)||_L1 dμz(x1) dμz(x2) dz
# ^ ignored
#           ^ replaced by integrals over z and weighting with N(xi(x1)-xi(x2))
#                   ^ approximated via kde

# note about the data:
# xs is a vector(sample) of vectors(coordinates), storing the samples in X space.
# ys is a vector(sample) of vectors(batch) of vectors(coordinates)
# that contains the batch of propagated xs, i.e. ys[i][j] ∼ Φ(xs[i]).

gaussian(x, sigma) = 1/(sigma * sqrt(2*pi)) * exp(-1/2 * (x/sigma)^2)
dists(a, b) = [norm(a, b) for a in a, b in b]

"""
l1 distance estimation
by using kernel density estimation (kde) around the given
point clouds ys1, ys2 and integrating their density difference
by monte carlo integration along points zs:
||p1 - p2||_L1μ where ys1∼p1, ys2∼p2, zs∼μ
"""
function l1dist(
        ys1::Vector{Vector},
        ys2::Vector{Vector},
        zs::Vector{Vector}, sigma_x::Real)

    d1 = dists(zs, ys1)
    d2 = dists(zs, ys2)
    sum(abs.(gaussian.(d1, sigma_x) - gaussian.(d2, sigma_x)))
end

"""
lumpability loss approximation,
using the "soft foliation" approach in Z direction and KDE for the L1 distance
"""
function loss(
        xs::Vector{Vector},
        ys::Vector{Vector{Vector}},
        xi::Function, # the candidate reaction coordinate
        sigma_z, # "soft threshold" for the foliation
        sigma_x  # kernel for the kde for the l1 distance estimation
        )

    l = 0.
    for i in 1:length(xs)     # here we probably don't want to go over all points
        for j in 1:length(ys) # but rather do monte carlo sampling
            x1 = xs[i]
            x2 = xs[j]
            # we replace the delta peaks representing the level sets in Z direction
            # by gaussians (still in Z direction)
            w = gaussian(xi(x1)-xi(x2), sigma_z)
            if w > 0.1 # some threshold
                # assuming xs are L1μ distributed
                zs = xs
                l += w * l1dist(ys[i], ys[j], zs, sigma_x)
            end
        end
    end
    return l
end

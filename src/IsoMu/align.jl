""" alignment of pointclouds / trajectories using procrustes alignment """


function aligntrajectory(traj::AbstractVector)
    aligned = [centermean(traj[1])]
    for x in traj[2:end]
        push!(aligned, align(centermean(x), aligned[end]))
    end
    return aligned
end
aligntrajectory(traj::Matrix) = reduce(hcat, aligntrajectory(eachcol(traj)))

centermean(x::AbstractMatrix) = x .- mean(x, dims=2)
centermean(x::AbstractVector) = as3dmatrix(centermean, x)

function align(x::AbstractMatrix, target::AbstractMatrix)
    r = kabsch(x, target)
    y = r * x
    return y
end
align(x::T, target::T) where {T<:AbstractVector} = as3dmatrix(align, x, target)


" compute R such that R*p is closest to q"
function kabsch(p::AbstractMatrix, q::AbstractMatrix)
    h = p * q'
    s = svd(h)
    R = s.V * s.U'
    return R
end


function rmsd(p::AbstractMatrix, q::AbstractMatrix)
    r = kabsch(p, q)
    norm(r * p .- q) / sqrt(size(p, 2))
end

# switch between flattened an blown up representation of 3d vectors
function as3dmatrix(f, x...)
    merge_first_dimensions(f(split_first_dimension.(x, 3)...))
end

function merge_first_dimensions(A)
    new_shape = (prod(size(A)[1:2]), size(A)[3:end]...)
    return reshape(A, new_shape)
end

function split_first_dimension(A, d)
    new_shape = (d, div(size(A, 1), d), size(A)[2:end]...)
    reshape(A, new_shape)
end

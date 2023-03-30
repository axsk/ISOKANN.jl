using LinearAlgebra
using StatsBase: mean

"""
center any given states by shifting their individual 3d mean to the origin
"""
function center(xs)
    mapslices(xs, dims=1) do x
        coords = reshape(x, 3, :)
        coords .-= mean(coords, dims=2)
        vec(coords)
    end
end

# https://naturegeorge.github.io/blog/2022/07/dihedral/
function dihedral(coord0, coord1, coord2, coord3)
    b = coord2 - coord1
    u = cross(b, coord1 - coord0)
    w = cross(b, coord2 - coord3)
    return atan(cross(u, w)'*b, u'*w * norm(b))
end

dihedral(x::AbstractMatrix) = @views dihedral(x[:,1], x[:,2], x[:,3], x[:,4])

function psi(x::AbstractVector)  # dihedral of the oxygens
    x = reshape(x, 3, :)
    @views dihedral(x[:, [7,9,15,17]])
end

function phi(x::AbstractVector)
    x = reshape(x, 3, :)
    @views dihedral(x[:, [5,7,9,15]])
end

phi(x::AbstractMatrix) = mapslices(phi, x, dims=1) |> vec
psi(x::AbstractMatrix) = mapslices(psi, x, dims=1) |> vec

function rotationmatrix(e1, e2)
    e1 ./= norm(e1)
    e2 .-= dot(e1, e2) * e1
    e2 ./= norm(e2)
    e3 = cross(e1, e2)
    A = hcat(e1, e2, e3)
    R = A / I  #  A * R = I
end

function rotatevec(vec)
    x = reshape(vec, 3, :)
    e1 = x[:, 19] .- x[:, 2]
    e2 = x[:, 11] .- x[:, 2]
    R = rotationmatrix(e1, e2)
    return R' * x
end

standardform(x::AbstractArray) = mapslices(x, dims=1) do col
    x = rotatevec(col)
    x .-= mean(x, dims=2)
    vec(x)
end

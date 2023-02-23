using LinearAlgebra: cross

# https://naturegeorge.github.io/blog/2022/07/dihedral/
function dihedral(coord0, coord1, coord2, coord3)
    b = coord2 - coord1
    u = cross(b, coord1 - coord0)
    w = cross(b, coord2 - coord3)
    return atan(cross(u, w)'*b, u'*w * norm(b))
end

dihedral(x::AbstractMatrix) = @views dihedral(x[:,1], x[:,2], x[:,3], x[:,4])

function scatter_ramachandran(x::Matrix, model=nothing)
    z = nothing
    !isnothing(model) && (z = model(x) |> vec)
    ph = phi(x)
    ps = psi(x)
    scatter(ph, ps, marker_z=z, xlims = [-pi, pi], ylims=[-pi, pi],
        markersize=3, markerstrokewidth=0, markeralpha=1, markercolor=:hawaii,
        xlabel="\\phi", ylabel="\\psi", title="Ramachandran",
    )
end

function psi(x::AbstractVector)  # dihedral of the oxygens
    x = reshape(x, 3, :)
    @views ISOKANN.dihedral(x[:, [7,9,15,17]])
end

function phi(x::AbstractVector)
    x = reshape(x, 3, :)
    @views ISOKANN.dihedral(x[:, [5,7,9,15]])
end

phi(x::Matrix) = mapslices(phi, x, dims=1) |> vec
psi(x::Matrix) = mapslices(psi, x, dims=1) |> vec


using LinearAlgebra
using StatsBase: mean

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






# TODO: unify or remove this
""" save data into a pdb file sorted by model evaluation """
function extractdata(data::AbstractArray, model, sys, path="out/data.pdb")
    dd = data
    dd = reshape(dd, size(dd, 1), :)
    ks = model(dd)
    i = sortperm(vec(ks))
    dd = dd[:, i]
    i = uniqueidx(dd[1,:] |> vec)
    dd = dd[:, i]
    dd = standardform(dd)
    ISOKANN.exportdata(sys, path, dd)
    dd
end



uniqueidx(v) = unique(i -> v[i], eachindex(v))

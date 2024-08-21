### Utilities to work with molecules.
### Dihedral angles, standardform, writing/reading trajectories and alignment

"""
centercoords any given states by shifting their individual 3d mean to the origin
"""
function centercoords(xs)
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
    return atan(cross(u, w)' * b, u' * w * norm(b))
end

dihedral(x::AbstractMatrix) = @views dihedral(x[:, 1], x[:, 2], x[:, 3], x[:, 4])

function psi(x::AbstractVector, inds=[7, 9, 15, 17])  # dihedral of the oxygens
    x = reshape(x, 3, :)
    @views dihedral(x[:, inds])
end

function phi(x::AbstractVector, inds=[5, 7, 9, 15])
    x = reshape(x, 3, :)
    @views dihedral(x[:, inds])
end

phi(x::AbstractMatrix) = mapslices(phi, x, dims=1) |> vec
psi(x::AbstractMatrix) = mapslices(psi, x, dims=1) |> vec

# compute the rotationmatrix rotating e1 and e2 onto the cartesian axes
function rotationmatrix(e1, e2)
    e1 ./= norm(e1)
    e2 .-= dot(e1, e2) * e1
    e2 ./= norm(e2)
    e3 = cross(e1, e2)
    A = hcat(e1, e2, e3)
    R = A / I  #  A * R = I
end

# rotates a 3xn vector into standardform taking the i0, i1, and i2 atoms as handles
function rotatevec(vec, rotationhandles)
    i0, i1, i2 = rotationhandles
    x = reshape(vec, 3, :)
    e1 = x[:, i1] .- x[:, i0]
    e2 = x[:, i2] .- x[:, i0]
    R = rotationmatrix(e1, e2)
    return R' * x
end

# compute the standardform by applying mean shift and rotatevec
standardform(x::AbstractArray, rotationhandles=(2, 11, 19)) =
    mapslices(x, dims=1) do col
        x = rotatevec(col, rotationhandles)
        x .-= mean(x, dims=2)
        vec(x)
    end

standardform(x::AbstractArray, sim::IsoSimulation) = standardform(x, rotationhandles(sim))

### alignment of pointclouds / trajectories using procrustes alignment
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

function kabsch_rmsd(p::AbstractMatrix, q::AbstractMatrix)
    r = kabsch(p, q)
    norm(r * p .- q) / sqrt(size(p, 2))
end

### switch between flattened an blown up representation of 3d vectors
function as3dmatrix(f, x...)
    flattenfirst(f(split_first_dimension.(x, 3)...))
end

function split_first_dimension(A, d)
    s1, s2... = size(A)
    reshape(A, (d, div(s1, d), s2...))
end

"""
    load_trajectory(filename; top=nothing, kwargs...)

wrapper around Python's `mdtraj.load()`.
Returns a (3 * natom, nframes) shaped array.
"""
function load_trajectory(filename; top::Union{Nothing,String}=nothing, stride=nothing, atom_indices=nothing)
    mdtraj = pyimport_conda("mdtraj", "mdtraj", "conda-forge")

    if filename[end-2:end] != "pdb" && isnothing(top)
        error("must supply topology file (.pdb) to the top argument")
    end

    traj = mdtraj.load(filename; top, stride, atom_indices)
    xs = permutedims(PyArray(py"$traj.xyz"o), (3, 2, 1))
    xs = reshape(xs, :, size(xs, 3))::Matrix{Float32}
end

"""
    save_trajectory(filename, coords::AbstractMatrix; top::String)

save the trajectory given in `coords` to `filename` with the topology provided by the file `top`
"""
function save_trajectory(filename, coords::AbstractMatrix; top::String)
    mdtraj = pyimport_conda("mdtraj", "mdtraj", "conda-forge")
    traj = mdtraj.load(filename, stride=-1)
    xyz = reshape(coords, 3, :, size(coords, 2))
    traj.xyz = PyReverseDims(xyz)
    traj.save(filename)
end
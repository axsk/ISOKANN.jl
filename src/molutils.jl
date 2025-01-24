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
        push!(aligned, align(x, aligned[end]))
    end
    return aligned
end
aligntrajectory(traj::AbstractMatrix) = reduce(hcat, aligntrajectory(eachcol(traj)))

centermean(x::AbstractMatrix) = x .- mean(x, dims=2)
centermean(x::AbstractVector) = as3dmatrix(centermean, x)

function align(x::AbstractMatrix, target::AbstractMatrix)
    m = mean(target, dims=2)
    x = centermean(x)
    r = kabschrotation(x, target .- m)
    y = r * x .+ m
    return y
end
align(x::S, target::T) where {S<:AbstractVector, T<:AbstractVector} = as3dmatrix(align, x, target)


" compute R such that R*p is closest to q"
function kabschrotation(p::AbstractMatrix, q::AbstractMatrix)
    h = p * q'
    s = svd(h)
    R = s.V * s.U'
    return R
end

function aligned_rmsd(p::AbstractMatrix, q::AbstractMatrix)
    p = align(p, q)
    n = size(p, 2)
    norm(p - q) / sqrt(n)
end

using NNlib: batched_mul, batched_transpose

function pairwise_aligned_rmsd(xs)
    n = size(xs, 2)
    xs = reshape(xs, 3, :, n)
    xs = xs .- mean(xs, dims=2)
    dists = similar(xs, n, n)
    for i in 1:n
        dists[:,i] = batched_kabsch_rmsd(xs[:,:,i], xs)
    end
    return dists
end


function batched_kabsch_rmsd(x::AbstractMatrix, xs::AbstractArray{<:Any, 3})
    h = batched_mul(x, batched_transpose(xs))
    s = batched_svd(h)
    r = batched_mul(s.V, batched_transpose(s.U))
    d = sqrt.(sum(abs2, batched_mul(r, x) .- xs, dims=(1, 2)) ./ size(x, 2))
    return vec(d)
end


batched_kabsch_rmsd(x::AbstractVector, xs::AbstractMatrix) = batched_kabsch_rmsd(reshape(x, 3, :), reshape(xs, 3, :, size(xs, 2)))


batched_svd(x::CuArray) = svd(x)

function batched_svd(x)
    u = similar(x)
    v = similar(x)
    s = similar(x, size(x,2), size(x,3))
    for i in 1:size(x, 3)
        u[:, :, i], s[:, i], v[:, :, i] = svd(x[:, :, i])
    end
    return (; U=u, S=s, V=v)
end

function _pairwise_aligned_rmsd(xs)
    d3, n = size(xs)
    p = div(d3, 3)
    xs = reshape(xs, 3, p, n)
    xs = xs .- mean(xs, dims=2)
    dists = similar(xs, n, n)
    for i in 1:n
        for j in 1:n
            x = @view xs[:, :, i]
            y = @view xs[:, :, j]
            s = svd(x * y')
            r = s.V * s.U'
            dists[i, j] = dists[j, i] = sqrt(sum(abs2, r * x - y) / p)
        end
    end
    return dists
end

#=
function batched_svd(x::Array)
    u = similar(x)
    v = similar(x)
    s = similar(x, size(x)[2:3])
    for i in 1:size(x,3)
        u[:,:,i], s[:,i], v[:,:,i] = svd(x[:,:,i])
    end
    return u,s,v
end

batched_svd(x::CuArray) = svd(x)
=#

### switch between flattened an blown up representation of 3d vectors
function as3dmatrix(f, x...)
    flattenfirst(f(split_first_dimension.(x, 3)...))
end

as3dmatrix(x) = split_first_dimension(x, 3)

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

    if isnothing(top)
        if filename[end-2:end] == "pdb"
            top = filename
        else
            error("must supply topology file (.pdb) to the top argument")
        end
    end

    if !isnothing(atom_indices)
        atom_indices = atom_indices .- 1
    end

    traj = mdtraj.load(filename; top, stride, atom_indices)
    xs = permutedims(PyArray(py"$traj.xyz"o), (3, 2, 1))
    xs = reshape(xs, :, size(xs, 3))
    return xs::Matrix{Float32}
end

"""
    save_trajectory(filename, coords::AbstractMatrix; top::String)

save the trajectory given in `coords` to `filename` with the topology provided by the file `top` using mdtraj.
"""
function save_trajectory(filename, coords::AbstractMatrix; top::String)
    mdtraj = pyimport_conda("mdtraj", "mdtraj", "conda-forge")
    traj = mdtraj.load(top, stride=-1)
    xyz = reshape(coords, 3, :, size(coords, 2))
    traj = mdtraj.Trajectory(PyReverseDims(xyz), traj.topology)
    traj.save(filename)
end

function atom_indices(filename::String, selector::String)
    mdtraj = pyimport_conda("mdtraj", "mdtraj", "conda-forge")
    traj = mdtraj.load(filename, stride=-1)
    inds = traj.top.select(selector) .+ 1
    return inds::Vector{Int}
end

import Chemfiles

function readchemfile(source::String, steps=:)
    traj = Chemfiles.Trajectory(source, 'r')
    try
        readchemfile(traj, steps)
    finally
        close(traj)
    end
end

function readchemfile(traj::Chemfiles.Trajectory, frames)
    frame = Chemfiles.read_step(traj, 0)
    xs = Array{Float32}(undef, length(Chemfiles.positions(frame)), length(frames))
    for (i, s) in enumerate(frames)
        Chemfiles.read_step!(traj, s - 1, frame)
        try
            xs[:, i] .= Chemfiles.positions(frame).data |> vec
        catch e
            i == size(xs, 2) || rethrow(e) ## hadle connect record which is not read properly by chemfiles
            xs = xs[:, 1:end-1]
        end
    end
    xs ./= 10 # convert from Angstrom to nm
    return xs
end

readchemfile(traj::Chemfiles.Trajectory, frames::Colon=:) =
    readchemfile(traj::Chemfiles.Trajectory, Base.OneTo(length(traj)))

readchemfile(traj::Chemfiles.Trajectory, frame::Int) =
    readchemfile(traj, frame:frame) |> vec

"""
    writechemfile(filename, data::Array{<:Any,2}; source)

Save the coordinates in `data` to `filename` with `source` as template using the Chemfiles library"""
function writechemfile(filename, data::Array{<:Any,2}; source)
    trajectory = Chemfiles.Trajectory(source, 'r')
    try
        frame = Chemfiles.read(trajectory)
        trajectory = Chemfiles.Trajectory(filename, 'w')#, uppercase(split(filename, ".")[end]))
        for i in 1:size(data, 2)
            Chemfiles.positions(frame) .= reshape(data[:, i], 3, :) .* 10 # convert from nm to Angstrom
            write(trajectory, frame)
        end
    finally
        close(trajectory)
    end
end

mutable struct LazyTrajectory <: AbstractMatrix{Float32}
    path::String
    traj::Chemfiles.Trajectory
    size::Tuple{Int,Int}
end

function LazyTrajectory(path::String)
    traj = Chemfiles.Trajectory(path, 'r')
    frame = read(traj)
    s = (length(Chemfiles.positions(frame)), Int(length(traj)))
    return LazyTrajectory(path, traj, s)
end

Base.size(ltl::LazyTrajectory) = ltl.size
Base.getindex(ltl::LazyTrajectory, i1, i2) = ISOKANN.readchemfile(ltl.traj, i2)[i1, :]
Base.getindex(ltl::LazyTrajectory, i1::Union{Colon,Base.OneTo}, i2) = ISOKANN.readchemfile(ltl.traj, i2)
Base.getindex(ltl::LazyTrajectory, i1::Union{Colon,Base.OneTo}, i2::Int) = ISOKANN.readchemfile(ltl.traj, i2)
Base.getindex(ltl::LazyTrajectory, i1, i2::Int) = vec(ISOKANN.readchemfile(ltl.traj, i2))[i1]

struct LazyMultiTrajectory <: AbstractMatrix{Float32}
    lts::Vector{LazyTrajectory}
end

Base.collect(l::LazyMultiTrajectory) = l[:, :]
Base.size(l::LazyMultiTrajectory) = (size(l.lts[1])[1], sum(size(l)[2] for l in l.lts))

function Base.getindex(l::LazyMultiTrajectory, I, j::Int)
    for t in l.lts
        len = size(t, 2)
        j <= len && return t[I, j]
        j = j - len
    end
end

function Base.getindex(l::LazyMultiTrajectory, V::Vararg)
    I, J = to_indices(l, V)
    res = Array{eltype(l)}(undef, length(I), length(J))
    for t in l.lts
        len = size(t, 2)
        c = findall(x -> 0 < x <= len, J)
        res[:, c] = t[I, J[c]]
        J = J .- len
    end
    return res
end
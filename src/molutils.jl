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


## Chemfiles to read/write trajectory data

function readchemfile(source::String, steps=nothing)
    traj = Chemfiles.Trajectory(source, 'r')
    readchemfile(traj, steps)
end

function readchemfile(traj::Chemfiles.Trajectory, steps=nothing)
    isnothing(steps) && (steps = 1:length(traj))
    frame = Chemfiles.read_step(traj, 0)
    xs = Array{Float32}(undef, size(Chemfiles.positions(frame))..., length(steps))
    for (i, s) in enumerate(steps)
        Chemfiles.read_step!(traj, s - 1, frame)
        xs[:, :, i] .= Chemfiles.positions(frame)
    end
    return xs
end

function writechemfile(filename, data::Array{<:Any,3}; source)
    trajectory = Chemfiles.Trajectory(source, 'r')
    frame = Chemfiles.read(trajectory)
    trajectory = Chemfiles.Trajectory(filename, 'w', uppercase(split(filename, ".")[end]))
    for i in 1:size(data, 3)
        Chemfiles.positions(frame) .= data[:, :, i]
        write(trajectory, frame)
    end
    close(trajectory)
end

function writechemfile(filename, data::Array{<:Any,2}; source)
    n = size(data, 2)
    data = reshape(data, 3, :, n)
    writechemfile(filename, data; source)
end


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
    merge_first_dimensions(f(split_first_dimension.(x, 3)...))
end

function merge_first_dimensions(A)
    new_shape = (prod(size(A)[1:2]), size(A)[3:end]...)
    return reshape(A, new_shape)
end

function split_first_dimension(A, d)
    s1, s2... = size(A)
    reshape(A, (d, div(s1, d), s2...))
end

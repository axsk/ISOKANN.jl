
"""
    save_iso(path::String, iso::Iso)

Save the complete Iso object to a JLD2 file """
save_iso(path::String, iso::Iso) = JLD2.save(path, "iso", cpu(iso))

"""
    load(path::String, iso::Iso)

Load the Iso object from a JLD2 file
Note that it will be loaded to the CPU, even if it was saved on the GPU.
"""
function load_iso(path::String) 
    iso = JLD2.load(path, "iso")
    if CUDA.has_cuda()
        return gpu(iso)
    else
        return iso
    end
end

load_coords(filename, steps) = load_coords_chemfiles(filename, steps)
save_coords(filename, coords, topology) = save_coords_chemfiles(filename, coords, topology=topology)

@deprecate readchemfile load_coords_chemfiles
@deprecate writechemfile(filename, data::Array{<:Any,2}; source)  save_coords_chemfiles(filename, coords, topology=source)
@deprecate load_trajectory(filename; top, stride, atom_indices) load_coords_mdcoords(filename, topology=top; stride, atom_indices)

### Chemfiles

import Chemfiles

function load_coords_chemfiles(topology::String, frames=:)
    traj = Chemfiles.Trajectory(topology, 'r')
    try
        load_coords_chemfiles(traj, frames)
    finally
        close(traj)
    end
end

function load_coords_chemfile(traj::Chemfiles.Trajectory, frames)
    frame = Chemfiles.read_step(traj, 0)
    xs = Array{Float32}(undef, length(Chemfiles.positions(frame)), length(frames))
    read = fill(length(frames), false)
    for i in frames
        Chemfiles.read_step!(traj, i - 1, frame)
        try
            xs[:, i] .= Chemfiles.positions(frame).coords |> vec
            read[i] = true
        catch
        end
        
    end
    xs = xs[:, read]
    xs ./= 10 # convert from Angstrom to nm
    return xs
end

load_coords_chemfiles(traj::Chemfiles.Trajectory, frames::Colon=:) =
    load_coords_chemfiles(traj::Chemfiles.Trajectory, Base.OneTo(length(traj)))

load_coords_chemfiles(traj::Chemfiles.Trajectory, frame::Int) =
    load_coords_chemfiles(traj, frame:frame) |> vec

function save_coords_chemfiles(filename, coords::Array{<:Any,2}; topology)
    coords = cpu(coords)
    trajectory = Chemfiles.Trajectory(topology, 'r')
    try
        frame = Chemfiles.read(trajectory)
        trajectory = Chemfiles.Trajectory(filename, 'w', uppercase(split(filename, ".")[end]))
        for i in 1:size(coords, 2)
            Chemfiles.positions(frame) .= reshape(coords[:, i], 3, :) .* 10 # convert from nm to Angstrom
            write(trajectory, frame)
        end
    finally
        close(trajectory)
    end
end

## MDTraj

"""
    load_coords_mdcoords(filename; topology=nothing, kwargs...)

wrapper around Python's `mdtraj.load()`.
Returns a (3 * natom, nframes) shaped array.
"""
function load_coords_mdcoords(filename; topology::Union{Nothing,String}=nothing, stride=nothing, atom_indices=nothing)
    mdtraj = pyimport_conda("mdtraj", "mdtraj", "conda-forge")

    if isnothing(topology)
        if filename[end-2:end] == "pdb"
            topology = filename
        else
            error("must supply topology file (.pdb) to the topology argument")
        end
    end

    if !isnothing(atom_indices)
        atom_indices = atom_indices .- 1
    end

    traj = mdtraj.load(filename; topology, stride, atom_indices)
    xs = permutedims(PyArray(py"$traj.xyz"o), (3, 2, 1))
    xs = reshape(xs, :, size(xs, 3))
    return xs::Matrix{Float32}
end

"""
    save_coords_mdcoords(filename, coords::AbstractMatrix; topology::String)

save the trajectory given in `coords` to `filename` with the topology provided by the file `topology`
"""
function save_coords_mdcoords(filename, coords::AbstractMatrix, topology::String)
    coords = cpu(coords)
    mdtraj = pyimport_conda("mdtraj", "mdtraj", "conda-forge")
    traj = mdtraj.load(topology, stride=-1)
    xyz = reshape(coords, 3, :, size(coords, 2))
    traj = mdtraj.Trajectory(PyReverseDims(xyz), traj.topology)
    traj.save(filename)
end

function atom_indices(filename::String, selector::String)
    mdtraj = pyimport_conda("mdtraj", "mdtraj", "conda-forge")
    traj = mdtraj.load(filename, stride=-1)
    inds = traj.topology.select(selector) .+ 1
    return inds::Vector{Int}
end

## Read/write trajectory data

# read in trajectory data from any Chemfiles supported file format
function readchemfile(source::String, steps=nothing)
    traj = Chemfiles.Trajectory(source, 'r')
    readchemfile(traj, steps)
end

function readchemfile(traj, steps=nothing)
    isnothing(steps) && (steps = 1:length(traj))
    frame = read_step(traj, 0)
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


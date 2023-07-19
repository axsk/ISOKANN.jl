# this allows to use the current isokann implementation with a DataLoader
# where (according to MLUtils practice) the ys have shape (dim x nkoop x npoints)
# we therefore permute the last dims to adhere to the ISOKANN.jl standard

function datasubsample(model, data::DataLoader, nx)
    x, y = first(data)
    y = permutedims(y, (1, 3, 2))
    return (x, y)
end
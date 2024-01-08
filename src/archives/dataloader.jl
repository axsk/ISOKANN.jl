# this allows to use the current isokann implementation with a DataLoader
# where (according to MLUtils practice) the ys have shape (dim x nkoop x npoints)
# we therefore permute the last dims to adhere to the ISOKANN.jl standard

function datasubsample(model, data::DataLoader, nx)
    x, y = first(data)
    y = permutedims(y, (1, 3, 2))
    return (x, y)
end

### Attempt to use Flux.jl data interface for data loading
struct IsoData{T}
    xs::Array{T,2}
    ys::Array{T,3}
end

numobs(d::IsoData) = size(d.xs, 2)
getobs(d::IsoData, idx) = (d.xs[:, idx], d.ys[:, idx, :])
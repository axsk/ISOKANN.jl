using StatsBase: sample

# in this file we aim at comparing different data collection techniques
# a) stratified chi sampling
# b) using a trajectory
# c) using stationary sampled koopman trajectories

# let us therefore collect all three datasets first


DataTuple = Tuple{Matrix{T},Array{T,3}} where {T<:Number}

using JLD2
function load_refiso()
    load("isoreference-6440710-0.jld2", "iso")
end

""" compute initial data by propagating the molecules initial state
to obtain the xs and propagating them further for the ys """
function bootstrap(sim::IsoSimulation, nx, ny)
    x0 = reshape(getcoords(sim), :, 1)
    xs = reshape(propagate(sim, x0, nx), :, nx)
    ys = propagate(sim, xs, ny)
    centercoords(xs), centercoords(ys)
end

function datasubsample(model, data, nx)
    # chi stratified subsampling
    xs, ys = data
    if size(xs, 2) <= nx || nx == 0
        return data
    end
    cs = model(xs) |> vec
    ks = shiftscale(cs)
    ix = ISOKANN.subsample_uniformgrid(ks, nx)
    xs = xs[:, ix]
    ys = ys[:, ix, :]

    return xs, ys
end

# this allows to use the current isokann implementation with a DataLoader
# where (according to MLUtils practice) the ys have shape (dim x nkoop x npoints)
# we therefore permute the last dims to adhere to the ISOKANN.jl standard
using MLUtils
function datasubsample(model, data::DataLoader, nx)
    x, y = first(data)
    y = permutedims(y, (1, 3, 2))
    return (x, y)
end


function subsample_inds(model, xs, n)
    ks = shiftscale(model(xs) |> vec)
    i = subsample_uniformgrid(ks, n)
end

""" subsample n points of data uniformly according to the provided model

Works for points provided either as plain Array or as (xs, ys) data tuple

subsample(model, data::Array, n) :: Matrix
subsample(model, data::Tuple, n) :: Tuple
"""
function subsample(model, xs::AbstractArray{<:Any,2}, n)
    xs[:, subsample_inds(model, xs, n)]
end

function subsample(model, ys::AbstractArray{<:Any,3}, n)
    xs = reshape(ys, size(ys, 1), :)
    subsample(model, xs, n)
end

function subsample(model, data::Tuple, n)
    xs, ys = data
    @show ix = subsample_inds(model, xs, n)
    return (xs[:, ix], ys[:, ix, :])
end



function adddata(data, model, sim::IsoSimulation, ny, lastn=1_000_000)
    _, ys = data
    nk = size(ys, 3)
    firstind = max(size(ys, 2) - lastn + 1, 1)
    x0 = stratified_x0(model, ys[:, firstind:end, :], ny)
    ys = propagate(sim, x0, nk)
    ndata = centercoords(x0), centercoords(ys)
    data = hcat.(data, ndata)

    datastats(data)
    return data
end

""" given an array of states, return a chi stratified subsample """
function stratified_x0(model, ys, n)
    ys = reshape(ys, size(ys, 1), :)
    ks = shiftscale(model(ys) |> vec)

    i = subsample_uniformgrid(ks, n)
    xs = ys[:, i]
    return xs
end

function datastats(data)
    xs, ys = data
    ext = extrema(xs[1, :])
    uni = length(unique(xs[1, :]))
    _, n, ks = size(ys)
    #println("\n Dataset has $n entries ($uni unique) with $ks koop's. Extrema: $ext")
end


""" save data into a pdb file sorted by model evaluation """
function extractdata(data::AbstractArray, model, sim, path="out/data.pdb")
    dd = data
    dd = reshape(dd, size(dd, 1), :)
    ks = model(dd)
    i = sortperm(vec(ks))
    dd = dd[:, i]
    i = uniqueidx(dd[1, :] |> vec)
    dd = dd[:, i]
    dd = standardform(dd)
    savecoords(sys, dd, path)
    dd
end

uniqueidx(v) = unique(i -> v[i], eachindex(v))

function testdata(ref=load_refiso())
    tdata = data_sliced(shuffledata(data_sliced(ref.data, 1000:2000)), 1:500)
    return tdata
end

function traindata(ref=load_refiso(); n=100, k=8, offset=500)
    x, y = ref.data
    shuffledata((x[:, offset:offset+n-1], y[:, offset:offset+n-1, 1:k]))
end

# when we already have a good data pool and a model, we can reuse that
function data_stratified(model, data::Tuple, nx, nk)
    xs, ys = data
    ks = shiftscale(model(xs) |> vec)
    i = subsample_uniformgrid(ks, nx)

    return xs[:, i], ys[:, i, 1:nk]
end

data_stratified(iso::IsoRun, nx, nk) = data_stratified(iso.model, iso.data, nx, nk)

function data_from_trajectory(xs::Matrix, nx=size(xs, 2) - 1)
    ys = reshape(xs[:, 2:nx+1], :, nx, 1)
    return xs[:, 1:nx], ys
end

function data_stationary(trajdata, nx, nk)
    xs, ys = trajdata
    i = sample(1:length(traj), nx, replace=false)
    return xs[:, i], ys[:, i, 1:nk]
end

function data_sliced(data::Tuple, slice)
    xs, ys = data
    (xs[:, slice], ys[:, slice, :])
end

function shuffledata(data)
    xs, ys = data
    n = size(xs, 2)
    i = randperm(n)
    return xs[:, i], ys[:, i, :]
end

function trajdata(sim, nx)
    siml = deepcopy(sim)
    logevery = round(Int, sim.T / sim.dt)
    siml.T = sim.T * nx
    xs = solve(siml; logevery=logevery)
    return xs
end

function iso_fixeddata(iso, data=iso.data)
    iso = deepcopy(iso)
    iso.data = data
    iso.minibatch = iso.nx
    iso.nx = size(data[1], 2)
    iso.nd = round(Int, iso.nd / (iso.nx / iso.minibatch))
    iso.nres = 0
    iso
end

#=
function iso_fixeddata(iso, data=iso.data)
    iso = deepcopy(iso)
    iso.data = data

    iso.nx = size(data[1],2)
    iso.nres = 0
    return iso
end
=#



function data_vs_loss(iso, log, data)
    ndata = size(iso.data[2], 2) * size(iso.data[2], 3)
    nd = []
    ls = []
    for i in eachindex(log)
        push!(nd, i / length(log) * ndata)
        push!(ls, loss(log[i], data))
    end
    return nd, ls
end

function data_vs_testloss(iso, log, data)
    ndata = size(iso.data[2], 2) * size(iso.data[2], 3)
    is = []
    ls = []
    for i in eachindex(log)
        push!(is, i / length(log) * ndata)
        push!(ls, loss(log[i], data))
    end
    return is, ls
end

function plot_iter_vs_testloss(iso, log, data)
    n = length(iso.losses)
    tl = map(log) do m
        loss(m, data)
    end
    plot!(iso.losses, label="trainloss", yaxis=:log)
    plot!(range(1, n, length(log)), tl, label="testloss")
end

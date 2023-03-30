using StatsBase: sample

# in this file we aim at comparing different data collection techniques
# a) stratified chi sampling
# b) using a trajectory
# c) using stationary sampled koopman trajectories

# let us therefore collect all three datasets first
using JLD2
function load_refiso()
    load("isoreference-6440710-0.jld2", "iso")
end

DataTuple = Tuple{Matrix{T}, Array{T, 3}} where T<:Number

function testdata(ref=load_refiso())
    tdata = data_sliced(shuffledata(data_sliced(ref.data, 1000:2000)), 1:500)
    return tdata
end

function traindata(ref=load_refiso(); n=100, k=8, offset=500)
    x,y = ref.data
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

function data_from_trajectory(xs::Matrix, nx = size(xs,2)-1)
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
    (xs[:,slice], ys[:,slice,:])
end

function shuffledata(data)
    xs, ys = data
    n = size(xs, 2)
    i = randperm(n)
    return xs[:, i], ys[:, i, :]
end

function trajdata(sim, nx, nk)
    siml = deepcopy(sim)
    logevery = round(Int, sim.T / sim.dt)
    siml.T = sim.T * nx
    xs = solve(siml; logevery=logevery)
    if nk > 0
        ys = propagate(sim, xs, nk)
    else
        ys = nothing
    end
    return xs, ys
end

function iso_fixeddata(iso, data=iso.data)
    iso = deepcopy(iso)
    iso.data = data
    iso.minibatch = iso.nx
    iso.nx = size(data[1],2)
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
    for i in 1:length(log)
        push!(nd, i/length(log) * ndata)
        push!(ls, loss(log[i], data))
    end
    return nd, ls
end

function data_vs_testloss(iso, log, data)
    ndata = size(iso.data[2], 2) * size(iso.data[2], 3)
    is = []
    ls = []
    for i in 1:length(log)
        push!(is, i/length(log) * ndata)
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

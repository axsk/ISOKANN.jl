# TODO: move to archives
function isobenchmarks(;ref, reps=6)
    [
        if 80 <= nx * nk <= 16000
            IsoRun(
                nres=0,
                data=data_stratified(ref.model, ref.data, nx,nk),
                loggers=[TrainlossLogger(data=testdata(ref)), autoplot(5)],
                opt=AdamRegularized(1e-3, 1e-4),
                nd=1000,
                nx=nx);
        else
            nothing
        end

        for nx in [10,20,50,100,200,500,1000,2000], nk in [1,2,4,8,16], i in 1:reps
    ]
end

function isotraj(;traj,ref, reps=6)
    [
        IsoRun(
                nres=0,
            data=data_from_trajectory(traj, nx),
            loggers=[TrainlossLogger(data=testdata(ref)), autoplot(5)],
                opt=AdamRegularized(1e-3, 1e-4),
                nd=1000,
                nx=nx)
        for nx in [10,20,50,100,200,500,1000,2000,5000,9999], nk in [1], i in 1:reps
    ]
end


function plotbenchmark(isos; plotargs...)
    for k in axes(isos, 2)
        @show i = [!isnothing(isos[i,k,1]) for i in axes(isos, 1)]
        col = filter(!isnothing, isos[:,k,1])
        ns = @show map(x->prod(size(x.data[2])[2:3]), col)
        nk = size(col[1].data[2], 3)
        ls = map(x->x.loggers[1].losses[end], isos[i,k,:])
        lss = mean_and_std(ls, 2)
        #lss[1] .= minimum(ls, dims=2)
        #ls = minimum(ls, dims=2)
        #scatter!(ns, ls, label="")
        plot!(ns, lss[1],
            #ribbon = lss[2],
            label="$nk"; plotargs...)
    end
    plot!(xaxis=:log, yaxis=:log)
end


function data_stratified(model, data::Tuple, nx, nk)
    xs, ys = subsample(model, data, nx)
    return xs, ys[:, :, 1:nk]
end

# see also scripts/runs.jl

import JLD2

global _REFISO = nothing
function referenceiso()
    global _REFISO
    if isnothing(_REFISO)
        _REFISO = JLD2.load("isoreference-6440710-0.jld2", "iso")
    end
    return _REFISO
end

function traindata(ref=referenceiso(); n=100, k=8, offset=500)
    x, y = ref.data
    shuffledata((x[:, offset:offset+n-1], y[:, offset:offset+n-1, 1:k]))
end

function testdata(ref=referenceiso())
    tdata = data_sliced(shuffledata(data_sliced(ref.data, 1000:2000)), 1:500)
    return tdata
end
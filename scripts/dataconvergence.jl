"""
In this script we compare the data quality of

1) a posteriori χ-stratified,
2) a posteriori π-stationary,
3) online χ-stratified and
4) offline trajectory

for the ISOKANN learning procedure.
To this end we compute a

- reference solution to obtain test-/verification-data with many koopman samples
- long trajectory to obtain the stationary samples and an offline trajectory

We may also inlcude a test of how many test datapoints (x) are necessary

"""

using ISOKANN
using ISOKANN: IsoSimulation, data_sliced, shuffledata
using SplitApplyCombine, Dictionaries, Plots
using StatsBase: mean
using ProgressMeter

global isos=Dictionary()

function ISOConvergence(;kwargs...)
    iso = ISORun(
            sim=MollyLangevin(sys=PDB_ACEMD(), dt=2e-3, T=1e-1, temp=298., gamma=100.),
            loggers=[];
            kwargs...)
end

function ISOConvergenceRef()
    ISOConvergence(; nd=10000, opt=ISOKANN.AdamRegularized(1e-4,1e-5))
end

function analyse_dataconvergence_high(isos=isos)
    analyse_dataconvergence(
        nxtest = 100,
        nktest=100,
        mindata=100,
        maxdata=2_000,
        i=1:5,
        isos = isos,
        iso = ISORun(
            sim=MollyLangevin(sys=PDB_ACEMD(), dt=2e-3, T=1e-1, temp=298., gamma=10.),
            loggers=[],)
    )
end

function analyse_dataconvergence_med()
    analyse_dataconvergence(nxtest = 200, nktest=50, mindata=100, maxdata=1000, i=1:1,
        nk=[1])
end

function analyse_dataconvergence_test()
    analyse_dataconvergence(nxtest = 10, nktest=4, mindata=1, maxdata=10, i=1:1)
end



function analyse_dataconvergence(;
        nxtest  = 100,    # number of x samples for ref data
        nktest = 10_000, # number of k samples for ref data
        mindata = 100,
        maxdata = 16_000,
        trajlength = maxdata,
        isos = Dictionary(),
        iso = ISORun(sim=MollyLangevin(sys=PDB_ACEMD(), dt=2e-3, T=1e-1, temp=298., gamma=100.)),
        nx = [10,20,50,100,200,500,1000,2000,5000,10000,20000],
        nk = [1,2,4,8,16],
        i = 1:5,
        lossevery = 100,
        data_train = nothing,
        data_test = nothing
    )

    if any(isnothing.([data_train, data_test]))
        nxtrain = min(maximum(nx), maxdata)
        data_train, data_test = generate_data_test(iso; nxtest, nxtrain, nktest)
    end

    # create trajectory data


    iso = deepcopyset(iso, loggers=[ISOKANN.TrainlossLogger(data = data_test, every=lossevery)])

    for i in i
        data_traj = generate_data_traj(iso.sim, trajlength)
        for nx in nx, nk in nk
            if mindata <= nx * nk <= maxdata

                get!(isos, (;type=:chistrat, nx, nk, i)) do
                    deepcopyset(iso, nres=0; nx, nk,
                        data = data_sub_or_augment(shuffledata(data_train), nx, nk, iso.sim))
                end

                get!(isos, (;type=:chiadapt, nx, nk, i)) do
                    deepcopyset(iso; nx, nk, nxmax = nx,
                        # start with only ny samples, setting the number of nk for further sampling
                        data = data_sub_or_augment(shuffledata(data_train), iso.ny, nk, iso.sim),
                        # adjust resample size to resample up to nxmax=nx after half the training
                        ny = max(ceil(Int, nx / (iso.nd/iso.nres*(2/3))),2))
                end

                if nk==1
                    get!(isos, (;type=:traj, nx, nk, i)) do
                        deepcopyset(iso, nres=0; nx, nk,
                            data = data_sliced(data_traj, 1:nx))
                    end

                    get!(isos, (;type=:stat, nx, nk, i)) do
                        deepcopyset(iso, nres=0; nx, nk,
                            data = data_sliced(shuffledata(data_traj), 1:nx)
                        )
                    end
                end
            end
        end
    end

    p = Progress(length(isos))
    for (k,v) in pairs(isos)
        @show k
        length(v.losses) > 0 && continue
        run!(v)
        next!(p)
    end


    return (;isos, data_train, data_test)
end

function adjust_nd_to_ndata(iso, ndata)
    (;ny, nk, nres) = iso
    nd = ceil(Int, ndata / ((ny)) * nres)
end


function deepcopyset(obj; kwargs...)
    obj = deepcopy(obj)
    for (k,v) in kwargs
        setfield!(obj, k, v)
    end
    return obj
end

function generate_data_traj(sim, nx, warmup = 32)
    xs, _ = ISOKANN.trajdata(sim, nx+warmup, 0)
    return ISOKANN.data_from_trajectory(xs[:, warmup+1:end]) :: ISOKANN.DataTuple
end

# generate test data of the shape (nxtest, nktest) and also trainingsdata of shape (nxtrain, iso.nk)
function generate_data_test(iso; nxtest::Integer, nxtrain::Integer, nktest::Integer, burnin=32)
    iso = deepcopyset(iso, nd = adjust_nd_to_ndata(iso, burnin+nxtrain+nxtest))
    run!(iso)
    #while size(iso.data[1], 2) < burnin+nxtrain+nxtest
    #    run!(iso)
    #end

    data = shuffledata(data_sliced(iso.data, burnin.+(1:nxtest+nxtrain)))

    data_test = data_sliced(data, 1:nxtest)
    data_train  = data_sliced(data, nxtest.+(1:nxtrain))

    data_train = koopman_sub_or_augment(data_train, nktest, iso.sim)

    return data_train, data_test
end

function data_sub_or_augment(data, nx, nk, sim::IsoSimulation)
    @assert size(data[1],2) >= nx
    data = data_sliced(data, 1:nx) # TODO: use stratified here?
    data = koopman_sub_or_augment(data, nk, sim)
end

function koopman_sub_or_augment(data, nk, sim)
    xs, ys = data
    nki = size(ys, 3)

    if nk <= nki
        data = xs, ys[:,:,1:nk]
    else
        ys = cat(ys, propagate(sim, xs, nk-nki), dims=3)
        data = xs, ys
    end
    return data
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



function plot_isos2(isos)
    plot()
    for (type, isos) in pairs(group(kv -> kv[1].type, kv->kv[2], pairs(isos)))
        for (k, isos) in pairs(group(i->i.nk, isos))
            @show k
            d = group(i->size(i.data[1],2) .* k, i->
                    i.loggers[1].losses[end],
                isos)
            d = sortkeys(d)
            d = mean.(d)
            #d = sortkeys(groupreduce(i->size(i.data[1],2) .* k, i->i.loggers[1].losses[end], mean, isos))
            plot!(collect(keys(d)), collect(values(d)), label="$k$type")
        end
    end
    plot!(yaxis=:log, xaxis=:log)
end

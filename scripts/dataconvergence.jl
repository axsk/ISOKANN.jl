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

using Distributed, ClusterManagers
@everywhere using ISOKANN
using ISOKANN: IsoSimulation, data_sliced, shuffledata
using SplitApplyCombine, Dictionaries, Plots
using StatsBase: mean, sample
using ProgressMeter
using JLD2
using Random

function addslurm(nworkers=1, ncores=4)
    ps = addprocs(SlurmManager(nworkers),
        exeflags="-t $ncores",
        env=["OPENBLAS_NUM_THREADS"=>"$(round(Int, ncores/2))"],
        cpus_per_task="$ncores",
        partition="big",
        #constraint="Gold6338",
        time="30-00:00:00")
end

function experiment(
    iso = IsoRun(
        sim=MollyLangevin(
            sys=PDB_ACEMD(),
            dt=2e-3,
            T=2e-1,
            gamma=10.,
            temp=200.),
        loggers=[],)
    )


    @time refiso, chidata = reference_chi(iso)  # 1400 sec
    @time traj, trajdata = reference_pi(iso)    # 66 sec

    isos = iso_matrix(;iso, refiso, traj)

    ##ps = addslurm(32,4)
    # find a way to run @everywhere here, eval?

    isos = @time pmap(i->(;i..., iso = run!(i.iso)), isos)
    ##rmprocs(ps)

    @save "dataconvergence.jld2" refiso chidata traj trajdata isos

    return (;isos, refiso, chidata, traj, trajdata)
end

function expl_expl_tradeoff(;
        iso = IsoRun(
            sim=MollyLangevin(
                sys=PDB_ACEMD(),
                dt=2e-3,
                T=2e-1,
                gamma=10.,
                temp=200.),
            loggers=[],),
        nk=[1,2,4,8,16,32],
        maxdata = 15_000,
        iters=20,
    )

    Random.seed!(1337)

    println("computing reference solution")
    @time refiso, chidata = reference_chi(iso; nd=30_000, nx=200, nk=1000)

    println("generating the iso matrix")
    isos = iso_matrix(;iso, refiso, iso_adapt=false, iso_traj=false,
        nk, maxdata, iters)

    println("running experiments")
    isos = @time pmap(i->(;i..., iso = run!(i.iso)), isos)

    return (;isos, refiso, chidata)
end

function thesis()
    isos, refiso, chidata = expl_expl_tradeoff()
    p = plot_dataconvergence(isos, chidata)
    savefig("scripts/dataconvergence.jl")

    return (;isos, refiso, chidata, p)
end


""" reference solution used for later chi-strat sampling """
function reference_chi(iso; nd=30_000, nx=200, nk=1_000)
    iso = deepcopyset(iso, nd = nd)

    @time run!(iso)

    liso = size(iso.data[1],2)
    liso < nx && @warn("iso has not enough samples to generate data ($liso < $nx)")

    xs = ISOKANN.stratified_x0(iso.model, iso.data[1], nx)
    println("Computing reference_chi->propagate()")
    ys = @time ISOKANN.propagate(iso.sim, xs, nk)

    chidata = (xs, ys)

    return iso, chidata
end

""" reference trajectory computation """
reference_pi(iso::IsoRun, args...) = reference_pi(iso.sim, args...)
function reference_pi(sim, nd=10_000, nx=200, nk=1)
    traj = @time ISOKANN.trajdata(sim, nd)

    inds = sample(1:size(traj, 2), nx, replace=false)
    xs = traj[:, inds]
    ys = @time ISOKANN.propagate(sim, xs, nk)

    trajdata = (xs,ys)

    return traj, trajdata
end

function iso_matrix(;
        mindata = 100,
        maxdata = 16_000,
        nx = [10,20,50,100,200,400,800,1600,3200,6400,12800, 24560],
        nk = [1,2,4],
        iters = 5,
        iso = nothing,
        refiso = nothing,
        traj = nothing,
        iso_adapt = true,
        iso_traj = true
    )
    isos = []

    for nx in nx, nk in nk, i in 1:iters
        mindata <= nx*nk <= maxdata || continue

        if iso_adapt
            push!(isos, (;type=:adapt, nx, nk,
                iso = iso_adapt(iso, nx, nk)))
        end

        if iso_traj && nk == 1
            push!(isos, (;type=:traj, nx, nk,
                iso = iso_traj(iso, nx)))
        end

        if !isnothing(refiso)
            push!(isos, (;type=:chi, nx, nk,
                iso = iso_chi(iso, nx, nk; refiso)))
        end

        if !isnothing(traj)
            push!(isos, (;type=:pi, nx, nk,
                iso = iso_pi(iso, nx, nk; traj)))
        end
    end
    return isos
end


""" sample chi-strat adaptively during training """
function iso_adapt(iso, nx, nk)
    # adjust resample size to resample up to nxmax=nx after 1/2 of the training
    ny = max(ceil(Int, nx / (iso.nd/iso.nres*(2/3))),2)
    return deepcopyset(iso;
        nxmax = nx,
        data = ISOKANN.bootstrap(iso.sim, ny, nk),
        ny = ny)
end

""" sample and train on trajectory data of length `nx` """
function iso_traj(iso, nx;)
    traj = ISOKANN.trajdata(iso.sim, nx)
    data = ISOKANN.data_from_trajectory(traj, nx)
    return deepcopyset(iso;
        nres=0, data)
end

""" trainingsdata is subsampled from a reference iso chi approximation """
function iso_chi(iso, nx, nk; refiso)
    xs = ISOKANN.stratified_x0(refiso.model, refiso.data[2], nx)
    ys = ISOKANN.propagate(iso.sim, xs, nk)
    return deepcopyset(iso;
        nres=0, data = (xs, ys))
end

""" trainingsdata is subsampled from a reference ergodic trajectory """
function iso_pi(iso, nx, nk; traj)
    inds = sample(1:size(traj, 2), nx, replace=false)
    xs = traj[:, inds]
    ys = ISOKANN.propagate(iso.sim, xs, nk)
    return deepcopyset(iso;
        nres=0, data = (xs, ys))
end

""" create a deepcopy of the object and set the kwargs fields """
function deepcopyset(obj; kwargs...)
    obj = deepcopy(obj)
    for (k,v) in kwargs
        setfield!(obj, k, v)
    end
    return obj
end

using StatsBase: mean, median

function plot_dataconvergence(isos, testdata)
    plot()
    ind = 1
    for ((type, nk), is) in sort((pairs(group(x->(x.type, x.nk), isos))))
        #nk > 2 && continue
        d = group(i->size(i.iso.data[1],2) .* nk,
                i->ISOKANN.loss(i.iso.model, testdata),
                is)
        d = sortkeys(d)
        @show length.(d), type, nk

        xs = [datasize(i.iso) for i in is]
        ys = [ISOKANN.loss(i.iso.model, testdata) for i in is]
        scatter!(xs, ys, label="", c=ind)

        global RES = d
        d = mean.(d)
        plot!(collect(keys(d)), collect(values(d)), label="$nk$type", c=ind)
        ind += 1
    end
    plot!(yaxis=:log, xaxis=:log)
end

function scatter_dataconvergence(isos, testdata)
    plot()
    for ((type, nk), isos) in pairs(group(x->(x.type, x.nk), isos))


        xs = [datasize(i.iso) for i in isos]
        ys = [ISOKANN.loss(i.iso.model, testdata) for i in isos]
        scatter!(xs, ys, label="$nk$type")
    end
    plot!(yaxis=:log, xaxis=:log)
end


function datasize(iso)
    s = size(iso.data[2])
    s[2]*s[3]
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

#= OLD STUFF

global isos=Dictionary()

function IsoConvergence(;kwargs...)
    iso = IsoRun(
            sim=MollyLangevin(sys=PDB_ACEMD(), dt=2e-3, T=5e-2, gamma=100.),
            loggers=[];
            kwargs...)
end

function IsoConvergenceRef()
    IsoConvergence(; nd=10000, opt=ISOKANN.AdamRegularized(1e-4,1e-5))
end

function analyse_dataconvergence_high()
    analyse_dataconvergence(
        nxtest = 300,
        nktest = 1000,
        mindata=100,
        maxdata=10_000,
        i=1:5,

    )
end


function analyse_dataconvergence_med()
    analyse_dataconvergence(nxtest = 200, nktest=100, mindata=100, maxdata=2000, i=1:5,
        nk=[1])
end

function analyse_dataconvergence_test()
    analyse_dataconvergence(nxtest = 10, nktest=4, mindata=1, maxdata=10, i=1:1)
end

function run_isos(isos)
    try
        p = Progress(length(isos))
        pmap(run!, isos)
        for (k,v) in pairs(isos)
            @show k
            length(v.losses) > 0 && continue
            run!(v)
            next!(p)
        end

        save("isos.jld2", "isos", isos, "data", (data_train, data_test))
    catch e
        isa(e, InterruptException) || rethrow(e)
    finally
        return (;isos, data_train, data_test)
    end
end

function run_parallel(isos)
    pmap(run!, isos)
end

function isos_dataconvergence(;
        nxtest  = 100,    # number of x samples for ref data
        nktest = 10_000, # number of k samples for ref data
        mindata = 100,
        maxdata = 16_000,
        trajlength = maxdata,
        isos = Dictionary(),
        iso = IsoConvergence(),
        nx = [10,20,50,100,200,500,1000,2000,5000,10000,20000],
        nk = [1,2,4,8],
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

end

function adjust_nd_to_ndata(iso, ndata)
    (;ny, nk, nres) = iso
    nd = ceil(Int, ndata / ((ny)) * nres)
end




function generate_data_traj(sim, nx, warmup = 32)
    xs = ISOKANN.trajdata(sim, nx+warmup)
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
using StatsBase:median


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

=#

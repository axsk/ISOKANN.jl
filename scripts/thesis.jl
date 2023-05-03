using Distributed

# supposed to be running on 128 cores
if ENV["SLURM_CPUS_PER_TASK"] == "128" && length(workers()) < 32
    addprocs(32, env=["OPENBLAS_NUM_THREADS"=>"2"], exeflags="-t 4")
end

include("dataconvergence.jl")

function thesis()

    iso = IsoRun(
            sim=MollyLangevin(
                sys=PDB_ACEMD(),
                dt=2e-3,
                T=2e-1,
                gamma=10.,
                temp=200.),
            loggers=[],
            opt=AdamRegularized(1e-3,1e-3))

    isos, refiso, chidata = expl_expl_tradeoff()
    @save "scripts/thesis.jld2" isos refiso chidata
    p = plot_dataconvergence(isos, chidata)
    savefig("scripts/dataconvergence.png")

    return (;isos, refiso, chidata, p)
end

#thesis()

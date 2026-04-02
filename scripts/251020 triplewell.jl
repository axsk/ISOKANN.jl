using ISOKANN
using Flux, Plots


isos = []
df = DataFrame()


s0 = 1  # temperature for exploration
s1 = 6.67^(-1/2)  # temperature of system

sim0 = Triplewell(sigma=s0) # hot simulation to get starting points
@time "horizontal sampling" x0 = laggedtrajectory(sim0, 1000, lagtime=10) # 1000 lags a 100 timeunits

# x0 = ISOKANN.picking(x0, 1000)

sim = Triplewell(sigma=s1) # "cold simulation", as is Metzner et al.
@time "vertical sampling" data = SimulationData(sim, x0, 8)  # 10 koopman samples  each
@time "validation data" validation = @time SimulationData(sim, laggedtrajectory(sim0, 100, lagtime=1000), 100)

log_residual_ritz() =
    ISOKANN.FunctionLogger(name="res_rr", logevery=10) do iso
        try
            ISOKANN.residual_ritz(iso, validation).relres
        catch
            NaN
        end
    end

log_residual_subspace() =
    ISOKANN.FunctionLogger(name="res_ss", logevery=10) do iso
        ISOKANN.residual_subspace(iso, validation).relres
    end

transforms = [ISOKANN.TransformISA(), 
    ISOKANN.TransformISA(),
    ISOKANN.TransformCross(;npoints=length(data), maxcols=3), 
    ISOKANN.TransformCross(;npoints=length(data), maxcols=6),
    ISOKANN.TransformCross(;npoints=length(data), maxcols=9)]

iters = [(5000, 1), (1000, 5), (500, 10), (250, 20)]
iters = [iters; map(i -> (i[1]*2, i[2]),iters)]

configs = []
for transform in transforms, 
    iters in iters,
    lr in [1e-3, 1e-4],
    opt in [AdamRegularized, NesterovRegularized],
    reg in [1e-3, 1e-4, 1e-5, 1e-6],
    layers in [[2,8,8,3], [2,8,8,8,3], [2,4,4,4,3], [2,16,16,3]],
    activation in [Flux.leakyrelu, Flux.sigmoid]
                    
    push!(configs, (;transform, iters, reg, layers, activation, lr, opt))
end

function trial()
    c = rand(configs)
    (; transform, iters, reg, layers, activation, lr, opt) = c
    transform isa ISOKANN.TransformCross1 && ISOKANN.reset!(transform)
    @show c




    iso = Iso(data;
        model=ISOKANN.densenet(;layers, layernorm=false, activation),
        opt=opt(lr, reg),
        transform,
        gpu=false, 
        minibatch=100, 
        autoplot=0,
        validation,
        loggers=[log_residual_ritz(), log_residual_subspace()]
        )

    @time "training" run!(iso, iters...)
    @show iso.loggers[2].values[end] |> sum

    try
    plot_training(iso) |> display
    catch 
    end

    push!(isos, (;c..., iso))
    push!(df, (; c..., iso), cols=:union)
end

map(isos) do x
    a,iso = x
    iso.loggers[2].values[end] |> sum
end

function dfstats(df)
    df.rr = map(iso->sum.(iso.loggers[1].values), df.iso)
    df.ss = map(iso->sum.(iso.loggers[2].values), df.iso)
    df.ssmin = map(s->minimum(filter(!isnan, s)), df.ss)
    df.sslast = map(last, df.ss)
    df.rrmin = map(s->minimum(filter(!isnan, s)), df.rr)
    df.rrlast = map(last, df.rr)
    sort!(df, :sslast)
end




ISOKANN.plot_targets(iso)

plot_residuals() = plot(reduce(vcat, transpose.(residuals)))
plot_residuals()



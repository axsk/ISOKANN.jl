@deprecate Iso2 Iso

@kwdef mutable struct Iso{M,D}
    model::M
    opt
    data::D
    transform
    losses = Float64[]
    loggers = Any[autoplot(1)]
    minibatch = 100
end

"""
    Iso(data; opt=NesterovRegularized(), model=defaultmodel(data), gpu=false, kwargs...)

"""
function Iso(data;
    opt=NesterovRegularized(),
    model=defaultmodel(data),
    gpu=CUDA.has_cuda(),
    autoplot=0,
    validation=nothing,
    loggers=Any[],
    transform = nothing,
    kwargs...)

    opt = Flux.setup(opt, model)
    if isnothing(transform)
        if outputdim(model) == 1
            transform = TransformShiftscale()
        else
            transform = TransformISA()
        end
    end

    loggers = Vector{Any}(loggers)
    autoplot > 0 && push!(loggers, ISOKANN.autoplot(autoplot))
    isnothing(validation) || push!(loggers, ValidationLossLogger(data=validation))

    iso = Iso(; model, opt, data, transform, loggers, kwargs...)
    gpu && (iso = ISOKANN.gpu(iso))
    return iso
end

"""
    Iso(sim::IsoSimulation; nx=100, nk=10, nd=1, kwargs...)

Convenience constructor which generates the `SimulationData` from the simulation `sim`
and constructs the Iso object. See also Iso(data; kwargs...)

## Arguments
- `sim::IsoSimulation`: The `IsoSimulation` object.
- `nx::Int`: The number of starting points.
- `nk::Int`: The number of koopman samples.
- `nout::Int`: Dimension of the χ function.
"""
Iso(sim::IsoSimulation; nx=100, nk=2, kwargs...) = Iso(SimulationData(sim, nx, nk); kwargs...)


#Iso(iso::IsoRun) = Iso(iso.model, iso.opt, iso.data, TransformShiftscale(), iso.losses, iso.loggers, iso.minibatch)

"""
    run!(iso::Iso, n=1, epochs=1)

Run the training process for the Iso model.

# Arguments
a `iso::Iso`: The Iso model to train.
- `n::Int`: The number of (outer) Koopman iterations.
- `epochs::Int`: The number of (inner) epochs to train the model for each Koopman evaluation.
"""
function run!(iso::Iso, n=1, epochs=1; showprogress=true)
    p = ProgressMeter.Progress(n)
    iso.opt isa Optimisers.AbstractRule && (iso.opt = Optimisers.setup(iso.opt, iso.model))

    for _ in 1:n
        #xs, ys = getobs(iso.data)
        #target = isotarget(iso.model, xs, ys, iso.transform)
        target = isotarget(iso)
        xs = features(iso.data)
        for i in 1:epochs
            loss = train_batch!(iso.model, xs, target, iso.opt, iso.minibatch)
            push!(iso.losses, loss)
        end

        diagnostics = [("loss", iso.losses[end]), ("n", length(iso.losses)), ("data", size(features(iso)))]
        for logger in iso.loggers
            log!(logger; iso, subdata=nothing)
            d = diagnostic(logger)
            isnothing(d) || push!(diagnostics, d)
        end

        showprogress && ProgressMeter.next!(p; showvalues=diagnostics)
    end
    return iso
end

diagnostic(logger) = nothing


@kwdef mutable struct FunctionLogger
    f # ::Iso -> ::Any
    name
    values = []
    iters = Int[]
    logevery = 1
end

function FunctionLogger(f; kwargs...)
    FunctionLogger(;f, kwargs...)
end

function log!(l::FunctionLogger; iso, kw...)
    length(iso.losses) % l.logevery == 0 || return
    push!(l.values, l.f(iso))
    push!(l.iters, length(iso.losses))
    return
end

diagnostic(l::FunctionLogger) = (l.name, if length(l.values) > 0
    v = l.values[end] 
    #round.(v, digits=5)
else
     nothing
end )

function ValidationLogger(valdata)
    FunctionLogger(name="validation loss") do iso
        validationloss(iso, valdata)
    end
end

@kwdef struct ValidationLossLogger{T}
    data::T
    losses = Float32[]
    iters = Int[]
    logevery = 10
end

diagnostic(l::ValidationLossLogger) = ("validation loss", length(l.losses) > 0 ? l.losses[end] : nothing)

gpu(v::ValidationLossLogger) = ValidationLossLogger(gpu(v.data), v.losses, v.iters, v.logevery)

function validationloss_old(iso, data)
    xs, ys = getobs(data)
    kx = iso.model(xs) |> vec
    ky = StatsBase.mean(iso.model(ys), dims=2) |> vec

    kx = shiftscale(kx)

    unit = fill!(copy(kx), 1)
    kx = hcat(kx, unit)
    ky = hcat(ky, unit)



    #  kx = ky * A
    return mean(abs2, (ky*(ky\kx)-kx)[:, 1])
end

function validationloss(iso, valdata)
    xs, ys = getobs(iso.data)
    vx, vy = getobs(valdata)
    c = iso.model(vx) |> vec
    k1 = expectation(iso.model, vy) |> vec
    k2 = expectation(iso.model, ys) |> vec # we use all data to estimate the shift-scale even for sparse validation coverage
    SKc = shiftscale([k1; k2])[1:length(c)]
    return mean(abs2, c - SKc)
end


function log!(v::ValidationLossLogger; iso, kw...)
    length(iso.losses) % v.logevery == 0 || return
    vl = validationloss(iso, v.data)
    push!(v.iters, length(iso.losses))
    push!(v.losses, vl)
    return
end

function train_batch!(model, xs::AbstractMatrix, ys::AbstractMatrix, opt, minibatch; shuffle=true)
    batchsize = minibatch == 0 || size(xs, 2) < minibatch ? size(ys, 2) : minibatch
    data = Flux.DataLoader((xs, ys); batchsize, shuffle)
    ls = 0.0
    Flux.train!(model, data, opt) do m, x, y
        l = sum(abs2, m(x) .- y)
        ls += l
        l / numobs(x)
    end
    return ls / numobs(xs)
end

chis(iso::Iso) = iso.model(features(iso.data))
chicoords(iso::Iso, xs) = iso.model(features(iso.data, iscuda(iso.model) ? gpu(xs) : xs))
#isotarget(iso::Iso) = isotarget(iso.model, getobs(iso.data)..., iso.transform)

coords(iso::Iso) = coords(iso.data)
features(iso::Iso) = features(iso.data)
propcoords(iso::Iso) = propcoords(iso.data)
propfeatures(iso::Iso) = propfeatures(iso.data)


# add new datapoints to iso, starting at positions `coords`
addcoords!(iso::Iso, coords) = (iso.data = addcoords(iso.data, coords); nothing)
laggedtrajectory(iso::Iso, n) = laggedtrajectory(iso.data, n)
resample_kde!(iso, ny; kwargs...) = (iso.data = resample_kde(iso.data, iso.model, ny; kwargs...))
addcoords!(iso::Iso, ny::Integer) = addcoords!(iso, laggedtrajectory(iso.data.sim, ny, x0=iso.data.coords[1][:, end]))
resample_strat!(iso, ny; kwargs...) = (iso.data = resample_strat(iso.data, iso.model, ny; kwargs...))

#Optimisers.adjust!(iso::Iso; kwargs...) = Optimisers.adjust!(iso.opt; kwargs...)
#Optimisers.setup(iso::Iso) = (iso.opt = Optimisers.setup(iso.opt, iso.model))

# TODO: should this handled via @functor?
gpu(iso::Iso) = Iso(Flux.gpu(iso.model), Flux.gpu(iso.opt), Flux.gpu(iso.data), Flux.gpu(iso.transform), iso.losses, gpu.(iso.loggers), iso.minibatch)
cpu(iso::Iso) = Iso(Flux.cpu(iso.model), Flux.cpu(iso.opt), Flux.cpu(iso.data), Flux.cpu(iso.transform), iso.losses, iso.loggers, iso.minibatch)

function Base.show(io::IO, mime::MIME"text/plain", iso::Iso)
    println(io, "Iso: ")
    println(io, " model: $(iso.model.layers)")
    println(io, first(" transform: $(iso.transform)", 160))
    println(io, " opt: $(optimizerstring(iso.opt))")
    println(io, " minibatch: $(iso.minibatch)")
    println(io, " loggers: $(length(iso.loggers))")
    println(io, " data: $(size.(getobs(iso.data))), $(typeof(getobs(iso.data)))")
    length(iso.losses) > 0 && println(io, " loss: $(iso.losses[end]) (length: $(length(iso.losses)))")
end

"""
    runadaptive!(iso; generations=1, nx=10, iter=100, cutoff=Inf)

Train iso with adaptive sampling. Sample `kde` new data points followed by `iter` isokann iterations and repeat this `generations` times.
`cutoff` specifies the maximal data size, after which new data overwrites the oldest data.
`unique` enforces resampling from yet unsampled `ys` only.
"""
function runadaptive!(iso; generations=1, iter=100, cutoff=Inf, kde=1, unique=true)
    p = ProgressMeter.Progress(generations)
    t_kde = 0.0
    t_train = 0.
    t_extra = 0.0
    for g in 1:generations
        GC.gc()

        t_kde += @elapsed ISOKANN.resample_kde!(iso, kde; unique)

        if length(iso.data) > cutoff
            iso.data = iso.data[end-cutoff+1:end]
        end

        t_train += @elapsed run!(iso, iter, showprogress=generations==1)

        #plot_training(iso)|>display

        ProgressMeter.next!(p;
            showvalues=() -> [
                (:generation, g),
                (:loss, iso.losses[end]),
                (:iterations, length(iso.losses)),
                (:data, size(propfeatures(iso.data))),
                ("t_train, t_extra, t_kde", (t_train, t_extra, t_kde)),
                ("simulated time", "$(simulationtime(iso))"), #TODO: doesnt work with cutoff
                (:macrorates, exit_rates(iso))],
            ignore_predictor=true)
    end
end

log!(f::Function; kwargs...) = f(; kwargs...)
log!(logger::NamedTuple; kwargs...) = :call in keys(logger) && logger.call(; kwargs...)



""" empirical shift-scale operation """
shiftscale(ks) =
    let (a, b) = extrema(ks)
        (ks .- a) ./ (b - a)
    end

""" compute the chi exit rate as per Ernst, Weber (2017), chap. 3.3 """
function chi_exit_rate(x, Kx, tau)
    @. shiftscale(x, p) = p[1] * x + p[2]
    γ1, γ2 = LsqFit.coef(LsqFit.curve_fit(shiftscale, vec(x), vec(Kx), [1, 0.5]))
    α = -1 / tau * Base.log(γ1)
    β = α * γ2 / (γ1 - 1)
    return α + β
end

chi_exit_rate(iso::Iso) = chi_exit_rate(vec.(cpu.(chi_kchi(iso)))..., lagtime(iso.data.sim))


function exit_rates(x, kx, tau)
    o = ones(length(x))
    x = vec(x)
    kx = vec(kx)
    P = [x o .- x] \ [kx o .- kx]
    return -1 / tau .* [p > 0 ? Base.log(p) : NaN for p in diag(P)]
end

# generalization of above approach to higher dimensions
function exit_rates(x::AbstractMatrix, kx::AbstractMatrix, tau)
    P = x \ kx
    return -1 / tau .* [p > 0 ? Base.log(p) : NaN for p in diag(P)]
end

exit_rates(iso::Iso) = exit_rates(vec.(cpu.(chi_kchi(iso)))..., lagtime(iso.data.sim))

function koopman_variance(iso, ys=iso.data.coords[2])
    chi = chicoords(iso, ys)
    i, k, n = size(chi)
    sum(abs2, chi .- (sum(chi, dims=2) ./ k)) ./ i ./ n
end

"""
    simulationtime(iso::Iso)

print and return the total simulation time contained in the data of `iso` in nanoseconds.
"""
function simulationtime(data::SimulationData)
    _, k, n = size(data.features[2])
    t = k * n * lagtime(data.sim)
    #println("$t nanoseconds")  # TODO: should we have nanoseconds here when we have picoseconds everywhere else?
    return t
end

simulationtime(iso::Iso) = simulationtime(iso.data)

"""
    savecoords(path::String, iso::Iso, coords::AbstractMatrix=coords(iso.data); sorted=true, aligned=true)

Save the coordinates of the specified matrix of coordinates to a file, using the molecule in `iso` as a template.
If `sorted` the sort the coordinates by their increasing χ value. If `align` then align each frame to the previous one.
"""
function savecoords(path::String, iso::Iso, coords::AbstractMatrix=coords(iso.data); sorted=true, aligned=true)
    if sorted
        coords = coords[:, cpu(sortperm(dropdims(chicoords(iso, coords), dims=1)))]
    end
    if aligned
        coords = aligntrajectory(coords)
    end
    savecoords(path, iso.data.sim, coords)
end

"""
    saveextrema(path::String, iso::Iso)

Save the two extermal configurations (metastabilities) to the file `path`.
"""
function saveextrema(path::String, iso::Iso)
    c = vec(chis(iso))
    savecoords(path, iso, [argmin(c), argmax(c)])
end

"""
    save(path::String, iso::Iso)

Save the complete Iso object to a JLD2 file """
function save(path::String, iso::Iso)
    iso = cpu(iso)
    JLD2.jldsave(path; iso)
end

"""
    load(path::String, iso::Iso)

Load the Iso object from a JLD2 file
Note that it will be loaded to the CPU, even if it was saved on the GPU.
An OpenMMSimulation will be reconstructed anew from the saved pdb file.
"""
function load(path::String)
    iso = JLD2.load(path, "iso")
    return iso
end

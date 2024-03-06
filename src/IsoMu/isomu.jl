""" ingestion and computation of the muop-receptor reaction path """

"""
    IMu

Combine ISOKANN with a DataLink

# Fields
- `data::DataLink`: reference to the data
- `learnrate::Float64=0.001`: learnrate for ADAM
- `regularization::Float64=0.0001`: regularization of the NN
- `hidden::Vector{Int}=[300, 100, 10]`: Size of hidden layers

# Passed to ISOKANN.IsoRun
- `nx`: Number of (χ-sub-)samples of the data per iteration
"""
mutable struct IMu
    iso
    data
end

function IMu(data::Union{DataLink,Vector{DataLink}};
    learnrate::Real=0.001,
    regularization::Real=0.0001,
    networkargs=(),
    gpu=true,
    kwargs...)

    idata = isodata(data)
    #model = getmodel(idata, layers=nlayers)
    model = pairnet(idata; networkargs...)

    # iso = ISOKANN.IsoRun(;
    #     nx=0, # no subsampling
    #     np=1, # no subreuse
    #     nl=1, # no powerreuse
    #     nres=0, # no resampling
    #     minibatch=200,
    #     sim=nothing,
    #     model=model,
    #     data=idata,
    #     opt=AdamRegularized(learnrate, regularization),
    #     loggers=[ISOKANN.autoplot(5)],
    #     kwargs...)

    iso = ISOKANN.Iso2(idata,
        opt=AdamRegularized(learnrate, regularization),
        model=model,
        transform=ISOKANN.TransformShiftscale(),
        minibatch=200,
        loggers=[ISOKANN.autoplot(1)],
    )


    mu = IMu(iso, data)
    gpu && gpu!(mu)
    return mu
end

""" see also IMu """
isokann(link::Union{DataLink,Vector{DataLink}}; kwargs...) = IMu(link; kwargs...)

function train!(mu::IMu, iter=nothing)
    ISOKANN.run!(mu.iso, iter)
end

getxs(mu::IMu) = mu.iso.data[1] |> collect
getys(mu::IMu) = mu.iso.data[2] |> collect
model(mu::IMu) = mu.iso.model
model(mu::IMu, xs) = mu.iso.model(xs) |> collect
chi(mu::IMu) = model(mu, mu.iso.data[1]) |> vec |> collect
coords(mu::IMu) = coords(mu.data)
pdbfile(mu::IMu) = pdbfile(mu.data)
MLUtils.numobs(mu::IMu) = size(getxs(mu), 2)


"""
    reactive_path(mu::IMu; kwargs...) where {T} -> (Vector{Int}, Matrix)

Compute the shortest Onsager-Machlup path through the data with the χ value as time.

# Arguments
- `mu`: contains the data and χ-vales

# Keywords
- `sigma::Float64=0.1`: the temperature for the allowed trajectory
- `window::Interval`: which coordinates to consider for the path
- `method`: Either one of
            `QuantilePath(quant)` for a top-bot quartile path
            `FromToPath(from::Int, to::Int)` specific frame numbers
            `FullPath()` from start to end

# Returns
- `inds::Vector`: indices of the selected path
- `path::Matrix`: aligned coordinates of the path
"""
# TODO: this should dispatch to ISOKANN.reactive_path
#=function reactive_path(
    mu::IMu;
    sigma=0.1,
    window=1:numobs(mu),
    kwargs...
)

    xi = model(mu, getxs(mu)[:, window]) |> vec
    co = flatten3d(coords(mu))[:, window]

    ids, _ = reactive_path(xi, co ./ norm(co, Inf), sigma; kwargs...)

    ids = isincreasing(ids) ? ids : reverse(ids)
    path = aligntrajectory(co[:, ids])
    ids = window[ids]

    return ids, path
end

function flatten3d(data)
    d, n, m = size(data)
    reshape(data, d * n, m)
end


# heuristic answer to the question whether a sequence is increasing
isincreasing(x) = sum(diff(x) .> 0) > length(x) / 2

"""
    save_reactive_path(mu::IMu; out=joinpath(outdir(mu.data), "reactive_path.pdb"),
    kwargs...) -> Vector{Int}

Compute the reactive path for the IMu object and save it as pdb
(defaulting to the outdir of mu) see also `reactive_path` for further arguments

# Arguments
- `mu::IMu`: contains data and chi function

# Keywords
- `out`: filename of the pdb
- passes through passwords to `reactive_path`

# Returns
- `inds::Vector{Int}`: the indices of the selected path
"""
# TODO: this should dispatch to ISOKANN.reactive_path
function save_reactive_path(mu::IMu;
    out=joinpath(outdir(mu.data), "reactive_path.pdb"),
    kwargs...)

    mkpath(dirname(out))
    ids, path = reactive_path(mu; kwargs...)
    println("Found reactive path of length $(length(ids))")
    xi = model(mu, getxs(mu)) |> vec
    plot_reactive_path(ids, xi) |> display
    writechemfile(out, path, source=pdbfile(mu))
    println("saved to $out")
    return ids
end
=#

function ISOKANN.save_reactive_path(mu::IMu;
    out=joinpath(outdir(mu.data), "reactive_path.pdb"),
    sigma=0.1,
    normalize=true,
    kwargs...)

    c = coords(mu) |> ISOKANN.flattenfirst
    source = pdbfile(mu)

    ISOKANN.save_reactive_path(mu.iso, c; source, out, sigma, normalize, kwargs...)
end




function benchmark()
    data = DataLink("datalink/7UL4_no_lig_ds_292_321_100ns")
    mu = IMu(data)
    mu.iso.loggers = []
    @elapsed train!(mu, 10)
end

function meanvelocity(mu::IMu)
    mean(abs2, diff(getxs(mu), dims=2), dims=1)' |> plot
    title!(mu.data.dir)
end

function adjust!(mu::IMu, args...; kwargs...)
    mu.iso.opt isa NamedTuple || error("have to train! model once before tuning parameters")
    Flux.adjust!(mu.iso.opt, args...; kwargs...)
end

function gpu!(mu::IMu)
    mu.iso = ISOKANN.gpu(mu.iso)
    return mu
end

function cpu!(mu::IMu)
    mu.iso = ISOKANN.cpu(mu.iso)
    return mu
end

function paperplot(mu::IMu)
    c = chi(mu)
    scatter(1:length(c), c,
        xlabel="frame",
        ylabel="\\chi",
        legend=false,
        markersize=1,
        xticks=0:1000:10000,
        markerstrokewidth=0)
    savefig("out/paperplot.pdf")
    plot!()
end

# to also plot the reactive path, pass its `ids` as argument to paperplots
function paperplots(mu::IMu, ids)
    paperplot(mu)
    plot!(ids, chi(mu)[ids],
        linealpha=1,
        linecolor="Salmon")
    savefig("out/paperplot2.pdf")
    plot!()
end

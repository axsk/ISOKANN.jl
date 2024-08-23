"""
scatter plot of all first "O" atoms of the starting points `xs`
as well as the "O" atoms from the koopman samples to the first point from `ys`
"""
function plotatoms!(xs, ys, model=nothing)
    dim, nx, nk = size(ys)
    i = 3 * (9 - 1) # first O atom
    cx = 1
    cy = 1

    if !isnothing(model)
        cx = model(xs) |> vec
        cy = model(ys[:, 1, :]) |> vec
    end

    a = reshape(xs[:, 1], 3, :)
    #scatter!(a[1,:], a[2,:], label="atoms of first x0") # all atoms of first x
    scatter!(xs[i+1, :], xs[i+2, :], label="x0", marker_z=cx, zlims=(0, 1))
    scatter!(ys[i+1, 1, :], ys[i+2, 1, :], label="ys",
        marker_z=cy, markershape=:xcross)
    return plot!()
end

""" combined plot of loss graph and current atoms """
function plotlossdata(losses, data, model=nothing)
    p1 = plot(losses, yaxis=:log)
    p2 = plot()
    plotatoms!(data..., model)
    plot(p1, p2)
end



## Plotting

function plot_training(iso; subdata=nothing)
    (; losses, data, model) = iso

    !isnothing(subdata) && (data = subdata)

    p1 = plot(losses[1:end], yaxis=:log, title="loss", label="trainloss", xlabel="iter")

    for v in filter(x -> isa(x, ValidationLossLogger), iso.loggers)
        p1 = plot!(v.loss, label="validation")
    end
    #=
    for tl in filter(l -> isa(l, TrainlossLogger), iso.loggers)
        if length(tl.losses) > 1
            plot!(tl.xs, tl.losses, label="validationloss")
        end
    end
    =#

    xs, ys = getobs(data)
    p2 = plot_chi(iso)



    p3 = scatter_chifix(data, model)
    #annotate!(0,0, repr(iso)[1:10])
    ps = [p1, p2, p3]
    for l in iso.loggers
        if l isa NamedTuple
            push!(ps, l.plot())
        end
    end
    plot(ps..., layout=(length(ps), 1), size=(400, 300 * length(ps)))
end

function plot_chi(iso; target=true)
    xs = getxs(iso.data)
    chi = iso.model(xs) |> cpu
    xs = xs |> cpu

    if size(xs, 1) == 1
        scatter(xs', chi', xlabel="x", ylabel="χ")
    elseif size(xs, 1) == 2
        scatter(xs[1, :], xs[2, :], marker_z=chi', label="", xlabel="x", ylabel="y", cbar_title="χ")
    elseif size(xs, 1) == 66  # TODO: dispatch on simulation
        scatter_ramachandran(xs, chi)
    else
        scatter(chi'; ylims=autolims(chi), xlabel="#")
        target && scatter!(isotarget(iso)' |> cpu)
    end


end

function autolims(chi)
    e = extrema(chi)
    if e[1] > 0 && e[2] < 1 && e[2] - e[1] > 0.2
        return (0, 1)
    else
        return :auto
    end
end

""" fixed point plot, i.e. x vs model(x) """
function scatter_chifix(data, model)
    xs, ys = getobs(data)
    target = koopman(model, ys) |> vec |> Flux.cpu
    xs = model(xs) |> vec |> Flux.cpu
    lim = autolims(xs)
    scatter(xs, target, markersize=2, xlabel="χ", ylabel="Kχ", xlims=lim, ylims=lim)
    #scatter(xs, target .- xs, markersize=2, xlabel="χ", ylabel="Kχ")
    #scatter(target .- xs, markersize=2, xlabel="χ", ylabel="Kχ")
    #plot!([minimum(xs), maximum(xs)], [minimum(target), maximum(target)], legend=false)
end

# DEPRECATED
#=
function inspecttrajectory(sys)
    @time sol = solve(SDEProblem(sys))
    x = reduce(hcat, sol.u)
    scatter_ramachandran(x) |> display
    return x
end
=#

# good colors
# berlin, delta, roma, tofino, tokyo

scatter_ramachandran(iso::Iso) = scatter_ramachandran(getcoords(iso.data) |> cpu, iso.model(getxs(iso.data)) |> cpu |> vec)

scatter_ramachandran(x, model; kwargs...) = scatter_ramachandran(x, vec(model(x)))
scatter_ramachandran(x, mat::Matrix; kwargs...) = plot(map(eachrow(mat)) do row
    scatter_ramachandran(x, vec(row))
end...)

function scatter_ramachandran(x::AbstractMatrix, z::Union{AbstractVector,Nothing}=nothing; kwargs...)
    ph = phi(x)
    ps = psi(x)
    scatter(ph, ps, marker_z=z, xlims=[-pi, pi], ylims=[-pi, pi],
        markersize=3, markerstrokewidth=0, markeralpha=1, markercolor=:tofino, legend=false,
        xlabel="\\phi", ylabel="\\psi", title="Ramachandran", ; kwargs...
    )
end

### Simplex plotting - should be in PCCAPlus.jl
using Plots

function euclidean_coords_simplex()
    s1 = [0, 0, 0]
    s2 = [1, 0, 0]
    s3 = [0.5, sqrt(3) / 2, 0]
    s4 = [0.5, sqrt(3) / 4, sqrt(3) / 2]
    hcat(s1, s2, s3, s4)'
end

function plot_simplex(; n=2, kwargs...)
    c = euclidean_coords_simplex()
    c = c[1:(n+1), 1:n]
    for i in 1:(n+1), j in i+1:(n+1)
        plot!(eachcol(c[[i, j], :])...; kwargs...)
    end
    plot!()
end

function bary_to_euclidean(x::AbstractMatrix)
    n = size(x, 2)
    x * euclidean_coords_simplex()[1:n, 1:(n-1)]
end

function scatter_chi!(chi; kwargs...)
    c = bary_to_euclidean(chi)
    scatter!(eachcol(c)...; kwargs...)
end

scatter_chi(chi; kwargs...) = (plot(); scatter_chi!(chi; kwargs...))

function plot_path(chi, path; kwargs...)
    plot!(eachcol(bary_to_euclidean(chi[path, :]))...; kwargs...)
end


## from former iso2.jl
### Visualization

function vismodel(model, grd=-2:0.03:2; xs=nothing, ys=nothing, float=0.01, kwargs...)
    defargs = (; markeralpha=0.1, markersize=0.5, markerstrokewidth=0)
    dim = ISOKANN.inputdim(model)
    if dim == 1
        plot(grd, model(collect(grd)')')
    elseif dim == 2
        p = plot()
        g = makegrid(grd, grd)
        y = model(g)
        for i in 1:ISOKANN.outputdim(model)
            yy = reshape(y[i, :], length(grd), length(grd))
            surface!(p, grd, grd, yy, clims=(0, 1); kwargs...)
        end
        if !isnothing(ys)
            yy = reshape(ys, 2, :)
            scatter!(eachrow(yy)..., maximum(model(yy), dims=1) .+ float |> vec; markercolor=:blue, defargs..., kwargs...)
        end
        !isnothing(xs) && scatter!(eachrow(xs)..., maximum(model(xs), dims=1) .+ float |> vec; markercolor=:red, kwargs...)
        plot!(; kwargs...)
    end
end

function makegrid(x, y)
    A = zeros(Float32, 2, length(x) * length(y))
    i = 1
    for x in x
        for y in y
            A[:, i] .= (x, y)
            i += 1
        end
    end
    A
end

## DEPRECATED
function vis_training(; model, data, target, losses, others...)
    p1 = visualize_diala(model, data[1],)
    p2 = scatter(eachrow(target)..., markersize=1)
    #p3 = plot(losses, yaxis=:log)
    plot(p1, p2)#, p3)
end

function visualize_diala(mm, xs; kwargs...)
    p1, p2 = ISOKANN.phi(xs), ISOKANN.psi(xs)
    plot()
    for chi in eachrow(mm(xs))
        markersize = max.(chi .* 3, 0.01)
        scatter!(p1, p2, chi; kwargs..., markersize, markerstrokewidth=0)
    end
    plot!()
end

# note there is also plot_callback in isokann.jl
function autoplot(secs=10)
    Flux.throttle(
        function plotcallback(; iso, subdata, kwargs...)
            p = plot_training(iso; subdata)
            try
                display(p)
            catch e
                e isa InterruptException && rethrow(e)
                @warn "could not print ($e)"
            end
        end, secs)
end

function plot_reactioncoords(iso)
    coords = getcoords(iso.data)
    dim = size(coords, 1)
    if dim == 66  # alanine dipeptide
        chi = chis(iso)
        scatter_ramachandran(coords, chi)
    elseif dim == 219
        inds = (41, 49)
        inds = (20, 62)
        i = findfirst(CartesianIndex(inds...) .== halfinds(73))
        scatter(iso.data.data[1][i, :] |> vec, chis(iso) |> vec)
    end
end

pdb(id::String) = Base.download("https://files.rcsb.org/download/$id.pdb")
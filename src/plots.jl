

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
    #=
    for tl in filter(l -> isa(l, TrainlossLogger), iso.loggers)
        if length(tl.losses) > 1
            plot!(tl.xs, tl.losses, label="validationloss")
        end
    end
    =#

    xs, ys = data
    p2 = plot_chi(xs, Flux.cpu(model(xs)))

    scatter!(isotarget(iso) |> cpu |> vec)

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

function plot_chi(xs, chi)
    if size(xs, 1) == 1
        scatter(vec(xs), chi)
    elseif size(xs, 1) == 2
        scatter(xs[1, :], xs[2, :], marker_z=chi, label="")
    elseif size(xs, 1) == 66
        scatter_ramachandran(xs, chi)
    else
        scatter(chi'; ylims=autolims(chi))
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
    xs, ys = data
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
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

function plot_training(iso; maxpoints=0)
    (; losses, data, model) = iso


    p1 = plot_losses(iso; maxpoints)

    p2 = plot_chi(iso; maxpoints)

    p3 = scatter_chifix(iso; maxpoints)
    #annotate!(0,0, repr(iso)[1:10])
    ps = [p1, p2, p3]
    for l in iso.loggers
        if l isa NamedTuple
            push!(ps, l.plot())
        end
    end
    plot(ps..., layout=(length(ps), 1), size=(400, 300 * length(ps)), fmt=:png)
end

function filter_maxpoints(x::AbstractVector, n)
    (n == 0 || length(x) <= n) && return x
    x[round.(Int, LinRange(1, length(x), n))]
end

function filter_maxpoints(x::AbstractMatrix, n)
    (n == 0 || size(x, 2) <= n) && return x
    x[:, round.(Int, LinRange(1, size(x, 2), n))]
end

function plot_losses(iso; maxpoints=0)
    losses = iso.losses
    ix = 1:length(losses)

    ix, losses = filter_maxpoints.((1:length(losses), losses), maxpoints)
    p = plot(ix, losses, yaxis=:log, title="loss", label="trainloss", xlabel="iter", ylabel="squared loss")
    
    for v in filter(x -> isa(x, ValidationLossLogger), iso.loggers)
        ix, vlosses = filter_maxpoints.((v.iters, v.losses), maxpoints)
        plot!(p, ix, vlosses, label="validation")
    end
    return p
end


function scatter_data(iso, x; kwargs...)
    xs, ys = eachrow(coords(iso))
    plot([scatter(xs, ys, marker_z=c; label=nothing, hover=c, kwargs...) for c in eachrow(x)]..., layout=(1,size(x, 1)); )
end

function plot_targets(iso)
    c = chis(iso)
    k = koopman(iso)
    t = isotarget(iso)
    clims = extrema(vcat(c,k,t))
    plot(
        scatter_data(iso, c, title="chi"; ),
        scatter_data(iso, k, title="Kchi"; ),
        scatter_data(iso, t, title="target"; ),
        layout=(3, 1), size=(300* size(c,1), 600 ), cbar=false)
end

function plot_chi(iso; target=false, maxpoints = 0)
    xs = features(iso.data)
    chi = iso.model(xs) |> cpu
    xs = xs |> cpu
    ix = 1:size(xs, 2)
    ix, xs, chi = filter_maxpoints.((ix, xs, chi), maxpoints)

    if size(xs, 1) == 1
        # 1D space: plot chi over x axis
        scatter(xs', chi', xlabel="x", ylabel="χ")
    elseif size(xs, 1) == 2
        # 2D space: scatter plot of chi as color over the 2D space
        if target == true
            chi = isotarget(iso) |> cpu
        end
        
        plot([scatter(xs[1, :], xs[2, :], marker_z=c, label=nothing, xlabel=nothing, ylabel=nothing, cbar_title="χ", cbar=false) for c in eachrow(chi)]...)
        
    elseif size(xs, 1) == 66  
        # alanine dipeptide: scatter plot of chi as color over the Ramachandran plot
        # TODO: dispatch on simulation
        scatter_ramachandran(xs, chi)
    else
        # otherwise: plot chi over data index
        plot()
        
        target && scatter!(isotarget(iso)' |> cpu, label="SK\\chi", markerstrokewidth=0.1, markersize=2)

        scatter!(chi'; ylims=autolims(chi), xlabel="#", label="\\chi", markerstrokewidth=0.1, markersize=2)
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
function scatter_chifix(iso; maxpoints=0)
    chi = chis(iso) |> cpu
    kchi = koopman(iso) |> cpu

    chi, kchi = filter_maxpoints.((chi, kchi), maxpoints)
    
    lim = autolims(chi)

    p = plot()
    for i in 1:size(chi, 1)
        scatter!(chi[i, :], kchi[i,:], markersize=2, xlabel="χ", ylabel="Kχ", xlims=lim, ylims=lim)
    end
    p
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

scatter_ramachandran(iso::Iso; kwargs...) = scatter_ramachandran(coords(iso.data) |> cpu, iso.model(features(iso.data)) |> cpu; kwargs...)

scatter_ramachandran(x, model; kwargs...) = scatter_ramachandran(x, vec(model(x)); kwargs...)
scatter_ramachandran(x, mat::AbstractMatrix; kwargs...) = plot(map(enumerate(eachrow(mat))) do (i, row)
    scatter_ramachandran(x, vec(row); title="$i", kwargs...)
end...)

function scatter_ramachandran(x::AbstractMatrix, z::Union{AbstractVector,Nothing}=nothing; kwargs...)
    ph = phi(cpu(x))
    ps = psi(cpu(x))
    z = cpu(z)
    scatter(ph, ps, marker_z=z, xlims=[-pi, pi], ylims=[-pi, pi],
        markersize=3, markerstrokewidth=0, markercolor=:viridis, legend=false,
        xlabel="\\phi", ylabel="\\psi", title="Ramachandran", aspect_ratio=1; kwargs...
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
                display("image/png", p)
            catch e
                e isa InterruptException && rethrow(e)
                @warn "could not print ($e)"
            end
        end, secs)
end

function plot_reactioncoords(iso)
    xs = coords(iso.data)
    dim = size(xs, 1)
    if dim == 66  # alanine dipeptide
        chi = chis(iso)
        scatter_ramachandran(xs, chi)
    elseif dim == 219
        inds = (41, 49)
        inds = (20, 62)
        i = findfirst(CartesianIndex(inds...) .== halfinds(73))
        scatter(iso.data.data[1][i, :] |> vec, chis(iso) |> vec)
    end
end

getpdb(id::String) = Base.download("https://files.rcsb.org/download/$id.pdb")

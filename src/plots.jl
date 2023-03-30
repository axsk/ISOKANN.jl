using Plots

export plot_learning,
    scatter_ramachandran

"""
scatter plot of all first "O" atoms of the starting points `xs`
as well as the "O" atoms from the koopman samples to the first point from `ys`
"""
function plotatoms!(xs, ys, model=nothing)
    dim, nx, nk = size(ys)
    i = 3*(9-1) # first O atom
    cx = 1
    cy = 1

    if !isnothing(model)
        cx = model(xs) |> vec
        cy = model(ys[:,1,:]) |> vec
    end

    a = reshape(xs[:,1], 3, :)
    #scatter!(a[1,:], a[2,:], label="atoms of first x0") # all atoms of first x
    scatter!(xs[i+1,:], xs[i+2,:], label = "x0", marker_z=cx, zlims=(0,1))
    scatter!(ys[i+1,1,:], ys[i+2,1,:], label = "ys",
        marker_z = cy, markershape=:xcross)
    return plot!()
end

""" combined plot of loss graph and current atoms """
function plotlossdata(losses, data, model=nothing)
    p1 = plot(losses, yaxis=:log)
    p2 = plot(); plotatoms!(data..., model)
    plot(p1, p2)
end



## Plotting

function plot_learning(iso; subdata = nothing)
    (;losses, data, model) = iso

    !isnothing(subdata) && (data = subdata)

    p1 = plot(losses[1:end], yaxis=:log, title="loss", label="trainloss")

    let td = filter(iso.loggers) do l isa(l, TrainlossLogger) end
        if length(td) > 0 && length(td[1].losses) > 1
            plot!(range(1, length(losses), length(td[1].losses)), td[1].losses, label = "testloss")
        end
    end

    p2 = scatter_ramachandran(reshape(data[2],66,:), model)

    p3 = scatter_chifix(data, model)
    #annotate!(0,0, repr(iso)[1:10])

    ps = [p1,p2,p3]
    plot(ps..., layout=(length(ps),1), size=(600,300*length(ps)))
end


""" fixed point plot, i.e. x vs model(x) """
function scatter_chifix(data, model)
    xs, ys = data
    target = koopman(model, ys)
    xs = model(xs)|>vec
    scatter(xs, target, markersize=2, xlabel = raw"\chi", ylabel=raw"K\chi")
    plot!([minimum(xs), maximum(xs)], [minimum(target),maximum(target)], legend=false)
end


function inspecttrajectory(sys)
    @time sol = solve(SDEProblem(sys))
    x = reduce(hcat, sol.u)
    scatter_ramachandran(x) |> display
    exportdata(sys, x, path="out/inspect.pdb")
    return x
end

# good colors
# berlin, delta, roma, tofino, tokyo
function scatter_ramachandran(x::AbstractMatrix, model=nothing;kwargs...)
    z = nothing
    !isnothing(model) && (z = model(x) |> vec)
    ph = phi(x)
    ps = psi(x)
    scatter(ph, ps, marker_z=z, xlims = [-pi, pi], ylims=[-pi, pi],
        markersize=3, markerstrokewidth=0, markeralpha=1, markercolor=:tofino,
        xlabel="\\phi", ylabel="\\psi", title="Ramachandran", ;kwargs...
    )
end

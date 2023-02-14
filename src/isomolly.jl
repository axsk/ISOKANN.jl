using StatsBase: mean
#import Optimisers: setup, AbstractRule, Adam
import Optimisers
import Molly: System


""" basic isokann routine for fixed data """
function isokann(data; model, poweriter, learniter, opt, losses=Float64[])
    # if opt is only a rule, initialize the optimizer
    isa(opt, Optimisers.AbstractRule) && (opt = Optimisers.setup(opt, model))

    local grad, target
    for i in 1:poweriter
        xs, ys = data
        cs = model(ys) :: Array{<:Number, 3}
        ks = vec(mean(cs[1,:,:], dims=2)) :: Vector

        target = ((ks .- minimum(ks)) ./ (maximum(ks) - minimum(ks))) :: Vector

        for _ in 1:learniter
            l, grad = let xs=xs  # `let` allows xs to not be boxed
                Zygote.withgradient(model) do model
                    sum(abs2, (model(xs)|>vec) .- target) / length(target)
                end
            end
            push!(losses, l)
            Optimisers.update!(opt, model, grad[1])
        end
    end
    return (;model, opt, losses, grad, target)
end

"""
generate data for koopman evaluation by propagting the dynamics
starting at the given `x0` (or sampling `nx` random starting points)
returns `xs` the starting points and `ys` the corresponding endpoints
"""
function generatedata(dynamics, nkoop::Integer;
        dt::Float64 = 0.01,
        alg = SROCK2(),
        nx::Integer = 10,
        x0::Matrix = randx0(dynamics, nx))

    dim, nx = size(x0)
    ys = zeros(dim, nx, nkoop)
    sde = SDEProblem(dynamics, dt = dt, alg=alg)

    @floop for i in 1:nx, j in 1:nkoop
        ys[:, i, j] = solve(sde; u0=x0[:,i])[end]
    end

    return center(x0), center(ys)
end

"""
center any given states by shifting their individual 3d mean to the origin
"""
function center(xs)
    mapslices(xs, dims=1) do x
        coords = reshape(x, 3, :)
        coords .-= mean(coords, dims=2)
        vec(coords)
    end
end

"""
sample `nx` initial starting points by propagating from the systems coordinate
"""
function bootstrapx0(sys::System, nx; dt, alg=SROCK2())
    x0 = reshape(getcoords(sys), :, 1)
    _, xs = generatedata(sys, nx; x0=x0, dt, alg)
    reshape(xs, :, nx)
end

""" empirical shift-scale operation """
shiftscale(ks) = (ks .- minimum(ks)) ./ (maximum(ks) - minimum(ks))

""" given an array of states, return a chi stratified subsample """
function stratified_x0(model, ys, n)
    ys = reshape(ys, size(ys,1), :)
    ks = shiftscale(model(ys) |> vec)

    i = subsample_uniformgrid(ks, n)
    xs = ys[:, i]
    return xs
end

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

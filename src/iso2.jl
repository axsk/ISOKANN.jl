import StatsBase
using ISOKANN
using LinearAlgebra: pinv, eigen, norm, diag, I
using Plots
include("isosimple.jl")
include("forced/langevin.jl")

function isokann2(model, opt, dataset, n, subsample=100)
    losses = Float64[]
    global target
    for i in 1:n
        xs, ys = getdata!(dataset, model)
        nx = size(xs, 2)
        batch = nx > subsample ? rand(1:size(xs, 2), subsample) : 1:nx
        xs = xs[:, batch]
        ys = ys[:, batch, :]

        cxs = model(xs)

        cs = model(ys)::AbstractArray{<:Number,3}
        ks = StatsBase.mean(cs[:, :, :], dims=3)[:,:,1]

        target = Kinv(model(xs), ks)

        #target = K_isa(ks)
        #target = target .- mean(target, dims=1) .+ .1

        target = target ./ norm.(eachrow(target), 1) .* size(target, 2)
        target = real.(target)

        crit = cxs * target'
        display(crit)
        P  = stableperm(crit)
        display(cxs * (P * target)')
        target = P * target


        debug = true
        if debug && rand() > 0.9
            if size(xs, 1) == 2
                vis2d(xs, target)
            elseif size(xs, 1) ==
                   scatter(vec(xs), target')
            plot!(vec(xs), cxs') |> display
            end
            #scatter(cxs', target')|> display
        end


        for j in 1:20
            loss   = learnstep!(model, xs, target, opt)  # Neural Network update
            push!(losses, loss)
        end
    end
    losses
end

function vis2d(xs, fxs)
    plot()
    for f in eachrow(fxs)
        scatter!(eachrow(xs)..., marker_z=f)
    end
    plot!()
end

# there are more stable versions using QR or SVD for the application of the pseudoinv
function Kinv(chi::Matrix, kchi::Matrix, direct=true)
    if direct
        Kinv = chi * pinv(kchi)
        e = eigen(Kinv)
        display(e)
        #@show e
        return inv(e.vectors) * Kinv * kchi
    else
        K = kchi*pinv(chi)
        e = eigen(K)
        # display(e)
        return inv(e.vectors) * inv(K) * kchi
    end
end


using Combinatorics

function stableA(A)
    n = size(A,1)
    p = argmax(permutations(1:n, n)) do p
        sum(diag(A[p, :]))
    end
    A[p,:]
end

# adjust only for the sign
function stablesign(A)
    n = size(A, 1)
    P = collect(Int, I(n))
    for i in 1:n
        P[i, i] *= sign(A[i, i])
    end
    return P
end




function stableperm(A)
    n = size(A,1)
    p = argmax(permutations(1:n, n)) do p
        sum(abs.(diag(A[p, :])))
    end
    P = collect(Int, I(n)[p,:])
    for i in 1:n
        P[p[i], i] *= sign(A[i,p[i]])
    end
    P
end

function K_isa(ks)
    A = innersimplexalgorithm(ks')'
    #A = stableA(A)
    A * ks
end

innersimplexalgorithm(X) = inv(X[indexmap(X), :])

function indexmap(X)
    @assert size(X,1) > size(X,2)
    # get indices of rows of X to span the largest simplex
    rnorm(x) = sqrt.(sum(abs2.(x), dims=2)) |> vec
    ind = zeros(Int, size(X, 2))
    for j in 1:length(ind)
        rownorm = rnorm(X)
        # store largest row index
        ind[j] = argmax(rownorm)
        if j == 1
            # translate to origin
            X = X .- X[ind[1], :]'
        else
            # remove subspace
            X = X / rownorm[ind[j]]
            vt = X[ind[j], :]'
            X = X - X * vt' * vt
        end
    end
    return ind
end

function doublewelldata(nx, ny)
    d = Doublewell()
    xs = randx0(d, nx)
    ys = propagate(d, xs, ny)
    return xs, ys
end

function test_dw(; kwargs...)
    iso2(nd=2, sys=Doublewell(); kwargs...)
end

function test_tw(; kwargs...)
    iso2(nd=3, sys=Triplewell(); kwargs...)
end

import Flux

function iso2(; n=1, nx=10, ny=10, nd=2, sys=Doublewell(), lr=1e-3, decay=1e-5)
    global s, model, xs, ys, opt, loss
    s = sys
    xs = randx0(sys, nx)
    if dim(sys) == 1
        xs = sort(xs, dims=2)
    end
    ys = propagate(sys, xs, ny)

    model = Flux.Chain(
        Flux.Dense(dim(sys),5, Flux.sigmoid),
        Flux.Dense(5,5,Flux.sigmoid),
        #Flux.Dense(30,10,Flux.sigmoid),
        Flux.Dense(5,nd))

    opt = Flux.setup(Flux.AdamW(lr, (0.9, 0.999), decay), model)
    loss = isokann2(model, opt, (xs,ys), n)
end

function vismodel(model, dim)
    if dim == 1
        plot(model(collect(-1.5:.1:1.5)')')
    elseif dim == 2
        plot()
        grd = -2:.1:2
        for i in 1paks = ["IJulia", "Plots", "LaTeXStrings", "Cubature", "StatsBase", "Molly", "StaticArrays", "Unitful", "Bio3DView", "KernelDensity", "Measurements", "Zygote"]
            Pkg.add(paks):3
            m = [model([x,y])[i] for x in grd, y in grd]
            m = m .- StatsBase.mean(m)
            m = m ./ StatsBase.std(m)
            contour!(grd, grd, m, color=i) |> display
        end

    end
end
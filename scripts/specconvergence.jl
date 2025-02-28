using LinearAlgebra
using SqraCore
using ISOKANN
M=3

xs = -1.5:0.1:1
ys = -0.5:0.1:1
pot = [ISOKANN.mueller_brown(x, y) for x in ys, y in ys]

Q = sqra_grid(pot, beta=0.02)
K = exp(collect(Q) .* 0.1)
@show eigen(K, sortby=x -> real(-x)).values[1:10]

eigenvecs = eigen(K, sortby=x->real(-x)).vectors

ev = eigenvecs[:, 1:M] |> real

#target = ISOKANN.TransformSVD()
target = ISOKANN.TransformLeftRight()

transforms = [
    ISOKANN.TransformLeftRight(),
    ISOKANN.TransformSVD(),
    ISOKANN.TransformSVDRev(),
    ISOKANN.TransformISA(),
    
    #ISOKANN.TransformGramSchmidt1(),
    ISOKANN.TransformGramSchmidt2(),
    ISOKANN.TransformPseudoInv(),
    ISOKANN.TransformPseudoInv(direct=false),
    ISOKANN.TransformPseudoInv(eigenvecs=false),
    ISOKANN.TransformPseudoInv(direct=false, eigenvecs=false),
    ]

mutable struct VecModel
    u
    K
end

noisekoop = 0.01
noisestorage = 0.01

(model::VecModel)(x) = (model.u)'
ISOKANN.expectation(model::VecModel, ys) = model.K * model.u + randn(size(model.u)) .* noisekoop |> permutedims 

function randmodel() 
    x0 = rand(256, M)
    m = VecModel(x0, K)
end

function getlosses(;m=randmodel(), ev=ev, target=target, iters=100)
    @show target
    losses = Float64[]
    push!(losses, norm(m.u - ev ./ sum(ev, dims=1) .* sum(m.u, dims=1)))
    for i in 1:iters
        m.u = ISOKANN.isotarget(m, rand(1,1), nothing, target)' + randn(size(model.u)) .* noisestorage
        #normalize!.(eachcol(m.u))
        push!(losses, lossmetric(ev, m.u))
    end
    return losses
end

# projection frobenius norm
lossmetric(A, B) = norm(A * inv(A' * A) * A' - B * inv(B' * B) * B')

#heatmap(reshape(m.u[:,2], 16, 16))

function compare(iters=10)
    plot()
    for i in 1:length(transforms)
        target = transforms[i]
        try
            ls = map(1:iters) do x
                try
                    getlosses(;target)
                catch
                    nothing
                end
            end
            ls = filter(!isnothing, ls)
            n =length(ls)
            ls = hcat(ls...)
            @show median(ls, dims=2)
            mn = mean(ls, dims=2)
            sig = maximum(ls, dims=2)
           # @show sig
            plot!(ls[:,1], label=string(target) * " ($n)", color=i)
            plot!(ls, label=nothing, color=i)
           
        catch 
        end
    end
    plot!(yaxis=:log, title="noisekoop $noisekoop noisestorage $noisestorage")
end


using LinearAlgebra
using SqraCore
using ISOKANN
using Plots
using StatsBase
M=3

__revise_mode__ = :evalassign

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
    ISOKANN.TransformGramSchmidt2(), # good
    ISOKANN.TransformISA(), # good

    () -> ISOKANN.TransformPinv2(ones(256, 4), ones(256,4), true),
    () -> ISOKANN.TransformPinv2(ones(256, 4), ones(256, 4), false),
    () -> ISOKANN.TransformPinv2(ones(256, 12), ones(256, 12), true),
    () -> ISOKANN.TransformPinv2(ones(256, 12), ones(256, 12), false),

    #ISOKANN.TransformLeftRight(),

    ISOKANN.TransformLeftRightHistory(M+1),
    ()->ISOKANN.TransformPinv3(ones(3, 256), ones(3, 256), false),
    () -> ISOKANN.TransformPinv3(ones(9, 256), ones(9, 256), false),
   # ISOKANN.TransformPinv3(ones(5, 256), ones(5, 256), true),
    ()->ISOKANN.TransformPinv3(ones(4, 256), ones(4, 256), true),
    ()-> ISOKANN.TransformPinv3(ones(12, 256), ones(12, 256), true),
    #ISOKANN.TransformLeftRightHistory(20),
    #ISOKANN.TransformPseudoInv(),
    #ISOKANN.TransformPinv(256,4),
    #ISOKANN.TransformPinv(256, 20),

    #ISOKANN.TransformSVD(),
    #ISOKANN.TransformSVDRev(),
    #ISOKANN.TransformGramSchmidt1(),
    
    #ISOKANN.TransformPseudoInv(direct=false),
    #ISOKANN.TransformPseudoInv(normalize=false),
    #ISOKANN.TransformPseudoInv(permute=false),
    #ISOKANN.TransformPseudoInv(eigenvecs=false),
    
    ]

mutable struct VecModel
    u
    K
end

noisekoop = 1e-4
noisestorage = 1e-2

(model::VecModel)(x) = (model.u)'
ISOKANN.expectation(model::VecModel, ys) = model.K * model.u + randn(size(model.u)) .* noisekoop |> permutedims 

function randmodel() 
    x0 = rand(256, M)
    m = VecModel(x0, K)
end

function getlosses(;m=randmodel(), ev=ev, target=target, iters=100)
    losses = Float64[]
    push!(losses, lossmetric(ev, m.u))
    for i in 1:iters
        m.u = (ISOKANN.isotarget(m, rand(1,1), nothing, target)')
        normalize!.(eachcol(m.u))
        m.u += randn(size(m.u)) .* noisestorage
        push!(losses, lossmetric(ev, m.u))
    end
    return losses
end

# projection frobenius norm
function lossmetric(A, B) 
    A = reduce(hcat, normalize.(eachcol(A)))
    B = reduce(hcat, normalize.(eachcol(B)))
    try
        norm(A * inv(A' * A) * A' - B * inv(B' * B) * B')
    catch
 #       Main.@infiltrate
    end
end

#heatmap(reshape(m.u[:,2], 16, 16))

function compare(;iters=10, nk=1e-4, ns = 1e-4)
    global noisekoop = nk
    global noisestorage = ns
    plot(palette = :tab10)
    for i in 1:length(transforms)
        target = transforms[i]
        label = string(target)

        try
            ls = map(1:iters) do x
                if target isa ISOKANN.TransformLeftRightHistory5
                    hist = size(target.L, 2)
                    target = ISOKANN.TransformLeftRightHistory(hist)
                    label = "LeftRightHistory($(hist))"
                end
                let target = target isa Function ? target() : target
                    @show typeof(target)
                    try
                        getlosses(;target)
                    catch e
                        #println("$target")
                        e isa InterruptException && rethrow(e)
                        @show e
                        
                    nothing
                    #rethrow(e)
                    end
                end
            end
            ls = filter(!isnothing, ls)
            n =length(ls)
            ls = hcat(ls...)
            #@show median(ls, dims=2)
            mn = mean(ls, dims=2)
            md = median(ls, dims=2)
            mx = maximum(ls, dims=2)
            sd = std(ls, dims=2)
            q1 = quantile.(eachrow(ls), 0.1)
            q2 = quantile.(eachrow(ls), 0.9)
            ls = mn
           # @show sig
            #label = label * " ($n)"
            
            #plot!(ls[:,1]; label, color=i)
            let series = mn
            plot!(series, 
                #ribbon=(series .- q1, q2 .- series)
                ; label, color=i)
            end
            #size(ls,2)>1 && plot!(ls, label=nothing, color=i)
           
        catch  e
            rethrow(e)
        end
    end
    plot!(yaxis=:log)
    plot!(yaxis=:log, title="k $noisekoop u $noisestorage")
    plot!()
end
###

#=
begin
    v = eigen(K).vectors[:, end:-1:end-2]' |> real
    L = copy(v)
    L[3,:] += R[2,:]

    R = rand(3,256)

    L = (K*R')'
    kinv = L * pinv(R)
    Q = eigen(kinv).vectors[:, end:-1:1] |> real
    R = rownormalize(inv(Q) * L)

    L = R
    R = (K*L')'
    kinv = L * pinv(R)
    Q = eigen(kinv).vectors[:, 1:end] |> real
    R = rownormalize(inv(Q) * R)
end
=#

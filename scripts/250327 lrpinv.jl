using ISOKANN
using SqraCore
using LinearAlgebra
using Plots
using StatsBase

colnormalize(x; p=2) = (x ./ norm.(eachcol(x), p)')
signfirstrow(x) = x ./ sign.(x[1, :])'
normcols(x) = signfirstrow(colnormalize(x))

""" generate_K_ev(M=4)

Generate Koopman matrix `K`` and the first `M` eigenvectors through SqRA """
function generate_K_ev(M=4)
    xs = -1.5:0.05:1
    ys = -0.5:0.05:1
    pot = [ISOKANN.mueller_brown(x, y) for x in ys, y in ys]

    Plots.heatmap(pot) |> display

    Q = sqra_grid(pot, beta=0.02)
    K = exp(collect(Q) .* 0.1)
    vals = eigen(K, sortby=x -> real(-x)).values[1:10]
    println("eigenvectors")
    println.(vals)

    eigenvecs = eigen(K, sortby=x -> real(-x)).vectors

    ev = eigenvecs[:, 1:M] |> real |> normcols
    heatmap(ev) |> display
    return K, ev
end
K, ev = generate_K_ev();





DEBUG = false

""" pi1(L, R)

Pseudoinverse estimation of the eigenfunctions of K^-1 """
function pi1(L, R)
    x = L'
    y = R'

    @assert size(x, 1) < size(x, 2)
    kinv = x * pinv(y)
    e = eigen(kinv)
    DEBUG && @show 1 ./ e.values
    Q = e.vectors
    target_pi1 = (inv(Q) * y)'
    target_pi1 = normcols(target_pi1)
end

""" pi2(L, R)

Pseudoinverse estimation of the eigenfunctions of K
Note: this is considerable worse. I as assume this is as we work on the old subspace span{L} """
function pi2(L, R)
    x = L'
    y = R'

    @assert size(x, 1) < size(x, 2)
    k = y * pinv(x)
    e = eigen(k)
    DEBUG && @show e.values[end:-1:1]
    Q = e.vectors[:, end:-1:1] |> real

    target_pi2 = (Q * y)'
    target_pi2 = normcols(target_pi2)
end

#=
# these are all the same
pinv(L) * R
L \ R
R' / L'
R' * pinv(L');
=#

""" lr1(L, R)

Eigenfunction estimation on the space span{L,R} of K^-1"""
function lr1(L, R)
    @assert size(L, 1) > 100
    M = size(L, 2)

    # compute qr basis of LR space -- q is basis, r are components
    LR = hcat(R, L)
    q, r = qr(LR)
    @assert q[:, 1:2*M] * r ≈ LR

    D = size(L, 2)
    # representation of image and preimage in that basis
    qR = r[:, 1:D]
    qL = r[:, D+1:end]

    # estimate map in this space
    A = qL * pinv(qR)
    e = eigen(A)
    #@show e.values
    Q = e.vectors

    # project eigenvectors back to original space
    t = q * Q
    return normcols(t[:, M+1:end])
end

""" lr1(L, R)

Eigenfunction estimation on the space span{L,R} of K"""
function lr2(L, R)
    @assert size(L, 1) > 100
    M = size(L, 2)

    # compute qr basis of LR space -- q is basis, r are components
    LR = hcat(R, L)
    q, r = qr(LR)
    @assert q[:, 1:2*M] * r ≈ LR

    D = size(L, 2)
    # representation of image and preimage in that basis
    qR = r[:, 1:D]
    qL = r[:, D+1:end]

    # estimate map in this space
    A = qR * pinv(qL)
    e = eigen(A)
    #@show e.values
    Q = e.vectors

    # project eigenvectors back to original space
    t = q * Q[:, end:-1:M+1]
    return normcols(t)
end


function benchmark(; eta=0.001, n=5, m=1000)
    es = mapreduce(vcat, 1:m) do x
        L, R = generate_LR(; eta, n)
        target_lr1 = lr1(L, R)
        target_lr2 = lr2(L, R)
        target_pi1 = pi1(L, R)
        target_pi2 = pi2(L, R)

        e = mapreduce(hcat, [target_lr1, target_lr2, target_pi1, target_pi2, R]) do x
            x = normcols(x)
            norm.(eachcol(x .- ev))
        end
        sum(e, dims=1)
    end

    [mean(es, dims=1)
        median(es, dims=1)
        std(es, dims=1)]
end


function generate_LR_hist(; eta=0.001, K=K, n=10, orthogonal=false)
    L = randn(size(ev)) |> normcols
    Ls = similar(L, size(L)..., n)
    Rs = similar(L, size(L)..., n)

    for i in 1:n
        if orthogonal
            q, r = qr(L)
            L = q[:, 1:size(L, 2)]
        end

        L = normcols(L)

        R = K * L

        Ls[:, :, i] = L
        Rs[:, :, i] = R

        L = R + randn(size(R)) * eta
    end
    return Ls, Rs
end

function generate_LR(;kwargs...)
    Ls, Rs = generate_LR_hist(;kwargs...)
    Ls[:,:,end], Rs[:,:,end]
end

eta = 0.01
n = 10
m = 100
orthogonal = false

function benchmark(; eta=eta, n=n, m=m, orthogonal=orthogonal, hist=hist)
    es = mapreduce(vcat, 1:m) do x
        Ls, Rs = generate_LR_hist(; eta, n, orthogonal)
        if hist
            L = mergehist(Ls)
            R = mergehist(Rs)
        else
            L = Ls[:, :, end]
            R = Rs[:, :, end]
        end
        target_lr1 = lr1(L, R)
        target_lr2 = lr2(L, R)
        target_pi1 = pi1(L, R)
        target_pi2 = pi2(L, R)

        e = mapreduce(everror, hcat, [target_lr1, target_lr2, target_pi1, target_pi2, R])

        sum(log.(e), dims=1)
    end
end

function test_hist()

    Ls, Rs = generate_LR_hist()


    hist = false

    for hist in [true, false]
        let es = benchmark_hist(; hist)
            [mean(es, dims=1)
             median(es, dims=1)
             std(es, dims=1)] |> display
        end
    end
end

mergehist(Ls) = reshape(Ls[:, :, end:-1:1], size(Ls, 1), :)
withhistory(method, Ls, Rs, n) = method(mergehist(Ls[:,:,n]), mergehist(Rs[:,:,n]))
everror(x) = norm.(eachcol(normcols(x[:, 1:size(ev,2)]) .- ev))

function test_all()
    test1()
    test_hist()
    test_lrhist()
end


function compare_hist()
    Ls, Rs = generate_LR_hist(; eta, n, orthogonal)

    plot()
    for method in [pi1, pi2, lr1, lr2]
        ls = map(n:-1:1) do i 
            p = withhistory(method, Ls, Rs, i:n)
            sum(log.(everror(p)))
        end
        plot!(ls, label=method)
    end
    plot!()
end
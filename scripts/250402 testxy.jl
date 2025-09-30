using LinearAlgebra
using SqraCore
using ISOKANN
using Plots
using ArnoldiMethod
using StatsBase
rownormalize(x,p=2) = mapreduce(x->normalize(x,p)', vcat, eachrow(x))
heat(x) = heatmap(log10.(abs.(x)), yflip=true)

heat2d(x) = heatmap(reshape(real(x),26,16), yflip=true)

xs = -1.5:0.1:1
ys = -0.5:0.1:1
pot = [ISOKANN.mueller_brown(x, y) for x in xs, y in ys]

Q = sqra_grid(pot, beta=0.02)
K = exp(collect(Q) .* 0.1)
aK, vK = eigen(K, sortby=x->-real(x))

nd = 2
nx = size(K,1)

eta = 0.0001

x=rand(nd,nx) |> rownormalize
Ls = zeros(nd, nx, 0)
Rs = zeros(nd, nx, 0)
for i in 1:4
    global x, Ls, Rs
    Ls = cat(Ls, x, dims=3)
    y = x * K' + randn(size(x)) * eta
    Rs = cat(Rs, y, dims=3)
    x = y |> rownormalize
end

L = vcat(eachslice(reverse(Ls, dims=3), dims=3)...)
R = vcat(eachslice(reverse(Rs, dims=3), dims=3)...)
    
kinv = L * pinv(R)
a,v=eigen(kinv, sortby=mysort)
1 ./ a
aK
t = inv(v) * R |> rownormalize

heat(t * vK[:, end-20:end])

heat(abs.(hcat(vec(1 ./ a) .- vec(reverse(aK)[1:30])')))
heat(1 ./ a .*t)
heat(1 ./ a .* v)

function domspace(x)
    schur = partialschur(x, which = :SR) |> first
    @show schur.eigenvalues
    schur.Q
end

function pschur(x; kwargs...)
    s = partialschur(x;nev=size(x,1),kwargs...) |> first
    s.eigenvalues, s.Q, s.R
end

# return true if a should come before b
mysort(a) = mysort(real(a), imag(a))
function mysort(c::Complex)
    a = real(c)
    a < 0.9 && return Inf
    a
end

function vcatinterleave(a,b)
    reshape(vcat(a, b), size(a,2), :)'
end


q,r = qr(R')


q = q[:, 1:size(R,1)]
l = q'*L'
r = q'*R'

k = r/l 
a,v = eigen(k, sortby=x->-real(x))

t2=q * v
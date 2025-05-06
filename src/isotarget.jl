### ISOKANN target transformations

""" expectation(f, xs)

Computes the expectation value of `f` over `xs`.
"""
expectation(f, xs) = dropdims(sum(f(xs); dims=2); dims=2) ./ size(xs, 2)





"""
    TransformPseudoInv(normalize, direct, eigenvecs, permute)

Compute the target by approximately inverting the action of K with the Moore-Penrose pseudoinverse.

If `direct==true` solve `chi * pinv(K(chi))`, otherwise `inv(K(chi) * pinv(chi)))`.
`eigenvecs` specifies whether to use the eigenvectors of the schur matrix.
`normalize` specifies whether to renormalize the resulting target vectors.
`permute` specifies whether to permute the target for stability.
"""
@kwdef struct TransformPseudoInv
    normalize::Bool = true
    direct::Bool = true
    eigenvecs::Bool = true
    permute::Bool = true
end

function isotarget(model, xs::S, ys, t::TransformPseudoInv) where {S}
    (; normalize, direct, eigenvecs, permute) = t
    chi = model(xs) |> cpu
    @assert size(chi, 1) > 1 "TransformPseudoInv does not work with one dimensional chi functions"

    kchi = expectation(model, ys) |> cpu

    kchi_inv = try
        pinv(kchi)
    catch
        throw(DomainError("Could not compute the pseudoinverse. The subspace might be singular/collapsed"))
    end

    if direct
        Kinv = chi * kchi_inv
        T = eigenvecs ? schur(Kinv).vectors : I
        target = T * Kinv * kchi
    else # this does not work at all!
        K = kchi * kchi_inv
        T = eigenvecs ? schur(K).vectors : I
        target = T * inv(K) * kchi
    end

    normalize && (target = target ./ norm.(eachrow(target), 1) .* size(target, 2))
    permute && (target = fixperm(target, chi))

    return S(target)
end



""" TransformISA(permute)

Compute the target via the inner simplex algorithm (without feasiblization routine).
`permute` specifies whether to apply the stabilizing permutation """
@kwdef struct TransformISA
    permute::Bool = true
end

# we cannot use the PCCAPAlus inner simplex algorithm because it uses feasiblize!,
# which in turn assumes that the first column is equal to one.
function myisa(X)
    try
        inv(X[PCCAPlus.indexmap(X), :])
    catch e
        throw(DomainError("Could not compute the simplex transformation. The subspace might be singular/collapsed"))
    end
end

function isotarget(model, xs::T, ys, t::TransformISA) where {T}
    chi = model(xs) |> cpu
    @assert size(chi, 1) > 1 "TransformISA does not work with one dimensional chi functions"
    ks = expectation(model, ys) |> cpu
    target = myisa(ks')' * ks
    t.permute && (target = fixperm(target, chi))
    return T(target)
end

""" TransformShiftscale()

Classical 1D shift-scale (ISOKANN 1) """
struct TransformShiftscale end

function isotarget(model, xs, ys, t::TransformShiftscale; shiftscale=false)
    ks = expectation(model, ys)
    @assert size(ks, 1) == 1 "TransformShiftscale only works with one dimensional chi functions"

    min, max = extrema(ks)
    max > min || throw(DomainError("Could not compute the shift-scale. chi function is constant"))
    target = (ks .- min) ./ (max - min)
    # TODO: find another solution for this, this is not typestable
    if (shiftscale)
        shift = min / (min + 1 - max)
        lambda = max-min
        return target, shift, lambda
    end
    return target
end


# TODO: design decision: do we want this as outer type or as part of what we have inside the other transforms?
""" TransformStabilize(transform, last=nothing)

Wraps another transform and permutes its target to match the previous target

Currently we also have the stablilization (wrt to the model though) inside each Transform. TODO: Decide which to keep
"""
@kwdef mutable struct Stabilize2
    transform
    last = nothing
end

function isotarget(model, xs, ys, t::Stabilize2)
    target = isotarget(model, xs, ys, t.transform)
    isnothing(t.last) && (t.last = target)
    if t.transform isa TransformShiftscale  # TODO: is this even necessary?
        if (sum(abs, target - t.last)) > length(target) / 2
            println("flipping")
            target .= 1 .- target
        end
        t.last = target
        return target
    else
        return fixperm(target, t.last)
    end
end


### Permutation - stabilization
import Combinatorics

"""
    fixperm(new, old)

Permutes the rows of `new` such as to minimize L1 distance to `old`.

# Arguments
- `new`: The data to match to the reference data.
- `old`: The reference data.
"""
function fixperm(new, old)
    # TODO: use the hungarian algorithm for larger systems
    n = size(new, 1)
    p = argmin(Combinatorics.permutations(1:n)) do p
        norm(new[p, :] - old, 1)
    end
    new[p, :]
end

using Random: shuffle
function test_fixperm(n=3)
    old = rand(n, n)
    @show old
    new = old[shuffle(1:n), :]
    new = fixperm(new, old)
    @show new
    norm(new - old) < 1e-9
end

struct TransformGramSchmidt1 end

function isotarget(model, xs::T, ys, t::TransformGramSchmidt1) where {T}
    chi = model(xs)
    dim = size(chi, 1)

    if dim == 1
        chi .-= sum(chi) ./ length(chi)
    end

    for i in 1:dim
        for j in 1:i-1
            chi[i, :] .-= (chi[i, :], chi[j, :]') .* chi[j, :]
        end
        chi[i, :] ./= norm(chi[i, :])
    end
    return chi
end

"""
    TransformGramSchmidt()

Compute the target through a Gram-Schmidt orthonormalisation.
"""
@kwdef struct TransformGramSchmidt2

end

global rs = []

function isotarget(model, xs, ys, t::TransformGramSchmidt2)

    renormalize = false
    firstconst = false

    #chi = model(xs)  #  TODO: we dont use ys anywhere! this cant be right!
    chi = expectation(model, ys)
    c = sqrt(size(chi, 2))

    if firstconst
        z = similar(chi, 1, size(chi, 2))
        z .= 1
        chi = vcat(z, chi)
    end

    renormalize && (chi ./= c)


    q, r = qr(chi')
    q = typeof(q).types[1](q) # convert from compact to full representation
    #q = Matrix(q)  # orthogonal basis
    rand() < 0.05 && display(r)

    push!(rs, diag(r))

    if firstconst
        t = q'[2:end, :] .* diag(sign.(r))[2:end]
    else
        t = q' .* diag(sign.(r))
    end

    renormalize && (t .*= c)
    return t
end


@kwdef struct TransformLeftRight
end

function isotarget(model, xs, ys, t::TransformLeftRight)
    L = model(xs)'
    R = expectation(model, ys)'
    n, d = size(L)

    addones(x) = hcat(fill!(similar(x, size(x, 1)), 1 ./ sqrt(size(x, 1))), x)

    L = addones(L)
    R = addones(R)

    transformleftright(L, R)[:, 1:d]'
end

@kwdef mutable struct TransformLeftRightHistory5
    L::AbstractMatrix
    R::AbstractMatrix
end

@functor TransformLeftRightHistory5

function Base.show(io::IO,  t::TransformLeftRightHistory5)#
    print(io, "TransformLeftRightHistory($(size(t.L)))")
end

TransformLeftRightHistory(hist::Int) = TransformLeftRightHistory5(ones(0,hist), ones(0,hist))

function isotarget(model, xs, ys, t::TransformLeftRightHistory5)
    L = t.L
    R = t.R
    
    l = model(xs)'
    r = expectation(model, ys)'
    n, d = size(l)

    @assert size(L,2) == size(R, 2) >= d + 1

    t.L = updatehistory(t.L, l)
    t.R = updatehistory(t.R, r)

    return transformleftright(t.L, t.R)[:, 2:d+1]' # dont return constant vector
end

function transformleftright(L::T, R) where {T}

    CUDA.@allowscalar @assert all(L[1,1] .== L[:,1] .== R[:,1]) "first columns are not constant"

    # we have A*L = R
    # and want to find the eigenfcts of A.
    # to that end we construct a basis from <L..., R...>
    # and compute the eigenfunctions of the projection of A onto that subspace 

    D = size(L, 2)
    LR = hcat(R, L)
    q, r = qr(LR)
    
    # we can project L and R onto the basis of the "Krylov" space
    #qL = q' * L
    #qR = q' * R
    # which is the same as 
    qR = r[:, 1:D]
    qL = r[:, D+1:end]

    # and look for the map that is mapping qL to qR
    A = qR / qL
    
    #res = A * qL - qR
    #@show norm(res) # the residuum should be zero
    # given A in the Krylov basis, we can now compute its eigenfunctions
    vecs, vals = domsubspaceeigen(A)
    #vals, vecs = LinearAlgebra.eigen(A, sortby=x->-abs(x))
    #vecs = T(vecs)
    #vecs, vals = domsubspaceschur(A)
    @show vals
    vals = vals[1:D]
    vecs = vecs[:, 1:D]
    

    # projecting the dom. eigenvecs. from Krylov basis back to data space we obtain our new target
    #vecs = cu(vecs)
    target = q * vecs

    # lets compare the orientation of previous and resulting vectors, as to flip then for stable training
    s = sum(L .* target, dims=1)
    target .*= sign.(s)
    
    # scale targets with their eigenvalue for stable training
    scale = 1
    scaling = real.(vals') .^ scale
    scaling = L isa CuArray ? cu(scaling) : scaling
    target = target .* scaling
    
    target .*= sqrt(size(target, 1))
    if !all(isfinite.(target)) || any(real.(vals) .< 1e-8)
        #Main.@infiltrate()
    end

    return target
end

struct TransformSVD
end

function isotarget(model, xs, ys, t::TransformSVD)
    # similar to DMD
    L = model(xs)'
    R = expectation(model, ys)'

    n, d = size(L)

    U,S,V = svd(L)
    H = U' * R * V * LinearAlgebra.Diagonal(inv.(S))
    vl, vc = LinearAlgebra.eigen(H, sortby=-)
    target = U*vc[:, 1:d]

    return target'
end

struct TransformSVDRev
end

function isotarget(model, xs, ys, t::TransformSVDRev)
    # similar to DMD
    L = model(xs)'
    R = expectation(model, ys)'

    n, d = size(L)

    U, S, V = svd(R)
    H = U' * R * V * LinearAlgebra.Diagonal(inv.(S))
    vl, vc = LinearAlgebra.eigen(H)
    target = U * vc[:, 1:d]

    return target'
end

abstract type TransformPinv end



@kwdef mutable struct TransformPinv1{T<:AbstractMatrix} <: TransformPinv
    L::T
    R::T
end



function Base.show(io::IO, t::TransformPinv)#
    print(io, "$(typeof(t))($(size(t.L,2)))")
end

TransformPinv(n::Int, hist::Int) = TransformPinv1(ones(n, hist), ones(n, hist))

function isotarget(model, xs, ys, t::TransformPinv)
    l = model(xs)'
    r = expectation(model, ys)'
    n, d = size(l)

    t.L = updatehistory(t.L, l)
    t.R = updatehistory(t.R, r)

    #target = transformpinv(t.L[:, 2:end], t.R[:, 2:end], t)'
    target = transformpinv(l,r, t)'

    #target = fixperm(target, l')
    target[1:d, :]
end

using ArnoldiMethod: partialschur

matchcuda(typed, value) = typed isa CUDA.CuArray ? gpu(value) : cpu(value)
function transformpinv(L, R, ::TransformPinv1)
    debug = true
    # we transpose as we work in rowspace
    L=L'
    R=R'

    @assert size(L, 1) < size(L,2)

    # map from R->L in the row basis of R
    kinv = L * pinv(R)

    debug && display(kinv)

    s, = partialschur(cpu(kinv), which=:SR)
    debug && @show s.eigenvalues
    # schur vectors wrt to the row basis of R
    Q = matchcuda(L, s.Q)

    debug && display(Q)
    # S T' R = T kinv R 
    target = Q * kinv * R
    #target = T * R

    

    target = rownormalize(target) .* size(target, 2)
    debug && display(target)
    return target'
end

@kwdef mutable struct TransformPinv2{T<:AbstractMatrix} <: TransformPinv
    L::T
    R::T
    direct::Bool
end

function transformpinv(L, R, t::TransformPinv2)
    # we transpose as we work in rowspace
    x = L'
    y = R'

    @show size(x)
    @show size(y)


    @assert size(x, 1) < size(x, 2)

    if t.direct
        kinv = x * pinv(y)
        @show size(kinv)
        Q = LinearAlgebra.eigen(kinv).vectors |> realsubspace
        inv(Q)
    else
        k = y * pinv(x)
        Q = LinearAlgebra.eigen(k).vectors[:, end:-1:1] |> realsubspace
        inv(Q)
    end
    

    #@show Q
    #display(Q)

    target = rownormalize(inv(Q) * y)
    return target'
end

rownormalize(x;p=2) = (x ./ norm.(eachrow(x), p))

function domsubspaceeigen(A::T) where {T}
    A = cpu(A)
    vals, vecs = LinearAlgebra.eigen(A, sortby=x->-abs(real(x)))
    vecs = T(realsubspace(vecs))
    return vecs, vals
end

function domsubspaceschur(A::T) where {T}
    s, = partialschur(A, which=:SR)
    vecs = T(s.Q)
    return vecs, s.eigenvalues
end


""" computation of the real invariant subspace from complex eigenvectors """
function realsubspace(V)
    V = copy(V)
    i = 1
    while i + 1 <= size(V, 2)
        if V[:, i] ≈ conj(V[:, i+1])
            V[:, i] = real(V[:, i])
            V[:, i+1] = imag(V[:, i+1])
            i += 2
        else
            i += 1
        end
    end
    real(V)
end


"""     
    updatehistory(L, l)

Insert the newest observations `l` of size `(n,d)` into columns 2:d+1 of the history matrix `L` of size `(n,h)`.
Automatically grow the `n` dimension of `L` if `l` is bigger.
"""
function updatehistory(L::T, l) where {T}
    n, d = size(l)
    m, h = size(L)
    # grow up for more data
    if n > m
        Lnew = T(zeros(n, h))
        Lnew[1:m, :] .= L
        L = Lnew
    end

    if n < m
        error("automated shrinking is not supported")
    end

    # we could add some history decay here
    L[:, 1] .= 1 / sqrt(size(L, 1))
    L[:, 2+d:end] = L[:, 2:end-d]
    L[:, 2:d+1] = l

    return L

end


mutable struct TransformPinv3{T<:AbstractArray} 
    L::T
    R::T
    fixedone::Bool
end

function Base.show(io::IO, mime::MIME"text/plain", T::TransformPinv3)
    println(io, "TransformPinv3, size $(size(T.L)), fixedone $(T.fixedone)")
end

function TransformPinv3(d::Int, h::Int, fixedone::Bool)
    @assert h >= d
    fixedone && (d += 1)
    x = ones(d, h)
    y = ones(d, h)
    TransformPinv3(x,y,fixedone)
end
    

function updatehistory!(x, y, t::TransformPinv3)
    d = size(x, 1)
    if t.fixedone
        t.L[d+2:end, :] = t.L[2:end-d, :]
        t.R[d+2:end, :] = t.L[2:end-d, :]
        t.L[2:d+1, :] = x
        t.R[2:d+1, :] = y
    else
        t.L[d+1:end, :] = t.L[1:end-d, :]
        t.R[d+1:end, :] = t.L[1:end-d, :]
        t.L[1:d, :] = x
        t.R[1:d, :] = y
    end
end

isotarget(model, xs, ys, t::TransformPinv3) = isotarget(model(xs), expectation(model,ys), t)

function isotarget(x::AbstractArray, y::AbstractArray, t::TransformPinv3)
    d, n = size(x)
    updatehistory!(x,y,t)
    #@show t.fixedone
    target = target_pseudoinverse(t.L, t.R)
    #return target[1:d,:]
    target = t.fixedone ? target[2:d+1,:] : target[1:d, :]
    
    return target
end

function target_pseudoinverse(x,y)
    @assert size(x, 1) < size(x, 2)
    DEBUG = true
    kinv = x * pinv(y)
    
    e = LinearAlgebra.eigen(cpu(kinv), sortby=mysort)
   
    DEBUG && @show e.values
    #DEBUG && @show 1 ./ e.values
    Q = realsubspace(e.vectors)
    Q = typeof(kinv)(Q)
    target = inv(Q) * y
    #return target
    #i = findfirst(x -> real(x) > (1e-3), e.values) # filter out vanishing subspace
    #i = sortperm(abs.(e.values .- 1))
    #@show i, size(target)
    #target = target[i, :]
    target = target ./ sqrt.(sum(abs2, target, dims=2)) * 50# normalize
    #DEBUG && @show sqrt.(sum(abs2, target, dims=2))
    target = target .* sign.(sum(x .* target, dims=2)) # adjust signs
    return target
end
mysort(c::Complex) = mysort(real(c))
function mysort(a::Real)
    a < 0.9 && return Inf
    a
end
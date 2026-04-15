import Mmap

struct MMapped{S,T,N} <: AbstractArray{T,N} where {S<:AbstractArray{T,N}}
    x::S
    io::IOStream

    MMapped(A::S, io::IOStream) where {T,N,S<:AbstractArray{T,N}} = new{S,T,N}(A, io)
end

function MMapped(A::S, file=tempname()::String) where {S}
    io = open(file, "w+")
    x = Mmap.mmap(io, typeof(A), size(A))
    x .= A
    MMapped(x, io)
end

Base.size(A::MMapped) = size(A.x)
Base.getindex(A::MMapped, i::Int) = getindex(A.x, i)
Base.getindex(A, I::Vararg) = getindex(A.x, I)
Base.IndexStyle(::MMapped{T}) where {T} = IndexStyle(T)

import ISOKANN: lastcat

function lastcat(M::MMapped{S}, y::Union{AbstractArray,MMapped}) where {S}
    sz = collect(size(M))
    sz[end] += size(y)[end]
    @show sz
    c = Mmap.mmap(M.io, S, Tuple(sz))
    @show size(c)
    @show length(M)
    copyto!(c, length(M) + 1, y, 1, length(y))
    return MMapped(c, M.io)
end

import Mmap

struct MMapped{S,T,N} <: AbstractArray{T,N} where {S<:AbstractArray{T,N}}
    x::S
    io::IOStream

    MMapped(A::S, io::IOStream) where {T,N,S<:AbstractArray{T,N}} = new{S,T,N}(A, io)
end

function MMapped(A::S, file="/tmp/mmap.bin"::String) where {S}
    io = open(file, "w+")
    x = Mmap.mmap(io, typeof(A), size(A))
    x .= A
    MMapped(x, io)
end

Base.size(A::MMapped) = size(A.x)
Base.getindex(A::MMapped, i::Int) = getindex(A.x, i)
Base.getindex(A, I::Vararg) = getindex(A.x, I)
Base.IndexStyle(::MMapped{T}) where {T} = IndexStyle(T)

function lastcat(M::MMapped{S}, y) where {S}
    sz = collect(size(M))
    sz[end] += size(y)[end]
    c = Mmap.mmap(M.io, S, Tuple(sz))
    c[length(M.x)+1:end] .= y
    return MMapped(c, M.io)
end

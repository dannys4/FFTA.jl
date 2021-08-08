module FFTA

using Primes, DocStringExtensions, LoopVectorization
import Base: getindex
export fft, bfft

include("callgraph.jl")
include("algos.jl")

function fft(x::AbstractVector{T}) where {T}
    y = similar(x)
    g = CallGraph{T}(length(x))
    fft!(y, x, Val(FFT_FORWARD), g[1].type, g, 1)
    y
end

function fft(x::AbstractVector{T}) where {T <: Real}
    y = similar(x, Complex{T})
    g = CallGraph{Complex{T}}(length(x))
    fft!(y, x, Val(FFT_FORWARD), g[1].type, g, 1)
    y
end

function fft(x::AbstractMatrix{T}) where {T}
    M,N = size(x)
    y1 = similar(x)
    y2 = similar(x)
    g1 = CallGraph{T}(size(x,1))
    g2 = CallGraph{T}(size(x,2))

    for k in 1:N
        @views fft!(y1[:,k],  x[:,k], Val(FFT_FORWARD), g1[1].type, g1, 1)
    end

    for k in 1:M
        @views fft!(y2[k,:], y1[k,:], Val(FFT_FORWARD), g2[1].type, g2, 1)
    end
    y2
end

function bfft(x::AbstractVector{T}) where {T}
    y = similar(x)
    g = CallGraph{T}(length(x))
    fft!(y, x, Val(FFT_BACKWARD), g[1].type, g, 1)
    y
end

function bfft(x::AbstractMatrix{T}) where {T}
    M,N = size(x)
    y1 = similar(x)
    y2 = similar(x)
    g1 = CallGraph{T}(size(x,1))
    g2 = CallGraph{T}(size(x,2))

    for k in 1:N
        @views fft!(y1[:,k],  x[:,k], Val(FFT_BACKWARD), g1[1].type, g1, 1)
    end

    for k in 1:M
        @views fft!(y2[k,:], y1[k,:], Val(FFT_BACKWARD), g2[1].type, g2, 1)
    end
    y2
end

end

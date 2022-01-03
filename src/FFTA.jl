module FFTA

using Primes, DocStringExtensions, Reexport, MuladdMacro, LinearAlgebra
@reexport using AbstractFFTs

import AbstractFFTs: Plan

include("callgraph.jl")
include("algos.jl")
include("plan.jl")

#=
function fft(x::AbstractMatrix{T}) where {T <: Real}
    M,N = size(x)
    y1 = similar(x, Complex{T})
    y2 = similar(x, Complex{T})
    g1 = CallGraph{Complex{T}}(size(x,1))
    g2 = CallGraph{Complex{T}}(size(x,2))

    for k in 1:N
        @views fft!(y1[:,k],  x[:,k], 1, 1, FFT_FORWARD, g1[1].type, g1, 1)
    end

    for k in 1:M
        @views fft!(y2[k,:], y1[k,:], 1, 1, FFT_FORWARD, g2[1].type, g2, 1)
    end
    y2
end

function bfft(x::AbstractVector{T}) where {T <: Real}
    y = similar(x, Complex{T})
    g = CallGraph{Complex{T}}(length(x))
    fft!(y, x, 1, 1, FFT_BACKWARD, g[1].type, g, 1)
    y
end

function bfft(x::AbstractMatrix{T}) where {T <: Real}
    M,N = size(x)
    y1 = similar(x, Complex{T})
    y2 = similar(x, Complex{T})
    g1 = CallGraph{Complex{T}}(size(x,1))
    g2 = CallGraph{Complex{T}}(size(x,2))

    for k in 1:N
        @views fft!(y1[:,k],  x[:,k], 1, 1, FFT_BACKWARD, g1[1].type, g1, 1)
    end

    for k in 1:M
        @views fft!(y2[k,:], y1[k,:], 1, 1, FFT_BACKWARD, g2[1].type, g2, 1)
    end
    y2
end =#



end

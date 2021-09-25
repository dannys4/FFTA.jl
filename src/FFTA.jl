module FFTA

using Primes, DocStringExtensions, Reexport, MuladdMacro, ComputedFieldTypes, LinearAlgebra
@reexport using AbstractFFTs

import AbstractFFTs: Plan

include("callgraph.jl")
include("algos.jl")
include("plan.jl")

#=
"""
$(TYPEDSIGNATURES)
Perform a fast Fourier transform of a vector. Preserves types given by the
user.

# Arguments
x::AbstractVector: The vector to transform.

# Examples
```julia
julia> x = rand(ComplexF64, 10)

julia> y = fft(x)
```
"""
function fft(x::AbstractVector{T}) where {T}
    y = similar(x)
    g = CallGraph{T}(length(x))
    fft!(y, x, 1, 1, FFT_FORWARD, g[1].type, g, 1)
    y
end

function fft(x::AbstractVector{T}) where {T <: Real}
    y = similar(x, Complex{T})
    g = CallGraph{Complex{T}}(length(x))
    fft!(y, x, 1, 1, FFT_FORWARD, g[1].type, g, 1)
    y
end

"""
$(TYPEDSIGNATURES)
Perform a fast Fourier transform of a matrix. Preserves types given by the
user.

# Arguments
x::AbstractMatrix: The matrix to transform (columnwise then rowwise).

# Examples
```julia
julia> x = rand(ComplexF64, 10, 10)

julia> y = fft(x)
```
"""
function fft(x::AbstractMatrix{T}) where {T}
    M,N = size(x)
    y1 = similar(x)
    y2 = similar(x)
    g1 = CallGraph{T}(size(x,1))
    g2 = CallGraph{T}(size(x,2))

    for k in 1:N
        @views fft!(y1[:,k],  x[:,k], 1, 1, FFT_FORWARD, g1[1].type, g1, 1)
    end

    for k in 1:M
        @views fft!(y2[k,:], y1[k,:], 1, 1, FFT_FORWARD, g2[1].type, g2, 1)
    end
    y2
end

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

"""
$(TYPEDSIGNATURES)
Perform a backward fast Fourier transform of a vector, where "backward"
indicates the same output signal down to a constant factor. Preserves types
given by the user.

# Arguments
x::AbstractVector: The vector to transform

# Examples
```julia
julia> x = rand(ComplexF64, 10)

julia> y = bfft(x)

julia> z = fft(y)

julia> x ≈ z/10
true
```
"""
function bfft(x::AbstractVector{T}) where {T}
    y = similar(x)
    g = CallGraph{T}(length(x))
    fft!(y, x, 1, 1, FFT_BACKWARD, g[1].type, g, 1)
    y
end

function bfft(x::AbstractVector{T}) where {T <: Real}
    y = similar(x, Complex{T})
    g = CallGraph{Complex{T}}(length(x))
    fft!(y, x, 1, 1, FFT_BACKWARD, g[1].type, g, 1)
    y
end

"""
$(TYPEDSIGNATURES)
Perform a backward fast Fourier transform of a matrix, where "backward"
indicates the same output signal down to a constant factor. Preserves types
given by the user.

# Arguments
x::AbstractMatrix: The matrix to transform

# Examples
```julia
julia> x = rand(ComplexF64, 10, 10)

julia> y = bfft(x)

julia> z = fft(y)

julia> x ≈ z/100
true
```
"""
function bfft(x::AbstractMatrix{T}) where {T}
    M,N = size(x)
    y1 = similar(x)
    y2 = similar(x)
    g1 = CallGraph{T}(size(x,1))
    g2 = CallGraph{T}(size(x,2))

    for k in 1:N
        @views fft!(y1[:,k],  x[:,k], 1, 1, FFT_BACKWARD, g1[1].type, g1, 1)
    end

    for k in 1:M
        @views fft!(y2[k,:], y1[k,:], 1, 1, FFT_BACKWARD, g2[1].type, g2, 1)
    end
    y2
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

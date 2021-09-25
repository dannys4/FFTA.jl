import Base: *
import LinearAlgebra: mul!

struct FFTAInvPlan{T} <: Plan{T} end

@computed struct FFTAPlan{T<:Union{Real, Complex},N} <: Plan{T}
    callgraph::NTuple{N, CallGraph{(T<:Real) ? Complex{T} : T}}
    region
    dir::Direction
    pinv::FFTAInvPlan{T}
end

function AbstractFFTs.plan_fft(x::AbstractArray{T}, region; kwargs...)::FFTAPlan{T} where T
    N = length(region)
    @assert N <= 2 "Only supports vectors and matrices"
    if N == 1
        g = CallGraph{T}(size(x,region[]))
        pinv = FFTAInvPlan{T}()
        return FFTAPlan{T,N}((g,), region, FFT_FORWARD, pinv)
    else
        g1 = CallGraph{T}(size(x,region[1]))
        g2 = CallGraph{T}(size(x,region[2]))
        pinv = FFTAInvPlan{T}()
        return FFTAPlan{T,N}((g1,g2), region, FFT_FORWARD, pinv)
    end
end

function AbstractFFTs.plan_bfft(p::FFTAPlan{T,N}) where {T,N}
    return FFTAPlan{T,N}(p.callgraph, p.region, -p.dir, p.pinv)
end

function LinearAlgebra.mul!(y::AbstractVector{T}, p::FFTAPlan{T,1}, x::AbstractVector{T}) where T
    fft!(y, x, 1, 1, p.dir, p.callgraph[1][1].type, p.callgraph[1], 1)
end

function LinearAlgebra.mul!(y::AbstractArray{T,N}, p::FFTAPlan{T,1}, x::AbstractArray{T,N}) where {T,N}
    Rpre = CartesianIndices(size(x)[1:p.region-1])
    Rpost = CartesianIndices(size(x)[p.region+1:end])
    for Ipre in Rpre
        for Ipost in Rpost
            @views fft!(y[Ipre,:,Ipost], x[Ipre,:,Ipost], 1, 1, p.dir, p.callgraph[1][1].type, p.callgraph[1], 1)
        end
    end
end

function *(p::FFTAPlan{T,1}, x::AbstractVector{T}) where {T<:Union{Real, Complex}}
    y = similar(x)
    LinearAlgebra.mul!(y, p, x)
    y
end
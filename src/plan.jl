struct FFTAInvPlan{T} <: Plan{T} end

@computed struct FFTAPlan{T<:Union{Real, Complex},N} <: Plan{T}
    callgraph::NTuple{N, CallGraph{(T<:Real) ? Complex{T} : T}}
    region::NTuple{N, Int}
    dir::Direction
    pinv::FFTAInvPlan{T}
end

function AbstractFFTs.plan_fft(x::AbstractArray{T}, region; kwargs...)::FFTAPlan{T} where T
    N = length(region)
    @assert N <= 2 "Only supports vectors and matrices"
    if N == 1
        g = CallGraph{T}(size(x,region[]))
        pinv = FFTAInvPlan()
        return FFTAPlan{T,N}((g,), region, FFT_FORWARD, pinv)
    else
        g1 = CallGraph{T}(size(x,region[1]))
        g2 = CallGraph{T}(size(x,region[2]))
        pinv = FFTAInvPlan()
        return FFTAPlan{T,N}((g1,g2), region, FFT_FORWARD, pinv)
    end
end

function AbstractFFTs.plan_bfft(p::FFTAPlan{T,N}) where {T,N}
    return FFTAPlan{T,N}(p.callgraph, p.region, -p.dir, p.pinv)
end

function LinearAlgebra.mul!(y, p::FFTAPlan, x)
    fft!(y, x, 1, 1, p.dir, p.callgraph[1].type, p.callgraph, 1)
end

function *(p::FFTAPlan, x)
    y = similar(x)
    LinearAlgebra.mul!(y, p, x)
    y
end
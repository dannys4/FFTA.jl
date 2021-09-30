import Base: *
import LinearAlgebra: mul!

abstract type FFTAPlan{T,N} <: Plan{T} end

struct FFTAInvPlan{T,N} <: FFTAPlan{T,N} end

struct FFTAPlan_cx{T,N} <: FFTAPlan{T,N}
    callgraph::NTuple{N, CallGraph{T}}
    region::Union{Int,AbstractVector{<:Int}}
    dir::Direction
    pinv::FFTAInvPlan{T}
end

struct FFTAPlan_re{T,N} <: FFTAPlan{T,N}
    callgraph::NTuple{N, CallGraph{Complex{T}}}
    region::Union{Int,AbstractVector{<:Int}}
    dir::Direction
    pinv::FFTAInvPlan{T}
end

function AbstractFFTs.plan_fft(x::AbstractArray{T}, region; kwargs...)::FFTAPlan_cx{T} where {T <: Complex}
    N = length(region)
    @assert N <= 2 "Only supports vectors and matrices"
    if N == 1
        g = CallGraph{T}(size(x,region[]))
        pinv = FFTAInvPlan{T,N}()
        return FFTAPlan_cx{T,N}((g,), region, FFT_FORWARD, pinv)
    else
        sort!(region)
        g1 = CallGraph{T}(size(x,region[1]))
        g2 = CallGraph{T}(size(x,region[2]))
        pinv = FFTAInvPlan{T,N}()
        return FFTAPlan_cx{T,N}((g1,g2), region, FFT_FORWARD, pinv)
    end
end

function AbstractFFTs.plan_bfft(x::AbstractArray{T}, region; kwargs...)::FFTAPlan_cx{T} where {T <: Complex}
    N = length(region)
    @assert N <= 2 "Only supports vectors and matrices"
    if N == 1
        g = CallGraph{T}(size(x,region[]))
        pinv = FFTAInvPlan{T,N}()
        return FFTAPlan_cx{T,N}((g,), region, FFT_BACKWARD, pinv)
    else
        sort!(region)
        g1 = CallGraph{T}(size(x,region[1]))
        g2 = CallGraph{T}(size(x,region[2]))
        pinv = FFTAInvPlan{T,N}()
        return FFTAPlan_cx{T,N}((g1,g2), region, FFT_BACKWARD, pinv)
    end
end

function AbstractFFTs.plan_rfft(x::AbstractArray{T}, region; kwargs...)::FFTAPlan_re{T} where {T <: Real}
    N = length(region)
    @assert N <= 2 "Only supports vectors and matrices"
    if N == 1
        g = CallGraph{Complex{T}}(size(x,region[]))
        pinv = FFTAInvPlan{T,N}()
        return FFTAPlan_re{T,N}((g,), region, FFT_FORWARD, pinv)
    else
        sort!(region)
        g1 = CallGraph{Complex{T}}(size(x,region[1]))
        g2 = CallGraph{Complex{T}}(size(x,region[2]))
        pinv = FFTAInvPlan{T,N}()
        return FFTAPlan_re{T,N}((g1,g2), region, FFT_FORWARD, pinv)
    end
end

function AbstractFFTs.plan_brfft(x::AbstractArray{T}, len, region; kwargs...)::FFTAPlan_cx{T} where {T}
    N = length(region)
    @assert N <= 2 "Only supports vectors and matrices"
    if N == 1
        g = CallGraph{Complex{T}}(size(x,region[]))
        pinv = FFTAInvPlan{T,N}()
        return FFTAPlan_cx{T,N}((g,), region, FFT_BACKWARD, pinv)
    else
        sort!(region)
        g1 = CallGraph{Complex{T}}(size(x,region[1]))
        g2 = CallGraph{Complex{T}}(size(x,region[2]))
        pinv = FFTAInvPlan{T,N}()
        return FFTAPlan_cx{T,N}((g1,g2), region, FFT_BACKWARD, pinv)
    end
end

function AbstractFFTs.plan_bfft(p::FFTAPlan_cx{T,N}) where {T,N}
    return FFTAPlan_cx{T,N}(p.callgraph, p.region, -p.dir, p.pinv)
end

function AbstractFFTs.plan_brfft(p::FFTAPlan_re{T,N}) where {T,N}
    return FFTAPlan_cx{T,N}(p.callgraph, p.region, -p.dir, p.pinv)
end

function LinearAlgebra.mul!(y::AbstractVector{U}, p::FFTAPlan{T,1}, x::AbstractVector{T}) where {T,U}
    fft!(y, x, 1, 1, p.dir, p.callgraph[1][1].type, p.callgraph[1], 1)
end

function LinearAlgebra.mul!(y::AbstractArray{U,N}, p::FFTAPlan{T,1}, x::AbstractArray{T,N}) where {T,U,N}
    Rpre = CartesianIndices(size(x)[1:p.region-1])
    Rpost = CartesianIndices(size(x)[p.region+1:end])
    for Ipre in Rpre
        for Ipost in Rpost
            @views fft!(y[Ipre,:,Ipost], x[Ipre,:,Ipost], 1, 1, p.dir, p.callgraph[1][1].type, p.callgraph[1], 1)
        end
    end
end

function LinearAlgebra.mul!(y::AbstractMatrix{U}, p::FFTAPlan{T,1}, x::AbstractMatrix{T}) where {T,U}
    rows,cols = size(x)[p.region]
    y_tmp = similar(y)
    for k in 1:cols
        @views fft!(y_tmp[:,k],  x[:,k], 1, 1, p.dir, p.callgraph[2][1].type, p.callgraph[2], 1)
    end

    for k in 1:rows
        @views fft!(y[k,:], y_tmp[k,:], 1, 1, p.dir, p.callgraph[1][1].type, p.callgraph[1], 1)
    end
end

function LinearAlgebra.mul!(y::AbstractArray{U,N}, p::FFTAPlan{T,2}, x::AbstractArray{T,N}) where {T,U,N}
    R1 = CartesianIndices(size(x)[1:p.region[1]-1])
    R2 = CartesianIndices(size(x)[p.region[1]+1:p.region[2]-1])
    R3 = CartesianIndices(size(x)[p.region[2]+1:end])
    y_tmp = similar(y, axes(y)[p.region])
    for I1 in R1
        for I2 in R2
            for I3 in R3
                rows,cols = size(x)[p.region]
                for k in 1:cols
                    @views fft!(y_tmp[:,k],  x[I1,:,I2,k,I3], 1, 1, p.dir, p.callgraph[2][1].type, p.callgraph[2], 1)
                end
            
                for k in 1:rows
                    @views fft!(y[I1,k,I2,:,I3], y_tmp[k,:], 1, 1, p.dir, p.callgraph[1][1].type, p.callgraph[1], 1)
                end
            end
        end
    end
end

function *(p::FFTAPlan{T,1}, x::AbstractVector{T}) where {T<:Union{Real,Complex}}
    y = similar(x, T <: Real ? Complex{T} : T)
    LinearAlgebra.mul!(y, p, x)
    y
end

function *(p::FFTAPlan{T,N1}, x::AbstractArray{T,N2}) where {T<:Union{Real, Complex}, N1, N2}
    y = similar(x, T <: Real ? Complex{T} : T)
    LinearAlgebra.mul!(y, p, x)
    y
end

function *(p::FFTAPlan_re{T,1}, x::AbstractVector{T}) where {T<:Union{Real, Complex}}
    y = similar(x, T <: Real ? Complex{T} : T)
    LinearAlgebra.mul!(y, p, x)
    y
end

function *(p::FFTAPlan_re{T,N1}, x::AbstractArray{T,N2}) where {T<:Union{Real, Complex}, N1, N2}
    y = similar(x, T <: Real ? Complex{T} : T)
    LinearAlgebra.mul!(y, p, x)
    y
end
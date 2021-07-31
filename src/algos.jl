@enum Direction FFT_FORWARD FFT_BACKWARD

"""
Power of 2 FFT in place, forward

"""
function pow2FFT!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}) where {T<:Complex}
    N = length(out)
    if N == 1
        out[1] = in[1]
        return
    end
    pow2FFT!(@view(out[1:(end÷2)]), @view(in[1:2:end]), Val(FFT_FORWARD))
    pow2FFT!(@view(out[(end÷2+1):end]), @view(in[2:2:end]), Val(FFT_FORWARD))

    inc = 2*π/N
    w1 = T(cos(inc), -sin(inc));
    wj = T(1,0)
    m = N ÷ 2
    for j in 1:m
        out_j    = out[j]
        out[j]   = out_j + wj*out[j+m]
        out[j+m] = out_j - wj*out[j+m]
        wj *= w1
    end
end

"""
Power of 2 FFT in place, backward

"""
function pow2FFT!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}) where {T<:Complex}
    N = length(out)
    if N == 1
        out[1] = in[1]
        return
    end
    pow2FFT!(@view(out[1:(end÷2)]), @view(in[1:2:end]), Val(FFT_BACKWARD))
    pow2FFT!(@view(out[(end÷2+1):end]), @view(in[2:2:end]), Val(FFT_BACKWARD))

    inc = 2*π/N
    w1 = T(cos(inc), sin(inc));
    wj = T(1,0)
    m = N ÷ 2
    for j in 1:m
        out_j    = out[j]
        out[j]   = out_j + wj*out[j+m]
        out[j+m] = out_j - wj*out[j+m]
        wj *= w1
    end
end
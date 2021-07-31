@enum Direction FFT_FORWARD FFT_BACKWARD
abstract type abstractFFTType end

struct CallGraphNode
    left::CallGraphNode
    right::CallGraphNode
    sz::Int
    type::abstractFFTType
end

struct Pow2FFT end

struct DFT end

"""
Power of 2 FFT in place, forward

"""
function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}, ::Pow2FFT) where {T<:Complex}
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
function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}, ::Pow2FFT) where {T<:Complex}
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

function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}, ::DFT) where {T<:Complex}
    N = length(out)
    inc = 2*π/N
    wn² = wn = w = T(cos(inc), sin(inc));
    wn_1 = T(1., 0.);

    tmp = in[1];
    out .= tmp;
    tmp = sum(in)
    out[1] = tmp;

    wk = wn²;
    for d in 1:N
        for k in (d+1):N
            wk *= wn
            out[d] = in[k]*wk + out[d]
            out[k] = in[d]*wk + out[k]
        end
        wn_1 = wn
        wn *= w
        wn² *= (wn*wn_1)
        wk = wn²
    end
end

function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}, ::DFT) where {T<:Complex}
    N = length(out)
    inc = 2*π/N
    wn² = wn = w = T(cos(inc), -sin(inc));
    wn_1 = T(1., 0.);

    tmp = in[1];
    out .= tmp;
    tmp = sum(in)
    out[1] = tmp;

    wk = wn²;
    for d in 1:N
        for k in (d+1):N
            wk *= wn
            out[d] = in[k]*wk + out[d]
            out[k] = in[d]*wk + out[k]
        end
        wn_1 = wn
        wn *= w
        wn² *= (wn*wn_1)
        wk = wn²
    end
end
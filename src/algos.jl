@enum Direction FFT_FORWARD FFT_BACKWARD
abstract type AbstractFFTType end

struct CallGraphNode
    left::Int
    right::Int
    type::AbstractFFTType
    sz::Int
end

struct CallGraph{T<:Complex}
    nodes::Vector{CallGraphNode}
    workspace::Vector{Vector{T}}
end

getindex(g::CallGraph, i::Int) = g.nodes[i]

left(g::CallGraph, i::Int) = g[i][i+g[i].left]

right(g::CallGraph, i::Int) = g[i][i+g[i].right]

struct CompositeFFT <: AbstractFFTType end

struct Pow2FFT <: AbstractFFTType end

struct DFT <: AbstractFFTType end

fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{<:Direction}, ::AbstractFFTType, ::CallGraph{T}, ::Int) = nothing

function (g::CallGraph{T})(out::AbstractVector{T}, in::AbstractVector{T}, v::Val{<:Direction}, t::AbstractFFTType, idx::Int)
    fft!(out, in, v, t, g, idx)
end

function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}, ::CompositeFFT, g::CallGraph{T}, idx::Int) where {T<:Complex}
    N = out.size()
    left = left(g,i)
    right = right(g,i)
    N1 = length(left)
    N2 = length(right)

    inc = 2*π/N
    w = T(cos(inc), -sin(inc))
    wj = T(1, 0)
    for j1 in 2:N1
        Complex<F,L> wk2 = wj1;
        @views g(out[j1:N1:end], tmp[N2*j1:N2*(j1-1)-1], Val(FFT_FORWARD), right.type, g, idx + g[idx].right)
        for k2 in 2:N2
            tmp[j1*N2+k2] *= wk2
            wk2 *= wj1
        end
        wj1 *= w1
    end

    for k2 in 1:N2
        @views g(tmp[k2:N2:end], out[k2:N2:end], Val(FFT_FORWARD), left.type, g, idx + g[idx].left)
    end
end

"""
Power of 2 FFT in place, forward

"""
function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}, ::Pow2FFT, ::CallGraph{T}, ::Int) where {T<:Complex}
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
function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}, ::Pow2FFT, ::CallGraph{T}, ::Int) where {T<:Complex}
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

function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}, ::DFT, ::CallGraph{T}, ::Int) where {T<:Complex}
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

function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}, ::DFT, ::CallGraph{T}, ::Int) where {T<:Complex}
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
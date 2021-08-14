function alternatingSum(x::AbstractVector{T}) where T
    y = x[1]
    @turbo for i in 2:length(x)
        y += (x[i] * convert(T,(2 * (i % 2) - 1)))
    end
    y
end

fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Direction, ::AbstractFFTType, ::CallGraph{T}, ::Int) where {T} = nothing

@inline function direction_sign(::FFT_BACKWARD)
    1
end

@inline function direction_sign(::FFT_FORWARD)
    -1
end

function (g::CallGraph{T})(out::AbstractVector{T}, in::AbstractVector{U}, v::Direction, t::AbstractFFTType, idx::Int) where {T,U}
    fft!(out, in, v, t, g, idx)
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, d::Direction, ::CompositeFFT, g::CallGraph{T}, idx::Int) where {T,U}
    N = length(out)
    left = leftNode(g,idx)
    right = rightNode(g,idx)
    N1 = left.sz
    N2 = right.sz

    w1 = convert(T, cispi(direction_sign(d)*2/N))
    wj1 = one(T)
    tmp = g.workspace[idx]
    @inbounds for j1 in 1:N1
        wk2 = wj1;
        @views g(tmp[(N2*(j1-1) + 1):(N2*j1)], in[j1:N1:end], d, right.type, idx + g[idx].right)
        j1 > 1 && @inbounds for k2 in 2:N2
            tmp[N2*(j1-1) + k2] *= wk2
            wk2 *= wj1
        end
        wj1 *= w1
    end

    @inbounds for k2 in 1:N2
        @views g(out[k2:N2:end], tmp[k2:N2:end], d, left.type, idx + g[idx].left)
    end
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, d::Direction, a::Pow2FFT, b::CallGraph{T}, c::Int) where {T,U}
    fft_pow2!(out, in, d)
end

"""
Power of 2 FFT in place, complex

"""
function fft_pow2!(out::AbstractVector{T}, in::AbstractVector{T}, d::Direction) where {T}
    N = length(out)
    if N == 2
        out[1] = in[1] + in[2]
        out[2] = in[1] - in[2]
        return
    end
    fft_pow2!(@view(out[1:(end÷2)]),     @view(in[1:2:end]), d)
    fft_pow2!(@view(out[(end÷2+1):end]), @view(in[2:2:end]), d)

    w1 = convert(T, cispi(direction_sign(d)*2/N))
    wj = one(T)
    m = N ÷ 2
    @inbounds for j in 1:m
        out_j    = out[j]
        out[j]   = out_j + wj*out[j+m]
        out[j+m] = out_j - wj*out[j+m]
        wj *= w1
    end
end

"""
Power of 2 FFT in place, real

"""
function fft_pow2!(out::AbstractVector{Complex{T}}, in::AbstractVector{T}, d::Direction) where {T<:Real}
    N = length(out)
    if N == 2
        out[1] = in[1] + in[2]
        out[2] = in[1] - in[2]
        return
    end
    fft_pow2!(@view(out[1:(end÷2)]),     @view(in[1:2:end]), d)
    fft_pow2!(@view(out[(end÷2+1):end]), @view(in[2:2:end]), d)

    w1 = convert(Complex{T}, cispi(direction_sign(d)*2/N))
    wj = one(Complex{T})
    m = N ÷ 2
    @inbounds @turbo for j in 2:m
        out[j] = out[j] + wj*out[j+m]
        wj *= w1
    end
    @inbounds @turbo for j in 2:m
        out[m+j] = conj(out[m-j+2])
    end
end

function fft_dft!(out::AbstractVector{T}, in::AbstractVector{T}, d::Direction) where {T}
    N = length(out)
    wn² = wn = w = convert(T, cispi(direction_sign(d)*2/N))
    wn_1 = one(T)

    tmp = in[1]
    out .= tmp
    tmp = sum(in)
    out[1] = tmp

    wk = wn²
    @inbounds for d in 2:N
        out[d] = in[d]*wk + out[d]
        @inbounds for k in (d+1):N
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

function fft_dft!(out::AbstractVector{Complex{T}}, in::AbstractVector{T}, d::Direction) where {T<:Real}
    N = length(out)
    halfN = N÷2
    wk = wkn = w = convert(Complex{T}, cispi(direction_sign(d)*2/N))

    out[2:N] .= in[1]
    out[1] = sum(in)
    iseven(N) && (out[halfN+1] = alternatingSum(in))
    
    @inbounds for d in 2:halfN+1
        tmp = in[1]
        @inbounds for k in 2:N
            tmp += wkn*in[k]
            wkn *= wk
        end
        out[d] = tmp
        wk *= w
        wkn = wk
    end
    @inbounds @turbo for i in 0:halfN-1
        out[N-i] = conj(out[halfN-i])
    end
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, d::Direction, ::DFT, ::CallGraph{T}, ::Int) where {T,U}
    fft_dft!(out, in, d)
end
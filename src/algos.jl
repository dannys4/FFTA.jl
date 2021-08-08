function alternatingSum(x::AbstractVector{T}) where T
    y = x[1]
    @turbo for i in 2:length(x)
        y += (x[i] * convert(T,(2 * (i % 2) - 1)))
    end
    y
end

fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{<:Direction}, ::AbstractFFTType, ::CallGraph{T}, ::Int) where {T} = nothing

function (g::CallGraph{T})(out::AbstractVector{T}, in::AbstractVector{U}, v::Val{FFT_FORWARD}, t::AbstractFFTType, idx::Int) where {T,U}
    fft!(out, in, v, t, g, idx)
end

function (g::CallGraph{T})(out::AbstractVector{T}, in::AbstractVector{U}, v::Val{FFT_BACKWARD}, t::AbstractFFTType, idx::Int) where {T,U}
    fft!(out, in, v, t, g, idx)
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, ::Val{FFT_FORWARD}, ::CompositeFFT, g::CallGraph{T}, idx::Int) where {T,U}
    N = length(out)
    left = leftNode(g,idx)
    right = rightNode(g,idx)
    N1 = left.sz
    N2 = right.sz

    w1 = convert(T, cispi(-2/N))
    wj1 = one(T)
    tmp = g.workspace[idx]
    for j1 in 1:N1
        wk2 = wj1;
        @views g(tmp[(N2*(j1-1) + 1):(N2*j1)], in[j1:N1:end], Val(FFT_FORWARD), right.type, idx + g[idx].right)
        j1 > 1 && for k2 in 2:N2
            tmp[N2*(j1-1) + k2] *= wk2
            wk2 *= wj1
        end
        wj1 *= w1
    end

    for k2 in 1:N2
        @views g(out[k2:N2:end], tmp[k2:N2:end], Val(FFT_FORWARD), left.type, idx + g[idx].left)
    end
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, ::Val{FFT_BACKWARD}, ::CompositeFFT, g::CallGraph{T}, idx::Int) where {T,U}
    N = length(out)
    left = left(g,i)
    right = right(g,i)
    N1 = left.sz
    N2 = right.sz

    w1 = convert(T, cispi(2/N))
    wj1 = one(T)
    tmp = g.workspace[idx]
    for j1 in 2:N1
        Complex<F,L> wk2 = wj1;
        @views g(tmp[j1:N1:end], in[N2*j1:N2*(j1-1)-1], Val(FFT_BACKWARD), right.type, idx + g[idx].right)
        for k2 in 2:N2
            tmp[j1*N2+k2] *= wk2
            wk2 *= wj1
        end
        wj1 *= w1
    end

    for k2 in 1:N2
        @views g(out[k2:N2:end], tmp[k2:N2:end], Val(FFT_BACKWARD), left.type, idx + g[idx].left)
    end
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, ::Val{FFT_FORWARD}, ::Pow2FFT, ::CallGraph{T}, ::Int) where {T,U}
    fft_pow2!(out, in, Val(FFT_FORWARD))
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, ::Val{FFT_BACKWARD}, ::Pow2FFT, ::CallGraph{T}, ::Int) where {T,U}
    fft_pow2!(out, in, Val(FFT_BACKWARD))
end

"""
Power of 2 FFT in place, forward

"""
function fft_pow2!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}) where {T}
    N = length(out)
    if N == 1
        out[1] = in[1]
        return
    end
    fft_pow2!(@view(out[1:(end÷2)]),     @view(in[1:2:end]), Val(FFT_FORWARD))
    fft_pow2!(@view(out[(end÷2+1):end]), @view(in[2:2:end]), Val(FFT_FORWARD))

    w1 = convert(T, cispi(-2/N))
    wj = one(T)
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
function fft_pow2!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}) where {T}
    N = length(out)
    if N == 1
        out[1] = in[1]
        return
    end
    fft_pow2!(@view(out[1:(end÷2)]),     @view(in[1:2:end]), Val(FFT_BACKWARD))
    fft_pow2!(@view(out[(end÷2+1):end]), @view(in[2:2:end]), Val(FFT_BACKWARD))

    w1 = convert(T, cispi(2/N))
    wj = one(T)
    m = N ÷ 2
    for j in 1:m
        out_j    = out[j]
        out[j]   = out_j + wj*out[j+m]
        out[j+m] = out_j - wj*out[j+m]
        wj *= w1
    end
end

"""
Power of 2 FFT in place, forward

"""
function fft_pow2!(out::AbstractVector{Complex{T}}, in::AbstractVector{T}, ::Val{FFT_FORWARD}) where {T<:Real}
    N = length(out)
    if N == 1
        out[1] = in[1]
        return
    end
    fft_pow2!(@view(out[1:(end÷2)]),     @view(in[1:2:end]), Val(FFT_FORWARD))
    fft_pow2!(@view(out[(end÷2+1):end]), @view(in[2:2:end]), Val(FFT_FORWARD))

    w1 = convert(Complex{T}, cispi(-2/N))
    wj = one(Complex{T})
    m = N ÷ 2
    @turbo for j in 2:m
        out[j] = out[j] + wj*out[j+m]
        wj *= w1
    end
    @turbo for j in 2:m
        out[m+j] = conj(out[m-j+2])
    end
end

"""
Power of 2 FFT in place, backward

"""
function fft_pow2!(out::AbstractVector{Complex{T}}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}) where {T<:Real}
    N = length(out)
    if N == 1
        out[1] = in[1]
        return
    end
    fft_pow2!(@view(out[1:(end÷2)]),     @view(in[1:2:end]), Val(FFT_BACKWARD))
    fft_pow2!(@view(out[(end÷2+1):end]), @view(in[2:2:end]), Val(FFT_BACKWARD))

    w1 = convert(Complex{T}, cispi(2/N))
    wj = one(Complex{T})
    m = N ÷ 2
    @turbo for j in 2:m
        out[j] = out[j] + wj*out[j+m]
        out[m+j] = conj(out[m-i+2])
        wj *= w1
    end
end

function fft_dft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}) where {T}
    N = length(out)
    wn² = wn = w = convert(T, cispi(-2/N))
    wn_1 = one(T)

    tmp = in[1];
    out .= tmp;
    tmp = sum(in)
    out[1] = tmp;

    wk = wn²;
    for d in 2:N
        out[d] = in[d]*wk + out[d]
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

function fft_dft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}) where {T}
    N = length(out)
    wn² = wn = w = convert(T, cispi(2/N))
    wn_1 = one(T)

    tmp = in[1]
    out .= tmp
    tmp = sum(in)
    out[1] = tmp

    wk = wn²;
    for d in 2:N
        out[d] = in[d]*wk + out[d]
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

function fft_dft!(out::AbstractVector{Complex{T}}, in::AbstractVector{T}, ::Val{FFT_FORWARD}) where {T<:Real}
    N = length(out)
    halfN = N÷2
    wk = wkn = w = convert(Complex{T}, cispi(-2/N))

    out[2:N] .= in[1]
    out[1] = sum(in)
    iseven(N) && (out[halfN+1] = alternatingSum(in))
    
    for d in 2:halfN+1
        tmp = in[1]
        for k in 2:N
            tmp += wkn*in[k]
            wkn *= wk
        end
        out[d] = tmp
        wk *= w
        wkn = wk
    end
    @turbo for i in 0:halfN-1
        out[N-i] = conj(out[halfN-i])
    end
end

function fft_dft!(out::AbstractVector{Complex{T}}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}) where {T<:Real}
    N = length(out)
    halfN = N÷2
    wn² = wn = w = convert(Complex{T}, cispi(2/N))
    wn_1 = one(T)

    out .= in[1]
    out[1] = sum(in)
    iseven(N) && (out[halfN+1] = alternatingSum(in))

    wk = wn²;
    for d in 2:halfN
        out[d] = in[d]*wk + out[d]
        for k in (d+1):halfN
            wk *= wn
            out[d] = in[k]*wk + out[d]
            out[k] = in[d]*wk + out[k]
        end
        wn_1 = wn
        wn *= w
        wn² *= (wn*wn_1)
        wk = wn²
    end
    out[(N-halfN+2):end] .= conj.(out[halfN:-1:2])
end


function fft!(out::AbstractVector{T}, in::AbstractVector{U}, ::Val{FFT_FORWARD}, ::DFT, ::CallGraph{T}, ::Int) where {T,U}
    fft_dft!(out, in, Val(FFT_FORWARD))
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, ::Val{FFT_BACKWARD}, ::DFT, ::CallGraph{T}, ::Int) where {T,U}
    fft_dft!(out, in, Val(FFT_BACKWARD))
end

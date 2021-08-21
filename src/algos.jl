fft!(::AbstractVector{T}, ::AbstractVector{T}, ::Int, ::Int, ::Direction, ::AbstractFFTType, ::CallGraph{T}, ::Int) where {T} = nothing

@inline function direction_sign(d::Direction)
    Int(d)
end

function (g::CallGraph{T})(out::AbstractVector{T}, in::AbstractVector{U}, start_out::Int, start_in::Int, v::Direction, t::AbstractFFTType, idx::Int) where {T,U}
    fft!(out, in, start_out, start_in, v, t, g, idx)
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, start_out::Int, start_in::Int, d::Direction, ::CompositeFFT, g::CallGraph{T}, idx::Int) where {T,U}
    root = g[idx]
    left_idx = idx + root.left
    right_idx = idx + root.right
    left = g[left_idx]
    right = g[right_idx]
    N  = root.sz
    N1 = left.sz
    N2 = right.sz
    s_in = root.s_in
    s_out = root.s_out

    w1 = convert(T, cispi(direction_sign(d)*2/N))
    wj1 = one(T)
    tmp = g.workspace[idx]
    @inbounds for j1 in 0:N1-1
        wk2 = wj1
        g(tmp, in, N2*j1+1, start_in + j1*s_in, d, right.type, right_idx)
        j1 > 0 && @inbounds for k2 in 1:N2-1
            tmp[N2*j1 + k2 + 1] *= wk2
            wk2 *= wj1
        end
        wj1 *= w1
    end

    @inbounds for k2 in 0:N2-1
        g(out, tmp, start_out + k2*s_out, k2+1, d, left.type, left_idx)
    end
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, start_out::Int, start_in::Int, d::Direction, ::Pow2FFT, g::CallGraph{T}, idx::Int) where {T,U}
    root = g[idx]
    N = root.sz
    s_in = root.s_in
    s_out = root.s_out
    fft_pow2!(out, in, N, start_out, s_out, start_in, s_in, d)
end

function fft_dft!(out::AbstractVector{T}, in::AbstractVector{T}, N::Int, start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction) where {T}
    tmp = in[start_in]
    @inbounds for j in 1:N-1
        tmp += in[start_in + j*stride_in]
    end
    out[start_out] = tmp
    
    wk = wkn = w = convert(T, cispi(direction_sign(d)*2/N))
    @inbounds for d in 1:N-1
        tmp = in[start_in]
        @inbounds for k in 1:N-1
            tmp += wkn*in[start_in + k*stride_in]
            wkn *= wk
        end
        out[start_out + d*stride_out] = tmp
        wk *= w
        wkn = wk
    end
end

function fft_dft!(out::AbstractVector{Complex{T}}, in::AbstractVector{T}, N::Int, start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction) where {T<:Real}
    halfN = N÷2
    wk = wkn = w = convert(Complex{T}, cispi(direction_sign(d)*2/N))

    tmpBegin = tmpHalf = in[start_in]
    @inbounds for j in 1:N-1
        tmpBegin += in[start_in + stride_in*j]
        iseven(j) ? tmpHalf += in[start_in + stride_in*j] : tmpHalf -= in[start_in + stride_in*j]
    end
    out[start_out] = convert(Complex{T}, tmpBegin)
    iseven(N) && (out[start_out + stride_out*halfN] = convert(Complex{T}, tmpHalf))
    
    @inbounds for d in 1:halfN
        tmp = in[start_in]
        @inbounds for k in 1:N-1
            tmp += wkn*in[start_in + k*stride_in]
            wkn *= wk
        end
        out[start_out + d*stride_out] = tmp
        wk *= w
        wkn = wk
    end
    @inbounds @turbo for k in halfN+1:N-1
        out[start_out + stride_out*k] = conj(out[start_out + stride_out*(N-k)])
    end
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, start_out::Int, start_in::Int, d::Direction, ::DFT, g::CallGraph{T}, idx::Int) where {T,U}
    root = g[idx]
    fft_dft!(out, in, root.sz, start_out, root.s_out, start_in, root.s_in, d)
end

"""
Power of 2 FFT in place

"""
function fft_pow2!(out::AbstractVector{T}, in::AbstractVector{U}, N::Int, start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction) where {T, U}
    if N == 2
        out[start_out]              = in[start_in] + in[start_in + stride_in]
        out[start_out + stride_out] = in[start_in] - in[start_in + stride_in]
        return
    end
    m = N ÷ 2

    fft_pow2!(out, in, m, start_out               , stride_out, start_in            , stride_in*2, d)
    fft_pow2!(out, in, m, start_out + m*stride_out, stride_out, start_in + stride_in, stride_in*2, d)

    w1 = convert(T, cispi(direction_sign(d)*2/N))
    wj = one(T)
    @inbounds for j in 0:m-1
        j1_out = start_out + j*stride_out
        j2_out = start_out + (j+m)*stride_out
        out_j    = out[j1_out]
        out[j1_out] = out_j + wj*out[j2_out]
        out[j2_out] = out_j - wj*out[j2_out]
        wj *= w1
    end
end

"""
Power of 4 FFT in place

"""
function fft_pow4!(out::AbstractVector{T}, in::AbstractVector{U}, N::Int, start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction) where {T, U}
    ds = direction_sign(d)
    if N == 4
        y[start_out +       0] = x[start_in] + x[start_in + s_in] +
                                 x[start_in + 2*s_in] + x[start_in + 3*s_in]
        y[start_out +   s_out] = x[start_in] + x[start_in + s_in]*convert(T,float(ds*1im)) -
                                 x[start_in + 2*s_in] + x[start_in + 3*s_in]*convert(T,float(-ds*1im))
        y[start_out + 2*s_out] = x[start_in] - x[start_in + s_in] +
                                 x[start_in + 2*s_in] - x[start_in + 3*s_in]
        y[start_out + 3*s_out] = x[start_in] + x[start_in + s_in]*convert(T,float(-ds*1im)) -
                                 x[start_in + 2*s_in] + x[start_in + 3*s_in]*convert(T,float(ds*1im))
        return
    end
    m = N ÷ 4

    fft_pow4!(out, in, m, start_out                 , stride_out, start_in              , stride_in*4, d)
    fft_pow4!(out, in, m, start_out +   m*stride_out, stride_out, start_in +   stride_in, stride_in*4, d)
    fft_pow4!(out, in, m, start_out + 2*m*stride_out, stride_out, start_in + 2*stride_in, stride_in*4, d)
    fft_pow4!(out, in, m, start_out + 3*m*stride_out, stride_out, start_in + 3*stride_in, stride_in*4, d)

    w1 = convert(T, cispi(direction_sign(d)*2/N))
    wj = one(T)
    
    inc = 2.*π/N;
    w1 = convert(T, cispi(ds*2/N))
    w2 = convert(T, cispi(ds*4/N))
    w3 = convert(T, cispi(ds*6/N))
    wk1 = wk2 = wk3 = one(T)

    @inbounds for k in 0:m-1
        k0 = start_out + k*stride_out
        k1 = start_out + (k+m)*stride_out
        k2 = start_out + (k+2*m)*stride_out
        k3 = start_out + (k+3*m)*stride_out
        y_k0, y_k1, y_k2, y_k3 = out[[k0, k1, k2, k3]]
        out[k0] = (y_k0 + y_k2*wk2) + (y_k1*wk1 + y_k3*wk2)
        out[k1] = (y_k0 - y_k2*wk2) + (y_k1*wk1 - y_k3*wk3) * convert(T, float(ds*1im))
        out[k2] = (y_k0 + y_k2*wk2) - (y_k1*wk1 + y_k3*wk3)
        out[k3] = (y_k0 - y_k2*wk2) + (y_k1*wk1 - y_k3*wk3) * convert(T, float(-ds*1im))
        wk1 *= w1
        wk2 *= w2
        wk3 *= w3
    end
end

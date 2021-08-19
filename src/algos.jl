function alternatingSum(x::AbstractVector{T}) where T
    y = x[1]
    @turbo for i in 2:length(x)
        y += (x[i] * convert(T,(2 * (i % 2) - 1)))
    end
    y
end

fft!(::AbstractVector{T}, ::AbstractVector{T}, ::Int, ::Int, ::Direction, ::AbstractFFTType, ::CallGraph{T}, ::Int) where {T} = nothing

@inline function direction_sign(::FFT_BACKWARD)
    1.
end

@inline function direction_sign(::FFT_FORWARD)
    -1.
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
    for j1 in 0:N1-1
        wk2 = wj1
        g(tmp, in, N2*j1+1, start_in + j1*s_in, d, right.type, right_idx)
        j1 > 0 && for k2 in 1:N2-1
            tmp[N2*j1 + k2 + 1] *= wk2
            wk2 *= wj1
        end
        wj1 *= w1
    end

    for k2 in 0:N2-1
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

"""
Power of 2 FFT in place, complex

"""
function fft_pow2!(out::AbstractVector{T}, in::AbstractVector{T}, N::Int, start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction) where {T}

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
Power of 2 FFT in place, real

"""
function fft_pow2!(out::AbstractVector{T}, in::AbstractVector{T}, N::Int, start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction) where {T<:Real}
    if N == 2
        out[1] = in[1] + in[2]
        out[2] = in[1] - in[2]
        return
    end
    m = N ÷ 2
    fft_pow2!(out, in, m, start_out    , stride_out, start_in    , stride_in*2, d)
    fft_pow2!(out, in, m, start_out + m, stride_out, start_in + 1, stride_in*2, d)

    w1 = convert(Complex{T}, cispi(direction_sign(d)*2/N))
    wj = one(Complex{T})
    @inbounds @turbo for j in 1:m-1
        j1_out = start_out + j*stride_out
        j2_out = start_out + (j+m)*stride_out
        out[j1_out] = out[j1_out] + wj*out[j2_out]
        wj *= w1
    end
    @inbounds @turbo for j in 1:m-1
        j1_out = start_out + (j+m)*stride_out
        j2_out = start_out + (m-j+1)*stride_out
        out[j1_out] = conj(out[j2_out])
    end
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

    out[start_out + 1:stride_out:start_out+stride_out*N] .= in[1]
    out[1] = sum(@view in[start_in:stride_in:start_int+stride_out*N])
    iseven(N) && (out[start_out + stride_out*halfN] = alternatingSum(@view in[start_in:stride_in:start_int+stride_out*N]))
    
    @inbounds for d in 2:halfN+1
        tmp = in[start_in]
        @inbounds for k in 2:N
            tmp += wkn*in[start_in + k*stride_in]
            wkn *= wk
        end
        out[start_out + d*stride_out] = tmp
        wk *= w
        wkn = wk
    end
    @inbounds @turbo for i in 0:halfN-1
        out[start_out + stride_out*(N-i)] = conj(out[start_out + stride_out*(halfN-i)])
    end
end

function fft!(out::AbstractVector{T}, in::AbstractVector{U}, start_out::Int, start_in::Int, d::Direction, ::DFT, g::CallGraph{T}, idx::Int) where {T,U}
    root = g[idx]
    fft_dft!(out, in, root.sz, start_out, root.s_out, start_in, root.s_in, d)
end
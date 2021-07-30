@enum Direction FFT_FORWARD FFT_BACKWARD

function pow2FFT!(out::AbstractArray{T,0}, in::AbstractArray{T,0}, ::Val) where T
    out[] = in[]
end

function pow2FFT!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}) where {T<:Complex}
    N = length(out)
    if N == 1
        out[1] = in[1]
        return
    end
    pow2FFT!(@view(out[1:2:end]), @view(in[1:2:end]), Val(FFT_FORWARD))
    pow2FFT!(@view(out[2:2:end]), @view(in[2:2:end]), Val(FFT_FORWARD))

    inc = 2*π/N
    w1 = T(cos(inc), -sin(inc));
    wj = T(1,0)
    m = N ÷ 2
    for j in 1:m
        out_j    = out[j]
        println(out_j)
        out[j]   = out_j + wj*out[j+m]
        println(out_j)
        println()
        out[j+m] = out_j - wj*out[j+m]
        wj *= w1
    end
    println()
end

even_zeroindexed_sv(x::SVector{N,T}) where {N,T} = SVector{N÷2}(x[1:2:end])

function myfft_sv(arr::SVector{N, T}) where {N, T<:Complex}
    Nhalf = N/2
    ft_even =  myfft_sv(even_zeroindexed_sv(arr))
    ft_odd =  myfft_sv(odd_zeroindexed_sv(arr))
    f(e, o, n) = e + exp(-2im * T(π*n/N)) * o
    ft_first = map(f, ft_even, ft_odd, 0:Nhalf-1)
    ft_second = map(f, ft_even, ft_odd, Nhalf:N-1)
    vcat(ft_first, ft_second)
end
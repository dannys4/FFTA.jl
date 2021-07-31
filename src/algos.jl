@enum Direction FFT_FORWARD FFT_BACKWARD

function pow2FFT!(out::AbstractArray{T,0}, in::AbstractArray{T,1}, ::Val) where T
    out[] = in[1]
end

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
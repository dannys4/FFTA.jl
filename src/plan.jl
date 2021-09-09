@computed struct FFTAPlan{T} <: Plan{T} where {T <: Union{Real, Complex}}
    callgraph::CallGraph{(T<:Real) ? Complex{T} : T}
    pinv::FFTAPlan{T}
end

function AbstractFFTs.plan_fft(x::AbstractArray{T,N}, region; kwargs...)::FFTAPlan{T}
    @assert N <= 2 "Only supports vectors and matrices"
end
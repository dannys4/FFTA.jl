@enum Direction FFT_FORWARD FFT_BACKWARD
abstract type AbstractFFTType end

# Represents a Composite Cooley-Tukey FFT
struct CompositeFFT <: AbstractFFTType end

# Represents a Radix-2 Cooley-Tukey FFT
struct Pow2FFT <: AbstractFFTType end

# Represents an O(N²) DFT
struct DFT <: AbstractFFTType end

"""
$(TYPEDSIGNATURES)
Node of a call graph

# Arguments
`left::Int`- Offset to the left child node
`right::Int`- Offset to the right child node
`type::AbstractFFTType`- Object representing the type of FFT
`sz::Int`- Size of this FFT

# Examples
```julia
julia> CallGraphNode(0, 0, Pow2FFT(), 8)
```
"""
struct CallGraphNode
    left::Int
    right::Int
    type::AbstractFFTType
    sz::Int
end

"""
$(TYPEDSIGNATURES)
Object representing a graph of FFT Calls

# Arguments
`nodes::Vector{CallGraphNode}`- Nodes keeping track of the graph
`workspace::Vector{Vector{T}}`- Preallocated Workspace

# Examples
```julia
julia> CallGraph{ComplexF64}(CallGraphNode[], Vector{T}[])
```
"""
struct CallGraph{T<:Complex}
    nodes::Vector{CallGraphNode}
    workspace::Vector{Vector{T}}
end

# Get the node in the graph at index i
Base.getindex(g::CallGraph{T}, i::Int) where {T} = g.nodes[i]

# Get the left child of the node at index `i`
leftNode(g::CallGraph, i::Int) = g[i+g[i].left]

# Get the right child of the node at index `i`
rightNode(g::CallGraph, i::Int) = g[i+g[i].right]

# Recursively instantiate a set of `CallGraphNode`s
function CallGraphNode!(nodes::Vector{CallGraphNode}, N::Int, workspace::Vector{Vector{T}})::Int where {T}
    facs = factor(N)
    Ns = [first(x) for x in collect(facs) for _ in 1:last(x)]
    if length(Ns) == 1 || Ns[end] == 2
        push!(workspace, T[])
        push!(nodes, CallGraphNode(0,0,Ns[end] == 2 ? Pow2FFT() : DFT(),N))
        return 1
    end

    if Ns[1] == 2
        N1 = prod(Ns[Ns .== 2])
    else
        # Greedy search for closest factor of N to sqrt(N)
        Nsqrt = sqrt(N)
        N_cp = cumprod(Ns[end:-1:1])[end:-1:1]
        N_prox = abs.(N_cp .- Nsqrt)
        _,N1_idx = findmin(N_prox)
        N1 = N_cp[N1_idx]
    end
    N2 = N ÷ N1
    push!(nodes, CallGraphNode(0,0,DFT(),N))
    sz = length(nodes)
    push!(workspace, Vector{T}(undef, N))
    left_len = CallGraphNode!(nodes, N1, workspace)
    right_len = CallGraphNode!(nodes, N2, workspace)
    nodes[sz] = CallGraphNode(1, 1 + left_len, CompositeFFT(), N)
    return 1 + left_len + right_len
end

# Instantiate a CallGraph from a number `N`
function CallGraph{T}(N::Int) where {T}
    nodes = CallGraphNode[]
    workspace = Vector{Vector{T}}()
    CallGraphNode!(nodes, N, workspace)
    CallGraph(nodes, workspace)
end

function fft(x::AbstractVector{T}) where {T}
    y = similar(x)
    g = CallGraph{T}(length(x))
    fft!(y, x, Val(FFT_FORWARD), g[1].type, g, 1)
    y
end

function fft(x::AbstractMatrix{T}) where {T}
    M,N = size(x)
    y1 = similar(x)
    y2 = similar(x)
    g1 = CallGraph{T}(size(x,1))
    g2 = CallGraph{T}(size(x,2))

    for k in 1:N
        @views fft!(y1[:,k],  x[:,k], Val(FFT_FORWARD), g1[1].type, g1, 1)
    end

    for k in 1:M
        @views fft!(y2[k,:], y1[k,:], Val(FFT_FORWARD), g2[1].type, g2, 1)
    end
    y2
end

function bfft(x::AbstractVector{T}) where {T}
    y = similar(x)
    g = CallGraph{T}(length(x))
    fft!(y, x, Val(FFT_BACKWARD), g[1].type, g, 1)
    y
end

function bfft(x::AbstractMatrix{T}) where {T}
    M,N = size(x)
    y1 = similar(x)
    y2 = similar(x)
    g1 = CallGraph{T}(size(x,1))
    g2 = CallGraph{T}(size(x,2))

    for k in 1:N
        @views fft!(y1[:,k],  x[:,k], Val(FFT_BACKWARD), g1[1].type, g1, 1)
    end

    for k in 1:M
        @views fft!(y2[k,:], y1[k,:], Val(FFT_BACKWARD), g2[1].type, g2, 1)
    end
    y2
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
    for j in 2:m
        out[j]   = out[j] + wj*out[j+m]
        wj *= w1
    end
    out[m+2:end] = conj.(out[m:-1:2])
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
    for j in 2:m
        out[j]   = out[j] + wj*out[j+m]
        wj *= w1
    end
    out[m+2:end] = conj.(out[m:-1:2])
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
    wn² = wn = w = convert(T, cispi(-2/N))
    wn_1 = one(T)

    out .= in[1]
    out[1] = sum(in)
    iseven(N) && (out[halfN+1] = foldr(-,in))

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

function fft_dft!(out::AbstractVector{Complex{T}}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}) where {T<:Real}
    N = length(out)
    halfN = N÷2
    wn² = wn = w = convert(T, cispi(2/N))
    wn_1 = one(T)

    out .= in[1]
    out[1] = sum(in)
    iseven(N) && (out[halfN+1] = foldr(-,in))

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


function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_FORWARD}, ::DFT, ::CallGraph{T}, ::Int) where {T}
    fft_dft!(out, in, Val(FFT_FORWARD))
end

function fft!(out::AbstractVector{T}, in::AbstractVector{T}, ::Val{FFT_BACKWARD}, ::DFT, ::CallGraph{T}, ::Int) where {T}
    fft_dft!(out, in, Val(FFT_BACKWARD))
end

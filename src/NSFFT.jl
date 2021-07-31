module NSFFT

using Primes, StaticArrays, LoopVectorization, DocStringExtensions
import Base: getindex
export pow2FFT!, DFT!

include("algos.jl")

end

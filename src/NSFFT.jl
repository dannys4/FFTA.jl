module NSFFT

using Primes, StaticArrays, LoopVectorization
import Base: getindex
export pow2FFT!, DFT!

include("algos.jl")

end

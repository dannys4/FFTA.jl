module NSFFT

using Primes, StaticArrays, LoopVectorization
export pow2FFT!, DFT!

include("algos.jl")

end

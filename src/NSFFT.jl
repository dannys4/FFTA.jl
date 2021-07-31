module NSFFT

using Primes, StaticArrays, LoopVectorization, DocStringExtensions
import Base: getindex
import AbstractFFTs: fft
export pow2FFT!, DFT!

include("algos.jl")

end

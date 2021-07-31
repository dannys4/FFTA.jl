module NSFFT

using Primes, StaticArrays, LoopVectorization, DocStringExtensions
import Base: getindex
export fft, bfft

include("algos.jl")

end

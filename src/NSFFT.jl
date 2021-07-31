module NSFFT

using Primes, StaticArrays, LoopVectorization, DocStringExtensions
import Base: getindex
export fft

include("algos.jl")

end

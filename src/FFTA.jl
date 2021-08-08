module FFTA

using Primes, DocStringExtensions, LoopVectorization
import Base: getindex
export fft, bfft

include("algos.jl")

end

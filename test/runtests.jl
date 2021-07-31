module NSFFT_test
using ReTest

@testset verbose = true "Unit Tests" begin
    include("testNSFFT.jl")
end
end
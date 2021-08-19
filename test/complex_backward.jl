using FFTA, Test

@testset verbose = true "bfft 1D complex, size $N" for N in [8, 11, 15, 100]
    x = ones(ComplexF64, N)
    y = bfft(x)
    @test y[1] ≈ N && y[2:end] ≈ 0*x[2:end] atol=1e-12
end
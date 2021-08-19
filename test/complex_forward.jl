using FFTA, Test

@testset verbose = true "fft 1D complex, size $N" for N in [8, 11, 15, 100]
    x = zeros(ComplexF64, N)
    x[1] = 1
    y = fft(x)
    @test y ≈ ones(size(x))
end
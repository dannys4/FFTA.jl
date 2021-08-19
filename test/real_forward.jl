using FFTA, Test
test_nums = [8, 11, 15, 100]
@testset verbose = true "fft 1D real, size $(padnum(maximum(test_nums),N))" for N in test_nums
    x = zeros(Float64, N)
    x[1] = 1
    y = fft(x)
    @test y ≈ ones(size(x))
end
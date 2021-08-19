using FFTA, Test
test_nums = [8, 11, 15, 100]
@testset verbose = true " forward" begin
    for N in test_nums
        x = zeros(ComplexF64, N)
        x[1] = 1
        y = fft(x)
        @test y ≈ ones(size(x))
    end
end
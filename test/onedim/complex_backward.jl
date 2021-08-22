using FFTA, Test
test_nums = [8, 11, 15, 16, 27, 100]
@testset "backward" begin
    for N in test_nums
        x = ones(ComplexF64, N)
        y = bfft(x)
        y_ref = 0*y
        y_ref[1] = N
        @test y ≈ y_ref atol=1e-12
    end
end
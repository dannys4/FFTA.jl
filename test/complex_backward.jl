using FFTA, Test
test_nums = [8, 11, 15, 100]
@testset "backward" begin
    for N in test_nums
        x = ones(ComplexF64, N)
        y = bfft(x)
        b1 = isapprox(y[1], N, atol=1e-12)
        b2 = isapprox(y[2:end], 0*x[2:end], atol=1e-12)
        @test b1 && b2
    end
end
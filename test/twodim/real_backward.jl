using FFTA, Test
test_nums = [8]
@testset "backward" begin
    for N in test_nums
        x = ones(Float64, N, N)
        y = brfft(x, 2(N-1))
        y_ref = 0*y
        y_ref[1] = N*(2(N-1))
        @test y_ref ≈ y atol=1e-12
    end
end
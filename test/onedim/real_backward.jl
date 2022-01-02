using FFTA, Test, LinearAlgebra
test_nums = [8, 11, 15, 16, 27, 100]
@testset "backward" begin
    for N in test_nums
        x = ones(Float64, N)
        y = brfft(x, 2*(N-1))
        y_ref = 0*y
        y_ref[1] = 2*(N-1)
        if !isapprox(y_ref, y, atol=1e-12)
            println(norm(y_ref - y))
        end
        @test y_ref ≈ y atol=1e-12
    end
end
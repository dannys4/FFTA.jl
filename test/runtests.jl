using Test, Random

function padnum(m,x)
    digs = floor(Int, log10(m))
    digs_x = floor(Int, log10(x))
    v = fill(' ', digs-digs_x)
    for d in digits(x)[end:-1:1] push!(v, '0' + d) end
    String(v)
end

Random.seed!(1)
@testset verbose = true "FFTA" begin
    @testset verbose = true "1D" begin
        @testset verbose = true "Complex" begin
            include("onedim/complex_forward.jl")
            include("onedim/complex_backward.jl")
            x = rand(ComplexF64, 100)
            y = fft(x)
            x2 = bfft(y)/length(x)
            @test x ≈ x2 atol=1e-12
        end
        @testset verbose = true "Real" begin
            include("onedim/real_forward.jl")
            include("onedim/real_backward.jl")
            x = rand(Float64, 100)
            y = fft(x)
            x2 = bfft(y)/length(x)
            @test x ≈ x2 atol=1e-12
        end
    end
    @testset verbose = true "2D" begin
        @testset verbose = true "Complex" begin
            include("twodim/complex_forward.jl")
            include("twodim/complex_backward.jl")
            x = rand(ComplexF64, 100, 100)
            y = fft(x)
            x2 = bfft(y)/length(x)
            @test x ≈ x2
        end
        @testset verbose = true "Real" begin
            include("twodim/real_forward.jl")
            include("twodim/real_backward.jl")
            x = rand(Float64, 100, 100)
            y = fft(x)
            x2 = bfft(y)/length(x)
            @test x ≈ x2
        end
    end
end
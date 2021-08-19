using Test

function padnum(m,x)
    digs = floor(Int, log10(m))
    digs_x = floor(Int, log10(x))
    v = fill(' ', digs-digs_x)
    for d in digits(x)[end:-1:1] push!(v, '0' + d) end
    String(v)
end

@testset verbose = true "1D" begin
    @testset verbose = true "Complex" begin
        include("complex_forward.jl")
        include("complex_backward.jl")
    end
    @testset verbose = true "Real" begin
        # include("real_forward.jl")
        include("real_backward.jl")
    end
end
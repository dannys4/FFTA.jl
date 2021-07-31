using FFTA

for N in [8, 11, 15, 100]
    x = zeros(ComplexF64, N)
    x[1] = 1
    y = fft(x)
    @test y ≈ ones(size(x))
end
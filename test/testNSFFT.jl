using NSFFT, FFTW, Profile, BenchmarkTools

N = 100
x = ComplexF64.(collect(1:N))
y1 = NSFFT.fft(x)
y2 = FFTW.fft(x)
# @btime NSFFT.fft(x) setup=(x = rand(ComplexF64, N))
# @btime FFTW.fft(x) setup=(x = rand(ComplexF64, N))
println(y1 ≈ y2)
hcat(y1,y2)
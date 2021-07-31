using NSFFT, FFTW, Profile

N = 15
x = rand(ComplexF64, N)
y1 = NSFFT.fft(x)
y2 = FFTW.fft(x)
y1 ≈ y2
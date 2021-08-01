# FFTA: Fastest Fourier Transform in my Apartment
## A library by Danny Sharp

This is a *pure Julia* implementation of FFTs, with the goal that this could supplant other FFTs for applications that require odd Julia objects. Currently this supports `AbstractArray{T,N}` for `T<:Complex` and `N` in `{1,2}` (i.e. `AbstractVector` and `AbstractMatrix`). If you're looking for more performance, checkout `FFTW.jl`.

Path Forward:
- Dispatch on `Real`
- `@inbounds` use
- Instead of `@view`, just `vec` and `ArrayInterfaces.restructure` and then use `CPtr` instead.
- Use `StaticArrays` for the workspace in small cases
- Strictly generate code for certain cases
- Create a SIMD type for Complex numbers
- E-Graphs for the call-graph
- Accelerate dynamic dispatching?
- Other performance left on the table....

Interface:
- `fft(x::AbstractVector{<:Complex})`-- Forward FFT
- `fft(x::AbstractMatrix{<:Complex})`-- Forward FFT
- `bfft(x::AbstractVector{<:Complex})`-- Backward FFT (unscaled inverse FFT)
- `bfft(x::AbstractMatrix{<:Complex})`-- Backward FFT (unscaled inverse FFT)

NOTE: Currently, my C++ code is actually faster than this, so "Fastest Fourier Transform in my Apartment" is a bit of a misnomer.

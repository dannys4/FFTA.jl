# FFTA: Fastest Fourier Transform in my Apartment
## A library by Danny Sharp

This is a *pure Julia* implementation of FFTs, with the goal that this could supplant other FFTs for applications that require odd Julia objects. Currently this supports `AbstractArray{T,N}` where `N` in `{1,2}` (i.e. `AbstractVector` and `AbstractMatrix`). If you're looking for more performance, checkout `FFTW.jl`. The only functions that need to be defined with `T` (besides arithmetic) are `convert(T, x::ComplexF64)` and `one(T)`. This means that `T<:Real` probably doesn't work yet (see Path Forward).

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

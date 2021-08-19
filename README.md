# FFTA: Fastest Fourier Transform in my Apartment
## A library by Danny Sharp

[![Github Action CI](https://github.com/dannys4/FFTA.jl/workflows/CI/badge.svg)](https://github.com/dannys4/FFTA.jl/actions)

This is a *pure Julia* implementation of FFTs, with the goal that this could supplant other FFTs for applications that require odd Julia objects. Currently this supports `AbstractArray{T,N}` where `N` in `{1,2}` (i.e. `AbstractVector` and `AbstractMatrix`). If you're looking for more performance, checkout `FFTW.jl`. Regardless of `T`, `one(::Type{T})` must be defined. Additionally, if `T<:Complex`, then `convert(::Type{T},ComplexF64)` has to be defined and if `T<:Real`, then `convert(::Type{T}, Float64)` has to be defined.

Path Forward:
- Make the code more readable.
- Use `@inbounds`
- Use `AbstractFFTs` interface
- Instead of `@view`, just `vec` and `ArrayInterfaces.restructure` and then use `CPtr` instead.
- Use `StaticArrays` for the workspace in small cases
- Strictly generate code for certain cases
- Create a SIMD type for Complex numbers
- E-Graphs for the call-graph
- Accelerate dynamic dispatching?
- Other performance left on the table....

Interface:
- `fft(x::AbstractVector{<:Union{Real,Complex})`-- Forward FFT
- `fft(x::AbstractMatrix{<:Union{Real,Complex}})`-- Forward FFT
- `bfft(x::AbstractVector{<:Union{Real,Complex}})`-- Backward FFT (unscaled inverse FFT)
- `bfft(x::AbstractMatrix{<:Union{Real,Complex}})`-- Backward FFT (unscaled inverse FFT)

NOTE: Currently, my C++ code is actually faster than this, so "Fastest Fourier Transform in my Apartment" is a bit of a misnomer.

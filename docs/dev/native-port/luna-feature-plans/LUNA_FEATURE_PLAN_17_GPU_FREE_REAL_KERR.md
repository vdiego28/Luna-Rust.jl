# Luna feature plan 17 — CUDA free-space RealGrid Kerr foundation

Status: complete (2026-08-08). Depends on plan 03; standing CUDA CI remains
strongly preferred.

## Outcome

`TransFree` + RealGrid + scalar Kerr runs entirely resident on CUDA using a
joint three-dimensional transform and agrees with CPU native.

## Geometry contract

Mirror `CpuNativeSim::rhs_free`: spectrum expansion per spatial column, one
joint `(t,y,x)` c2r transform, pointwise Kerr, time window, one joint r2c
transform, spectral crop, and the transferred complex normalization. Julia's
column-major `(n_t,n_y,n_x)` maps to cuFFT dimensions in reversed order; the
halved dimension must be time.

## Implementation

1. Implement transactional CUDA `set_free_params` for dimensions, plans,
   buffers, window, Kerr coefficient, and normalization.
2. Add joint 3-D cuFFT plans with explicit normalization
   `1/(n_t*n_y*n_x)` and overflow checks.
3. Add pad/crop, Kerr/window, and final normalization kernels respecting
   column-major layout and non-square `n_y != n_x` grids.
4. Add a literal CUDA-vs-Julia transform reference using nonsymmetric data;
   a CUDA round trip alone cannot catch swapped axes.
5. Broaden eligibility only for RealGrid free-space scalar Kerr with constant
   norm/linop, no plasma/Raman/noise/mixture.
6. Keep `:auto` false.

## Acceptance

Test non-square transverse dimensions, direct stage/non-vacuous Kerr control,
fixed-step `<1e-6`, reject/retry, adaptive trajectory, invalid dimensions, and
transactional second setup. Run strict CUDA, existing CPU free-space/3-D FFT
tests, focused Julia item, Rust group, and `git diff --check`.

Update docs and append exact dimension/scaling evidence to `PORT_LOG.md`.

## Non-goals

EnvGrid, plasma, Raman, shot noise, z-dependent norm, or auto dispatch.

## Implementation complete

`CudaNativeSim` now stages a transactional `FreeSetup` with separate
`(n_time_over,n_y,n_x)` real scratch, `(n_time_over/2+1,n_y,n_x)` complex
scratch, Julia's transferred free-space normalization, and independent 3-D
cuFFT D2Z/Z2D plans. cuFFT receives `(n_x,n_y,n_time_over)`, which preserves
Julia's column-major `(t,y,x)` layout and makes time the halved dimension.
The resident RHS reuses the established column-major expand/window/crop
kernels around one inverse and one forward 3-D transform, with the explicit
`1/(n_time_over*n_y*n_x)` inverse normalization. Julia eligibility admits only
constant-linop/constant-norm RealGrid scalar Kerr and keeps CUDA `:auto` false.
The focused strict hardware test covers `n_y != n_x`, nonsymmetric spectral
data, non-vacuity, fixed/adaptive/rejected steps, invalid dimensions, and
transactional replacement setup.

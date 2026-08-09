# Luna feature plan 18 — CUDA free-space EnvGrid Kerr

Status: complete (2026-08-08). Depends on plan 17; standing CUDA CI remains
strongly preferred.

## Outcome

`TransFree` + EnvGrid + scalar Kerr runs resident on CUDA with joint complex
three-dimensional transforms and matches CPU native.

## Implementation

1. Add transactional joint c2c 3-D plans and complex buffers to plan 17's
   free-space setup.
2. Mirror `CpuNativeSim::rhs_free_env`, including low/high spectral expansion,
   explicit c2c scaling, envelope Kerr, time window, crop, and transferred
   normalization.
3. Preserve reversed cuFFT dimensions and test a non-square transverse grid
   with asymmetric complex values.
4. Broaden eligibility only for EnvGrid free-space Kerr under existing scalar
   density/constant norm/no-noise restrictions.
5. Keep EnvGrid plasma/Raman ineligible and `:auto` false.

## Spectral-half and resource contract

For each `(y,x)` series of even length `Nω`, expansion copies source
`[0,Nω/2)` to oversampled `[0,Nω/2)` and source `[Nω/2,Nω)` to
`[No-Nω/2,No)`, leaving the middle zero. The post-forward crop is the exact
inverse map: output `[0,Nω/2)` reads the low oversampled half and output
`[Nω/2,Nω)` reads `[No-Nω/2,No)`. The crop must never read the first `Nω`
oversampled entries contiguously when `No > Nω`.

The committed `free_fft_c2c` handle is owned by `CudaNativeSim` just like the
free-space D2Z/Z2D handles. Transactional replacement destroys the old handle,
staged failure destroys the staged handle through `FreeSetup`, and final
simulation teardown must destroy the currently committed c2c handle.

## Acceptance

Literal transform reference, direct stage/non-vacuity, fixed-step `<1e-6`,
reject/retry, adaptive trajectory, and transactional c2c setup on real CUDA.
The direct-stage checks must include a high-half-only, multi-column spectrum
with non-negligible amplitudes so both expansion and post-forward crop are
observable. Run strict CUDA, CPU free-space EnvGrid tests, plan-17 regressions,
Rust group, and `git diff --check`.

Update docs and append scaling/layout results to `PORT_LOG.md`.

## Non-goals

Plasma, Raman, noise, z-dependence, mixtures, or auto dispatch.

## Implementation complete

`CudaNativeSim` now extends Plan 17's transactional `FreeSetup` with full
complex buffers and one joint 3-D Z2Z cuFFT plan for EnvGrid. The reversed
dimensions `(n_x,n_y,n_time_over)` preserve Julia's column-major
`(n_time,n_y,n_x)` volume. The resident RHS preserves both low and high
spectral halves, applies the explicit `1/(n_time_over*n_y*n_x)` inverse
normalization, evaluates scalar `Kerr_env`, windows, forwards, crops, and
applies Julia's transferred normalization. Eligibility admits only constant
linop/constant-norm scalar free-space Kerr and keeps CUDA `:auto` false.
Strict hardware verification covers a non-square grid, asymmetric complex
spectra, non-vacuity, fixed/adaptive trajectories, rejected steps, and
transactional c2c setup replacement.

A 2026-08-09 review found two completion defects: the shared EnvGrid finalizer
read the first `Nω` oversampled bins contiguously, and final simulation teardown
did not destroy the committed free-space c2c plan. The finalizer now performs
the inverse two-half crop for every series, teardown destroys `free_fft_c2c`,
and a lifecycle test proves the handle is invalid after `CudaNativeSim` drops.
The new high-half-only multi-column stage check measured
`1.0958008920889427e-15` CUDA-vs-CPU relative error; the strict post-repair
Rust gate passed 43,455/43,455.

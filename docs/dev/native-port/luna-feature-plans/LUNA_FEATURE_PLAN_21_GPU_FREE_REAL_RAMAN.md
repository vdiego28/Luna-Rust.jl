# Luna feature plan 21 — CUDA free-space RealGrid SDO Raman

Status: complete (2026-08-09; strict CUDA hardware verified).

## Outcome

RealGrid free-space Kerr + supported SDO `RamanPolarField` runs resident on
CUDA, with one independent ADE series per transverse point.

## Implementation

1. Generalize the ADE launch to `n_series=n_y*n_x` contiguous time series and
   use plan 02's oscillator-capacity contract.
2. Allocate intensity/polarization/Hilbert scratch for the full oversampled
   volume using checked sizes.
3. Support `thg=true` through `E^2` and `thg=false` through batched Hilbert
   transforms along time only; spatial axes must not enter the Hilbert FFT.
4. Accumulate `pto += density*E*P` before the free-space time window and
   forward 3-D FFT.
5. Broaden eligibility only for RealGrid free-space SDO Raman, scalar density,
   constant norm, no plasma/noise/mixture. Keep `:SiO2` and EnvGrid excluded.
6. Keep `:auto` false.

## Acceptance

Use non-square transverse dimensions and distinct per-point signals to detect
axis or state leakage. Cover both THG values and N2 vibration plus rotation.
Require Julia Raman non-vacuity, direct stage agreement, fixed-step `<1e-6`,
reject/retry, and adaptive trajectory. Run strict CUDA, CPU free-space Raman,
mode-averaged Raman, 3-D FFT regressions, Rust group, and diff check.

 Update support docs and append oscillator/series/layout evidence to
`PORT_LOG.md`.

## Implementation contract (2026-08-09)

Plan 21 reuses the resident free-space RealGrid 3-D transform and the Plan
12/16 SDO kernels. The flattened free-space volume has
`n_series=n_y*n_x`, series `s=iy+n_y*ix`, and contiguous sample
`j=s*n_time_over+i`; every intensity, Hilbert, ADE, and accumulation stage
must preserve that boundary. The shared Raman setter therefore sizes its
scratch and batched c2c Hilbert plan for `n_time_over*n_series` and must stage
all allocations before replacing the prior configuration.

For `RamanPolarField(thg=true)`, the pointwise intensity is `I=E^2`. For
`thg=false`, each spatial series is transformed independently along time only;
the local analytic-signal mask keeps DC/Nyquist single-weighted, doubles the
positive temporal half, and zeros negative temporal frequencies before the
inverse c2c transform, giving `I=0.5*abs2(hilbert(E))`. The existing
`raman_ade_kernel` then runs one oscillator state vector per spatial series
with Julia's flattened SDO coefficients and the existing `PrecomputedStepCoeffs`.

The free-space RealGrid RHS order is:

```text
inverse joint 3-D FFT → volume scale → scalar Kerr
→ optional Raman intensity/Hilbert/ADE → Pto += density*E*P
→ time window → forward joint 3-D FFT → crop/scale/normalization
```

Eligibility admits exactly one plain scalar Kerr plus one
`RamanPolarField`/flattened `CombinedRamanResponse`, scalar density, constant
norm, no plasma/noise/mixture, and explicit CUDA dispatch only. EnvGrid Raman,
`:SiO2`, z-dependent free-space norm/linop, and `:auto` remain rejected.

## Non-goals

EnvGrid Raman, intermediate broadening, plasma composition, z-dependent norm,
or auto dispatch.

## Completion record (2026-08-09)

Implemented with the existing free-space joint 3-D transform and resident
Raman kernels. `CudaNativeSim::set_raman_params` now sizes the shared scratch
and batched c2c Hilbert plan for `n_y*n_x` free-space series; the free RealGrid
RHS applies Kerr, optional batched Raman, the existing time window, and the
forward transform in the documented order. Julia eligibility admits only one
plain Kerr plus one scalar `RamanPolarField` with a flattenable 1–64 oscillator
response; plasma+Raman, EnvGrid Raman, noise, mixtures, z-dependent norm/linop,
and `:auto` remain rejected.

The strict focused test passed **44/44** on the RTX 5060 Ti with CUDA 13.3.
Non-square `10×8` point-series checks covered N₂ vibration, rotation, and
rotation+vibration with both `thg=true` and `thg=false`; direct CUDA-vs-CPU
stage errors were `1.28e-15`–`1.35e-15`, fixed-solve errors were
`2.62e-16` and `2.68e-16`, and Julia Raman-on/off effects were `1.176e-3`
and `1.181e-3`. The same item covered unsupported-response/mixed-plasma
rejection and a rejected adaptive step with state preservation. The full
strict Rust-group rerun passed **43,445/43,445** in **16m37.2s**, and the
timing manifest includes `test_native_cuda_free_raman.jl` at 132.1 seconds.

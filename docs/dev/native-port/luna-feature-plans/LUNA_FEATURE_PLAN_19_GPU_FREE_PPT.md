# Luna feature plan 19 — CUDA free-space RealGrid PPT plasma

Status: complete 2026-08-08.

## Outcome

RealGrid free-space Kerr + PPT plasma runs resident on CUDA, with an
independent deterministic plasma scan for every `(y,x)` time series.

## Implementation

1. Generalize the completed prefix-scan pipeline to
   `n_series=n_y*n_x`, preserving contiguous time-series boundaries.
2. Allocate rate/fraction/current/polarization/scan scratch for the full
   oversampled volume with checked arithmetic.
3. Reuse the exact PPT spline upload and current formulas; accumulate plasma
   before the free-space time window and forward 3-D FFT.
4. Make free setup precede plasma setup and make failed plasma replacement
   transactional.
5. Broaden eligibility only for RealGrid free-space, one Kerr + one PPT,
   scalar density, constant norm, no Raman/noise/mixture.
6. Keep `:auto` false.

## Acceptance

Primitive scan tests must cover multiple spatial series, multiple blocks, a
partial last block, and sentinels proving no cross-series carry. Require a
non-vacuous Julia plasma effect, direct stage comparison, fixed-step `<1e-6`,
rejected-state parity/retry, and adaptive agreement. Run strict CUDA, CPU
free-space plasma, existing scan/PPT tests, Rust group, and diff check.

Update docs and append scan shape/tolerances to `PORT_LOG.md`.

## Non-goals

ADK, EnvGrid plasma, Raman, z-dependent norm combinations, or auto dispatch.

## Implementation contract (2026-08-08)

The free-space RealGrid layout is column-major in Julia: the flattened device
index for spatial series `s = iy + n_y*ix` and oversampled time sample `i` is
`j = s*n_time_over + i`. The plasma scan is therefore segmented by `s`; it
must never use a prefix value from `s-1`, even when a series spans multiple
256-thread blocks or its last block is partial. The block-total buffer is
`[s*n_blocks + b]`, where `n_blocks=ceil(n_time_over/256)`. A finalizer
reconstructs the preceding-block offset by summing only the block totals for
the same `s`.

For each series, the transferred PPT rate `R_i = R(|E_i|)` is integrated with
the same CPU trapezoid convention:

```
A_0 = 0
A_i = A_{i-1} + 0.5*(R_{i-1}+R_i)*dt
F_i = preionfrac + 1 - exp(-A_i)
H_i = F_i * e_ratio * E_i
J_i = sum_trap(H, dt)_i + ionpot*R_i*(1-F_i)/E_i   (E_i != 0)
P_i = sum_trap(J, dt)_i
Pto_i += density * P_i
```

The zero-field branch of the current add-in is exactly zero. This ordering
matches `CpuNativeSim::apply_plasma_free`: Kerr first, then plasma
polarization, then `towin`, then the one joint `(n_x,n_y,n_time_over)` forward
transform and the transferred `M` normalization. The c2c EnvGrid path remains
untouched and ineligible.

The CUDA setup reuses the PPT spline representation and stages all five
plasma buffers plus `n_series*n_blocks` scan totals before assigning them to
the live state. A failed allocation, copy, null handle, or invalid parameter
must leave the previously committed free-space configuration usable. The
focused test uses `n_y != n_x`, `n_time_over > 512`, a multi-series sentinel,
and a strong independent Julia field to demonstrate both scan isolation and a
nonzero plasma contribution; direct stage, fixed-step, rejection, and
adaptive comparisons use the Julia `PreconStepper` as oracle.

## Implementation complete (2026-08-08)

`amalthea/src/cuda_native.rs` now sizes PPT scratch and scan totals for
`n_series=n_y*n_x`, uses the generalized `plasma_scan_series` pipeline, and
calls `apply_plasma_series_real` between free-space Kerr and the time window.
The CUDA kernels in `amalthea/src/kernels.cu` use `(series,block)`-indexed
totals and series-local offsets; the same kernels continue to serve radial
plasma. `src/RK45.jl` admits exactly one plain Kerr plus one
`IonRatePPTAccel`/`PlasmaCumtrapz` for RealGrid free space, while EnvGrid
plasma and free-space `:auto` remain rejected. `native_set_plasma_params`
stages all replacement buffers before committing them, and free-space ADK is
explicitly rejected as a non-goal.

The strict focused test passed 28/28 on CUDA 13.3 with a non-square `10×8`
grid and `n_time_over=8192`: direct and asymmetric-complex stage errors were
`1.2918835724298099e-15` and `1.2633763496880677e-15`; the strong-field Julia
plasma effect was `1.5696720458555424e-6`; GPU-vs-Julia strong solve error was
`6.537665790889942e-16`; fixed and adaptive CPU/GPU trajectory errors were
`4.960731457415347e-16` and `1.3151815943992969e-14`. The CUDA unit scan
covers three independent series, two full blocks plus a partial block, and
zero-series-boundary sentinels.

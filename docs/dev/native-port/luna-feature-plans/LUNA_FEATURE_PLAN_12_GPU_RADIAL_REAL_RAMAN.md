# Luna feature plan 12 — CUDA radial RealGrid SDO Raman

Status: complete 2026-08-04; standing CUDA CI remains strongly preferred.

## Outcome

RealGrid radial Kerr + one supported `RamanPolarField` runs resident on CUDA
for both `thg=true` and `thg=false`, with one independent ADE series per radial
column.

## Implementation

1. Generalize the CUDA ADE launch from one mode-averaged series to `n_r`
   independent contiguous time series. Use plan 02's shared capacity contract.
2. Allocate intensity and Raman polarization for `n_time_over*n_r`.
3. For `thg=true`, form `E^2` per cell. For `thg=false`, use batched c2c
   Hilbert transforms per radial column and the exact parity mask/scaling.
4. Accumulate `pto += density*eto*P` before the radial time window/QDHT.
5. Ensure oscillator state and Hilbert scratch cannot leak between columns or
   RK stages; no host arrays transfer per RHS.
6. Broaden eligibility only for RealGrid radial matching SDO Raman, scalar
   density, constant linop/norm, no plasma/noise/mixture. Keep `:SiO2` out.
7. Keep `:auto` false.

## Implementation contract

The radial RealGrid buffers use the existing column-major resident layout:
`offset = radial_column*n_time_over + time_index`.  The CUDA Raman intensity,
polarization, Hilbert scratch, and ADE launch therefore use one contiguous
`n_time_over*n_r` allocation.  `raman_ade_kernel` receives `n_series=n_r`; its
one CUDA thread per series owns fresh oscillator states, so no state is shared
between radial columns or RHS stages.  For `thg=false`, `cufftPlan1d` is used
with `batch=n_r` over those contiguous columns, followed by the existing parity
filter and `1/n_time_over` inverse scale.  Mode-averaged setup keeps
`n_series=1` and its current allocation sizes.  Raman is evaluated after Kerr
and any supported radial plasma contribution, before the radial time window and
QDHT multiplication; no host transfer is introduced in an RHS call.

The setter stages allocations and the batched Hilbert plan using the current
radial geometry (`n_r` when radial, otherwise one), then commits them together.
Eligibility is limited to RealGrid radial `RamanPolarField` SDO responses with
one scalar density and otherwise the existing Plan 08 constraints.  `:auto`
continues to reject every radial configuration.

## Acceptance

Test at least two radial columns with distinct signals, vibration-only and N2
rotational Raman, both THG values, and a column-isolation sentinel. Require a
non-vacuous Julia Raman control, direct stage comparison, fixed-step `<1e-6`,
reject/retry, and adaptive agreement. Run strict CUDA, CPU radial Raman,
mode-averaged Raman, Rust group, and diff check.

Update docs and append oscillator/series counts and errors to `PORT_LOG.md`.

## Non-goals

EnvGrid, `:SiO2`, plasma composition, mixtures, z-dependence, or auto dispatch.

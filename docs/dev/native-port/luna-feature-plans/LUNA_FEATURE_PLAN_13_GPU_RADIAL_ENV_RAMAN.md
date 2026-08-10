# Luna feature plan 13 — CUDA radial EnvGrid SDO Raman

Status: complete 2026-08-04; standing CUDA CI remains strongly preferred.

## Outcome

EnvGrid radial Kerr + `RamanPolarEnv` runs resident on CUDA with one ADE series
per radial column.

## Implementation

1. Reuse plan 12's segmented ADE storage/launch and plan 09's complex radial
   buffers.
2. Form real intensity `0.5*abs2(E)` for every `(t,r)` cell; no Hilbert branch.
3. Accumulate complex `pto += E*(density*P)` before window/QDHT/forward c2c.
4. Broaden eligibility only for matching EnvGrid radial SDO Raman. Keep
   intermediate broadening, plasma, noise, mixtures, and z-dependence out.
5. Keep `:auto` false.

## Implementation contract

The CUDA radial EnvGrid buffers remain column-major with one contiguous complex
time series per radial column: `offset = radial_column*n_time_over + time`.
The EnvGrid Raman branch uses the existing `raman_intensity_env_kernel` to
form `0.5*abs2(E)` over the flattened `(time, radial-column)` buffer, launches
the existing `raman_ade_kernel` with `n_series=n_r`, and uses
`raman_accumulate_env_kernel` to add `density*E*P` to the complex polarization.
There is no Hilbert transform or carrier `thg` branch for `RamanPolarEnv`.
The branch runs after radial Kerr and before the shared time window, QDHT, and
forward c2c transform; all buffers remain resident on the device.

Eligibility admits exactly one scalar-density EnvGrid `RamanPolarEnv` whose
response is a non-empty `CombinedRamanResponse` flattening to at most 64 SDO
oscillators, paired with one plain Kerr response.  EnvGrid `:SiO2`
intermediate broadening, plasma, noise, mixtures, and `:auto` dispatch remain
unsupported.

## Acceptance

Use distinct complex signals in at least two radial columns and assert series
isolation. Prove a non-vacuous Julia Raman effect, compare direct stages, run a
fixed-step trajectory `<1e-6`, and exercise rejection/retry/adaptive behavior.
Run strict CUDA, focused EnvGrid radial Raman, CPU radial EnvGrid Raman,
mode-averaged Raman regressions, Rust group, and `git diff --check`.

Update support docs and append `PORT_LOG.md` with achieved errors.

## Non-goals

`:SiO2`, EnvGrid plasma, plasma composition, auto dispatch, or new Raman math.

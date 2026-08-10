# Luna feature plan 20 — CUDA free-space RealGrid thresholded ADK

Status: depends on plan 19.

## Outcome

RealGrid free-space Kerr + thresholded ADK plasma reuses plan 19's segmented
volume scans on CUDA and matches CPU native.

## Implementation

1. Launch the existing pointwise ADK rate over all `(t,y,x)` cells with exact
   threshold/non-finite semantics.
2. Feed plan 19's per-spatial-series fraction/current/polarization scans.
3. Broaden eligibility only for `IonRateADK(threshold=true)` in the supported
   free-space RealGrid shape. Keep `threshold=false` CPU-only.
4. Add multi-series rate-boundary and setup-rollback tests.
5. Keep `:auto` false pending geometry-specific performance evidence.

## Implementation contract (2026-08-08)

Plan 20 extends Plan 19's already validated free-space RealGrid pipeline without
adding a second scan implementation. The flattened volume is
`n_series=n_y*n_x`, with series `s=iy+n_y*ix` and sample
`j=s*n_time_over+i`. Every cumulative scan therefore resets at each transverse
column; raw block totals are stored at `[s*n_blocks+b]`, and each series
finalizer sums only blocks `0:b-1` for that same `s`.

For thresholded ADK, the pointwise rate kernel receives Julia's seven
precomputed constants `(occupancy, omega_p, cn_sq, nstar,
omega_t_prefac, thr, avfac)` and must reproduce `IonRateADK(E)` exactly:

```text
a = abs(E)
R = 0                                      if !isfinite(a) or a < thr
x = 4*omega_p/(omega_t_prefac*a)
R = occupancy*omega_p*cn_sq*x^(2*nstar-1)
    * exp(-(4/3)*omega_p/(omega_t_prefac*a))
R *= avfac*sqrt(a)                         if avfac != 1
```

The subsequent shared series stages are unchanged from Plan 19:

```text
A[0] = 0
A[i] = A[i-1] + 0.5*(R[i-1] + R[i])*dt
F[i] = preionfrac + 1 - exp(-A[i])
H[i] = F[i]*e_ratio*E[i]
J[i] = J[i-1] + 0.5*(H[i-1] + H[i])*dt
       + ionpot*R[i]*(1-F[i])/E[i]   when E[i] != 0
P[i] = P[i-1] + 0.5*(J[i-1] + J[i])*dt
Pto[i] += density*P[i]
```

The CUDA free-space RHS applies this ADK branch after scalar Kerr and before
the common time window and joint forward 3-D transform. `native_set_free_params`
must complete before `native_set_plasma_params_adk`; the ADK setter stages all
free-space scratch allocations before replacing the live buffers, so null,
non-finite, invalid-threshold, and allocation failures preserve the prior
working setup. Only `IonRateADK(threshold=true)` is eligible; unthresholded ADK,
EnvGrid plasma, z-dependent combinations, Raman/noise mixtures, and `:auto`
remain rejected.

## Acceptance

Assert a Julia ADK effect at least 100× tolerance, direct stage agreement,
fixed-step `<1e-6`, rejected-state bit parity/retry, adaptive trajectory, and
no cross-series contamination. Run strict CUDA, mode-averaged ADK regressions,
CPU free-space plasma tests, Rust group, and `git diff --check`.

Update docs and append exact rate/trajectory evidence to `PORT_LOG.md`.

## Implementation complete (2026-08-08)

`amalthea/src/cuda_native.rs` now dispatches the existing free-space segmented
series scan through the exact thresholded-ADK pointwise rate, with
transactional free-space scratch replacement. `src/RK45.jl` admits only
thresholded ADK for the explicit free-space RealGrid path; unthresholded ADK,
EnvGrid plasma, and free-space `:auto` remain rejected. The strict focused CUDA
test passed 43/43 on the non-square `10×8` fixture: direct stage errors were
`1.219997607646526e-15` and `1.290476856284764e-15`, the Julia ADK effect was
`0.0026768995301431862`, strong native-vs-Julia error was
`6.704619060731584e-16`, fixed-solve error was `4.771968773563592e-16`, and
adaptive-solve error was `1.348203925172025e-16`. CPU free-space controls passed
15/15; strict Rust/CUDA unit validation passed 80/80 Rust tests, 3/3
build-policy tests, and docs. The full Rust-group run reached 43,397 passes
and one manifest-coverage failure because the new timing entry was initially
absent; the entry is now recorded, but the rerun is intentionally pending.

## Non-goals

Unthresholded ADK, EnvGrid plasma, z-dependent combinations, or auto dispatch.

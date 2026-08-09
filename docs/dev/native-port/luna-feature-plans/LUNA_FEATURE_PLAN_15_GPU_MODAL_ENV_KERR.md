# Luna feature plan 15 — CUDA modal EnvGrid Kerr

Status: complete — hardware-verified 2026-08-08.

## Outcome

Eligible EnvGrid modal Kerr configurations run through plan 14's CUDA modal
point-evaluation pipeline for `full=false|true` and `npol=1|2`.

## Implementation

1. Add transactional c2c modal plans and complex time/spectral scratch while
   preserving plan 14's cubature batch protocol.
2. Mirror the CPU modal EnvGrid spectrum expansion/crop and explicit c2c
   scaling; use `0.75*|E|^2E` scalar/vector envelope Kerr as appropriate.
3. Extend device field synthesis and projection to complex envelope data and
   retain mode/polarization ordering exactly.
4. Broaden eligibility only for supported EnvGrid modal Kerr. Keep Raman,
   plasma, noise, mixtures, and `StepIndexMode` excluded.
5. Keep `:auto` false.

## Mathematical and layout contract

For each cubature node `(r, θ)`, mode synthesis is identical to Plan 14:

```
Er_ω,p = Σ_m Em_ω,m · J_qm(unm_m*r/a) / sqrt(N_m) · A_m,p(θ)
```

The mode metadata, polarization selectors, Jacobian (`2πr` for `full=false`,
`r` for `full=true`), and point-major packed real/imaginary callback output are
unchanged. Only the temporal/frequency representation changes. If `Nω` is the
normal EnvGrid spectral length and `No` is the oversampled time length, with
`Noω = No`, each polarization column is expanded as

```
Eo[i]       = (No/Nω) * E[i]                         i < Nω/2
Eo[No-Nω/2+i] = (No/Nω) * E[Nω/2+i]                  0 ≤ i < Nω/2
Eo[remaining] = 0
```

The resident c2c inverse transform is unnormalised, so its result is scaled by
`1/No` before the envelope Kerr response. The response must be the exact Julia
`Kerr_env` formula, including its SVEA factor:

```
P = 0.75*kerr_fac*|E|²*E                              npol=1
Px = 0.75*kerr_fac*((|Ex|² + 2|Ey|²/3)*Ex
                     + conj(Ex)*Ey²/3)
Py = 0.75*kerr_fac*((|Ey|² + 2|Ex|²/3)*Ey
                     + conj(Ey)*Ex²/3)                 npol=2
```

After `P *= towin`, the forward c2c transform is cropped back with
`Nω/No`, again using the low and high halves above. Each retained spectral bin
is multiplied by the transferred `nlfac = ωwin .* norm_modal(grid)` and then
projected back onto every mode. All buffers are column-major by polarization
series (`series = node*npol + pol`) with the temporal/spectral index fastest;
this is the same layout as Plan 14 and as `native.rs::modal_pointcalc`.

The c2c plan is batched over `npol*batch_capacity` independent node/polarization
series. It must not use the RealGrid r2c/c2r plan or the RealGrid `E³` formula;
doing either would silently discard the negative-frequency envelope half or
apply the wrong vector-polarization physics. No host transfer occurs for the
resident field or scratch; only node coordinates and packed cubature output
cross the boundary.

### Review correction — oversampled synthesis

The shared modal synthesis launch must distinguish the RealGrid contiguous
half-spectrum from the EnvGrid two-half layout. For EnvGrid, destination index
`i` reads source `i` only for `i < Nω/2`; destination
`No-Nω/2+i` reads source `Nω/2+i`; every middle destination is zero. In
particular, it must not copy all `Nω` source bins contiguously before the c2c
inverse. Regression coverage must use a high-half-only spectrum whose retained
bins have the same scale as the low half, rather than merely phase-perturbing a
physical pulse with negligible edge content.

## Verification rationale

The Julia `PreconStepper` is the primary oracle and the CPU resident
`RustNativeStepper` with `AMALTHEA_NATIVE_GPU=off` is the implementation control.
The focused CUDA item first compares supplied asymmetric nodes and direct FSAL
stage derivatives, then compares a fixed-step trajectory. It covers one and
two modes, both cubature branches, and `npol=1|2`; the vector case uses a
nontrivial `ϕ` and complex envelope spectrum so both the conjugate cross term
and negative-frequency c2c copy are exercised. A Kerr-on/Kerr-off Julia control
must change the final field by more than the trajectory tolerance, and a hot
adaptive trial must be rejected with the GPU field bit-exactly unchanged before
the retry. The expected direct comparison is the modal method tier (target
`<5e-8` for the CUDA-vs-CPU stage, allowing device/libm reassociation), while
the fixed-step trajectory target is `<1e-6`; measured results, exact commands,
and any hardware limitation belong in the completion section and PORT_LOG.

## Resource and dispatch decisions

The transactional `native_set_modal_params` replacement remains the only setup
seam. For EnvGrid it stages c2c inverse/forward plans and complex time and
polarization buffers; a failed allocation, copy, plan, or cubature load leaves
the previous modal setup untouched. The existing batch capacity of 32 is kept,
and modal `:auto` remains disabled because Plan 14’s callback traffic is a
correctness measurement rather than a production-shaped performance threshold.
The CPU modal EnvGrid path remains unchanged and is the explicit fallback.

## Acceptance

Test supplied-node point evaluation, one/two modes, both cubature modes, and a
non-vacuous two-polarization case with asymmetric complex data. Require direct
stage agreement, nonzero modal transfer, fixed-step `<1e-6`, reject/retry, and
adaptive agreement. Also require a high-half-only direct-stage comparison that
fails if the upper half is copied into the oversampled middle. Run strict CUDA,
CPU modal EnvGrid tests, plan-14
regressions, Rust group, and `git diff --check`.

Update support docs and append exact c2c scaling/tolerance evidence to
`PORT_LOG.md`.

## Completion record

Implemented in `amalthea/src/cuda_native.rs`, `amalthea/src/cuda.rs`, and
`amalthea/src/kernels.cu`; Julia eligibility is in `src/RK45.jl`, and the
focused regression is `test/test_native_cuda_modal_env.jl`. The existing
transactional modal setup now selects batched Z2Z/Z2Z cuFFT plans and complex
scratch for EnvGrid, while the RealGrid r2c/c2r path is unchanged. No new FFI
ABI was needed: the existing `native_set_modal_params`,
`native_set_fftw_plans`, and modal debug/statistics exports are reused.

Strict verification used the host CUDA 13.3 toolkit and RTX 5060 Ti driver
610.43.02. The focused Plan 15 item passed 35/35: fixed-node point errors were
`4.82e-16`–`6.12e-16`, direct-stage errors were `3.07e-16`–`3.27e-16`, and
the fixed 50-step solve error was `5.97e-16`. The two-mode HE11→HE12 transfer
was `8.41e-6`, the Julia Kerr-on/off control was `2.52e-2`, and the adaptive
solve error was `7.02e-17`; the rejected hot trial preserved the field and
retry state. The CPU modal EnvGrid controls passed at `1.07e-17`–`1.12e-17`.
The strict Plan 14 + Plan 15 focused run passed 72/72, and the complete strict
`LUNA_TEST_GROUP=rust` gate passed 43,227/43,227.

A 2026-08-09 review found that the shared synthesis kernel still copied the
upper EnvGrid half contiguously into the oversampled middle. The corrected
kernel now receives the grid representation and applies the two-half map above;
the new high-half-only direct-stage regression measured
`1.0905464182781277e-15` CUDA-vs-CPU relative error. The strict post-repair
Rust gate passed 43,455/43,455.

The landed scope remains explicit CUDA-on only for constant-radius
Marcatili/Zeisberger/Vincetti modal scalar Kerr, both cubature branches, and
`npol=1|2`; modal `:auto`, Raman, plasma, noise, mixtures, tapered radius, and
free-space remain outside this plan.

## Non-goals

Raman, plasma, shot noise, mixtures, z-dependence expansion, or auto dispatch.

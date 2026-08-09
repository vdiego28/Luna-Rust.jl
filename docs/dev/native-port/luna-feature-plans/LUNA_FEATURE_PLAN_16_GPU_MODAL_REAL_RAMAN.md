# Luna feature plan 16 — CUDA modal RealGrid scalar SDO Raman

Status: complete (2026-08-08). Depends on plans 02 and 14.

## Outcome

RealGrid modal `npol=1` Kerr + supported `RamanPolarField` runs on CUDA inside
each cubature-node evaluation and agrees with CPU native.

## Contract and layout

The modal callback batch is the unit of ownership. `modal_field_time_d` and
`modal_polarization_d` use the existing layout
`series = node*npol + polarization`; Plan 16 admits only `npol=1`, so each
accepted cubature node owns one contiguous series of `n_time_over` samples.
The staged modal allocation keeps a fixed `batch_capacity=32`, while a
callback may use any `1 <= count <= batch_capacity`. Raman intensity, Raman
ADE polarization, and the two complex Hilbert work arrays are allocated for
every capacity series, never as one buffer shared between nodes.

For each RealGrid node, after the existing inverse D2Z and normalization, the
device computes the same sequence as the CPU modal point RHS:

1. `P = P_Kerr(E)`;
2. for `thg=true`, `I = E^2`; for `thg=false`, form the analytic signal with
   the existing Hilbert FFT/filter kernels and use `I = |E_a|^2/2`;
3. reset and integrate the supported SDO oscillators for this RHS,
   `P_R = ADE(I)`, using the Plan 02 oscillator coefficients;
4. add `density*E*P_R` to `P`;
5. apply the existing time window and forward/project pipeline.

The Hilbert plan is `Z2Z`, length `n_time_over`, and batch `batch_capacity`.
Only the first `count` transforms and outputs are consumed; the fixed plan
prevents plan creation or scratch resizing inside a callback. Raman ADE launch
uses one CUDA block per node series, so no node can observe another node's
state. The state is deliberately reset on every RHS, matching CPU native
semantics rather than introducing propagation-time oscillator memory.

## Implementation

1. Reuse plan 14's device point batches and plan 02's oscillator capacity.
2. Allocate per-node/per-series Raman intensity, ADE polarization, and Hilbert
   scratch. Each cubature node is independent; state reset each RHS exactly
   CPU.
3. Support both THG values: direct `E^2` or the exact batched Hilbert analytic
   signal. Accumulate Raman additively before time window/projection.
4. Prevent scratch reuse races when callback batches evaluate several nodes.
5. Broaden eligibility only for RealGrid modal `npol=1` supported SDO Raman.
   Keep `npol=2`, EnvGrid Raman, `:SiO2`, plasma, noise, and mixtures
   excluded.
6. Keep `:auto` false.

The existing `native_set_raman_params` FFI remains the single configuration
entry point. It is called after `native_set_modal_params`, so the native setter
branches on the already-committed modal shape and stages modal Raman buffers
and its Hilbert plan separately from the resident radial/mode-averaged
buffers. On allocation or plan failure the old configuration remains
installed. The shared coefficient buffer and scalar Raman metadata retain the
existing FFI ownership and oscillator-capacity checks (`1..=64`).

## Acceptance

Use multiple cubature nodes with distinct fields and N2 vibrational plus
rotational cases. Assert Julia Raman-on/off non-vacuity, direct point/stage
agreement, fixed-step `<1e-6`, rejected-state preservation/retry, and
adaptive trajectory. Run strict CUDA, CPU modal Raman/threading tests,
mode-averaged Raman regressions, Rust group, and `git diff --check`.

Verification also requires a shape-level dispatch test: RealGrid modal
`npol=1` with one plain Kerr and one flattenable SDO `RamanPolarField` is
eligible under `gpu_dispatch=:on`, while EnvGrid Raman, modal `npol=2`
Raman, unsupported Raman response forms, plasma/noise, and Kerr mixtures
remain rejected. `gpu_dispatch=:auto` must continue selecting CPU native.

Update docs and append series ownership, oscillator counts, and tolerances to
`PORT_LOG.md`.

Implementation and hardware acceptance are complete. The focused strict CUDA
item passed 28/28; both oscillator families use the same per-node ownership
contract (vibration: 1 oscillator, rotation: 49), and both THG branches are
covered. CPU modal/threading, mode-averaged CUDA Raman, strict Rust-group,
manifest, and diff checks are green.

## Non-goals

Modal `npol=2` Raman, EnvGrid Raman, intermediate broadening, plasma, or auto
dispatch.

# Resident CPU Performance Audit

> Status: exhaustive campaign paused by the lead on 2026-08-24 after the
> completed baseline, correctness, adaptive, small/medium component, and large
> setup/field-sync evidence. The action-oriented handoff is
> [`PERFORMANCE_AUDIT_PRELIMINARY_RESULTS.md`](PERFORMANCE_AUDIT_PRELIMINARY_RESULTS.md).
> Root-cause checkpoints remain; no production optimization is accepted yet.

## Frozen baseline

The audit compares three process-isolated implementations:

1. Amalthea installed-default portable resident CPU backend at
   `73e32dcf45d93f11136d419faeae3b3641c9577d`.
2. The retained Julia oracle at the same Amalthea commit and manifest.
3. Upstream Luna.jl at
   `0a52ffbba6d5dd6820bb3dc3c300b8b38d724214` for configurations proven
   publicly equivalent.

The acceptance Rust artifact is built with release mode, CPU-only features,
and `RUSTFLAGS=''` through `deps/build.jl`. This neutralizes the checkout's
manual-build `target-cpu=native` setting and represents the portable installed
package contract. Host-native and LTO/codegen variants remain diagnostics.

Machine-readable host, dependency, toolchain, and artifact metadata are in
`test/performance_audit/results/baseline.json`. The initial host is an AMD Ryzen
5 5600X (Zen 3, 6 physical/12 logical cores), microcode `0xa20102e`, Linux
7.0.0-28, Julia 1.12.6, Rust 1.95.0, FFTW 3.3.11, and OpenBLAS ILP64. The
captured governor is `powersave`, AMD P-state is active, and boost is enabled;
these states are part of this baseline rather than unrecorded noise. Primary
measurements will use one pinned physical core and one Julia/FFTW/BLAS/OMP
thread. Thread scaling uses 1, 2, 4, and 6 physical cores.

## Exhaustive workload inventory

`test/performance_audit/workloads.toml` contains the resident CPU branch
inventory. It covers all four geometries and both grids, the PPT and ADK rate
branches, SDO/rotational/intermediate-broadening Raman, THG/Hilbert paths,
polarization counts, modal integration representations and mode families,
shot noise, gas mixtures, and supported z dependence. Each control-flow class
has small, medium, and large shapes; orthogonal integration/output/thread
variants are declared once rather than multiplied into redundant fixtures.

The inventory is derived from `src/RK45.jl` guard/setter code and cross-checked
against `NATIVE_SUPPORT_MATRIX.md` and native tests. This found and corrected
stale derived documentation: current source explicitly implements modal and
free-space plain-Kerr mixtures, and its radial mixture branch is not restricted
to RealGrid. Executable small fixtures for all eight geometry/grid mixture
cells constructed the CPU backend and passed single-step, fixed-solve, and
non-vacuity gates. Independent upstream equivalence is still a separate probe.

## Results

The one-thread medium adaptive matrix converged for all 36 fixtures that
passed the original fixed-step gate. One subsequently discovered adaptive-only
failure, `free_real_zdependent`, is excluded from accepted aggregate claims:
its Rust/Julia field difference is reproducibly `1.45314e-2` despite both
backends accepting four steps with no rejection. The remaining 35 fixtures
have a Julia/native geometric-mean speed ratio of `1.1091x`. Geometry-level
ratios are `1.2180x` for mode-averaged, `1.0047x` for radial, `1.5408x` for the
two correctness-admitted modal Raman cells, and `0.9933x` for free-space.
Fifteen individual fixtures regress by more than 5%, so the aggregate is not
evidence that the backend is uniformly faster.

The large adaptive sweep collected ten observations for every admitted cell
and extended noisy cells up to 30. It found two additional adaptive-only modal
Raman failures. Julia takes 9 accepted steps while Rust takes 10 in both;
`modal_real_raman_thg` differs by `2.04562e-6` against a `1e-6` tier and
`modal_real_raman_nothg` differs by `1.91107e-6` against `1.5e-6`. Two
correctness-valid Julia radial-plasma cells reached the 30-sample cap without
meeting the 3% MAD gate: PPT is 5.11% MAD/6.87% CI half-width and ADK is 4.28%
MAD/2.40% CI half-width. Those four fixture pairs remain diagnostic and are
excluded from accepted aggregate claims.

The accepted medium+large adaptive analysis contains 66 fixture-size pairs,
passes 66/66 post-timing field checks, and meets both stability gates in every
included backend cell. Its geometric-mean Julia/native ratio is `1.11913x`,
below the `1.20x` target. By geometry it is `1.28429x` for mode-averaged,
`0.97223x` for radial, `1.00760x` for free-space, and `1.54077x` for the two
correctness-admitted medium modal Raman cells (both large modal Raman cells are
excluded). Twenty-seven of the 66 pairs regress by more than 5%. The large-only
accepted set has 31 pairs, `1.13060x` geometric mean, and 12 regressions.

Seven medium component matrices are now correctness- and stability-accepted.
The ratios below are geometric means of Julia time divided by Rust time; a
value below one means the Rust component is slower:

| Component | Accepted fixture pairs | Julia/Rust ratio | Rust regressions >5% |
|---|---:|---:|---:|
| Setup | 34 | `0.90942x` | 15 |
| Field synchronization | 36 | `0.68219x` | 26 |
| Fixed RHS | 36 | `1.19741x` | 11 |
| One complete fixed step | 36 | `1.00658x` | 20 |
| Complete fixed solve | 35 | `1.38159x` | 6 |
| Dense output | 35 | `1.50780x` | 6 |
| Preallocated result copy | 32 | `0.99773x` | 1 |

The component split localizes two broad costs already: Rust construction and
explicit Julia/native field synchronization are slower, while fixed-solve and
dense-output work are faster in aggregate. The near-parity complete-step ratio
and its 20 regressions show that branch-level RHS/stage costs remain the main
end-to-end issue. The fixed-RHS aggregate is close to the `1.20x` target but
contains pronounced radial, rotational-Raman, and shot-noise regressions; the
final report will attribute their stage percentages with profiles rather than
infer causes from ratios alone.

Two Rust setup cells (`free_real_kerr`, `free_real_mixture`) exhausted 30
samples above the 3% MAD gate and are excluded from the accepted setup
aggregate. The first field-sync pass batched only 20 calls and left 13/72
backend cells unresolved; a calibrated 20-ms pass converged all 72 cells in
10--17 samples. Fixed-RHS synthetic repetition causes 11 final-batch field
drifts in stateful Raman scratch, so correctness is gated by the frozen strict
fresh single-step result while retaining the repeated-batch error as a
diagnostic. Fixed-solve and dense-output exclude `free_real_zdependent`
because its known interpolation seam fails even though raw fixed terminal state
passes. Dense-output needed up to 28 observations for convergence.

Component microbenchmarks calibrate per-request repetitions to about 20 ms.
Complete-step latency uses exactly one step per observation: an aborted
five-step medium trial showed individual step costs from 0.24 ms to 1.27 s,
making a fixed batch an avoidable fivefold multiplier. An analogous aborted
200-call dense-output trial was replaced after a modal interpolation measured
about 7 ms per call. Both superseded raw trials are preserved as diagnostic
directories rather than included in accepted summaries.

Result-copy timing needed two additional protocol corrections. Copying
`stepper.yn` after an adaptive solve was invalid because that is an internal
right-hand proposal/synchronization buffer, not the requested terminal result.
Timing allocating `copy(result)` then left 28/70 cells unstable through 30
samples because GC dominated. The accepted probe materializes the valid
terminal interpolant outside timing and measures `copyto!` into a preallocated
destination, matching `solve(...; output=true)` assignment. It reduced capped
instability to five cells in three free-space fixture pairs. Those pairs and
`free_real_zdependent` are excluded from the accepted 32-pair aggregate; both
superseded attempts and the full correctness-admissible preallocated matrix are
retained as diagnostics.

The complete small component split is also accepted:

| Component | Accepted fixture pairs | Julia/Rust ratio | Rust regressions >5% |
|---|---:|---:|---:|
| Adaptive solve | 45 | `1.08860x` | 18 |
| Setup | 47 | `0.99348x` | 3 |
| Field synchronization | 47 | `0.45356x` | 46 |
| Fixed RHS | 47 | `1.21796x` | 17 |
| One complete fixed step | 47 | `1.05762x` | 18 |
| Complete fixed solve | 45 | `1.42550x` | 9 |
| Dense output | 45 | `1.44746x` | 6 |
| Preallocated result copy | 44 | `0.98128x` | 2 |

Small adaptive solve is below the target and 18/45 branches regress by more
than 5%, so the plan's small-workload acceptance criterion is not met. Field
synchronization is especially unfavorable at `0.45356x`, which supports
investigating automatic Julia fallback or deferred synchronization rather than
assuming small native work can amortize the FFI seam. `modal_real_tapered`
joins `free_real_zdependent` as a small interpolation-invalid exclusion: its
adaptive terminal error is `1.1818e-6` and its fixed interpolated terminal
error is `4.3981e-4`, although raw-state gates pass. Small result copy also
excludes `modeavg_env_raman_sio2` because its Julia cell capped at 6.68% MAD
and 7.07% CI half-width.

The first medium matrix and first three complete large rounds used a fresh
process for each observation. Resumed timing uses one persistent, clean
process per implementation, with two warmups on first use of each cell and one
randomized round-robin observation per request. This preserves implementation
isolation while avoiding repeated full-solve warmups. Persistent-session
`Sys.maxrss()` values are cumulative and are excluded from per-cell RSS
summaries; saved fresh-process samples and the dedicated RSS/counter pass are
the RSS evidence.

Large free-space fixtures exposed a harness-lifecycle limit: discarded Rust
warmup `NativeSim` handles were retained until finalization and accumulated to
45.2 GiB, causing a kernel OOM kill. The runner now explicitly finalizes every
discarded native handle, performs full GC between warmups, and uses a 6-GiB
post-sample recycle guard. The exact failing `free_env_kerr/rust` cell then
completed two warmups plus one measured solve; its measured solve was 68.39 s
and the process still reached a 33.3-GiB peak. No production runtime source was
changed by this harness fix.

The pre-audit reference result
(`0.885x`, Julia-oracle median 0.985609 s versus native median 1.113455 s,
relative field error 1.6624e-9) is a result to reproduce and explain, not a
frozen audit conclusion.

## Focused accepted optimization unit — 2026-08-24

The design selected from the profiling evidence is implemented: allocation-free
native attempt ownership, no-op callback resync elision, direct configured-BLAS
resident QDHT, AVX2/NEON Raman recurrence, scratch-isolated Julia modal callback
batching, and bounded process-scan topology. The matched focused medians are:

| Fixture | Native fixed step before → after | Gain | Native adaptive solve before → after | Gain |
|---|---:|---:|---:|---:|
| modeavg rotational Raman | 1.48384 → 1.03398 ms | 30.3% | 4.49170 → 3.09266 ms | 31.1% |
| radial Kerr | 19.6087 → 10.2834 ms | 47.6% | 61.7572 → 31.1040 ms | 49.6% |
| radial mixture | 19.7691 → 10.2726 ms | 48.0% | 62.1780 → 31.2697 ms | 49.7% |
| radial rotational Raman | 110.020 → 69.5207 ms | 36.8% | 2695.22 → 1673.11 ms | 37.9% |
| radial shot noise | 41.5752 → 22.3505 ms | 46.2% | 302.870 → 156.823 ms | 48.2% |

Fixed-step allocation fell to 96 bytes in every native cell; adaptive-solve
allocation fell to 480–1,088 bytes. Errors are `7.21e-18`–`4.03e-8` for the
fixed-step captures and `2.43e-16`–`2.01e-7` for adaptive solves, all below the
unchanged `1e-6` gate. Direct field synchronization changed by +0.67% radially
and -2.02% in the tiny modal case, both within noise.

Local validation includes 88 Rust unit/build-policy tests; an initial native
Julia gate with 42,901 passes whose only two failures were the new timing-file
registrations, followed by a repaired 406/406 scheduler manifest;
`sim-multimode` 53/53; `sim-interface` 314/314; and focused legacy plus
concurrent scan coverage 193/193. AArch64 compilation passes. No Apple
runtime result is claimed on this x86_64 Linux host; the quick runner is ready
to collect M-chip/Accelerate/NEON evidence and production LTO remains unchanged.

The fresh-process small correctness sweep admits 47/49 fixtures; medium and
large each admit 36/49. Both universally rejected
cells are manually constructed mode-averaged EnvGrid plasma branches for which
the Julia oracle throws `InexactError` in `PlasmaScalar!`; the public interface
does not expose either combination, so they are explicit invalid-oracle audit
exclusions rather than supported cells. At medium size, 36/49 pass. Eleven
four-mode modal cells fail the documented modal single-step tier: most are
approximately `3.18e-9`, full-representation cells are `2.20e-10`, and the
general-mode cell is `8.90e-8`, against a `1e-10` ceiling. Raman modal cells use
their separately documented reassociation tier and pass. These failed cells
are not admitted to timing results.

The upstream probe compares raw fixed-step terminal state to avoid pinned
Luna's known deferred-FSAL dense-output defect. All 46 common configurations
pass the `1e-6` tier at small and medium size. At large size, all 35 common
configurations that also pass the fork gate are equivalent. The worst observed
relative differences are `7.902832940604446e-11`,
`2.7645567126914224e-13`, and `7.796588377966324e-14` for small, medium, and
large respectively; each maximum is `modeavg_env_raman_sio2`. The only
fork-only branch is `free_real_zdependent`: pinned Luna has no
`LinearOps.make_linop_free_gradient`.

## Limitations at this checkpoint

- Medium/large modal timing is correctness-blocked where the strict gate fails;
  those cells are explicit exclusions rather than silently loosened tests.
- `perf_event_paranoid=4` on the capture host prevents ordinary unprivileged
  hardware counters; profiling will require approved elevated execution.
- Claims currently apply only to this Ryzen 5 5600X host.
- No production source or optimization has changed.

# CPU performance-audit harness

This directory implements the audit specified in
`docs/dev/native-port/PERFORMANCE_AUDIT_PLAN.md`. Production optimization is
out of scope until the measurement and root-cause checkpoints are complete.

## Checkpoints

1. **Frozen baselines and branch inventory — complete.** Pin Amalthea and
   upstream Luna commits, capture dependency/toolchain/host state, identify
   the portable installed-build artifact, and enumerate every distinct
   resident CPU control-flow class at three sizes.
2. **Runnable correctness-gated fixtures — complete.** All 49 constructors ran
   at all three sizes. The small gate admits 47/49; medium and large each admit
   36/49. Two EnvGrid-plasma cells have no working Julia oracle, and scaled
   non-Raman modal cells expose a resident-backend equivalence regression. All
   provisional upstream classifications are resolved. Pinned Luna equivalence
   is 46/46 at small and medium size and 35/35 for the large, fork-admissible
   common subset; the free-space z-dependent operator is fork-only.
3. **Stable timing and counter matrix — paused with preliminary evidence.** The correctness-gated
   one-core medium and large adaptive matrices are complete. Medium setup,
   field-sync, fixed-RHS, fixed-step, fixed-solve, dense-output, and result-copy
   matrices are also complete at small and medium size. Large setup and
   field-sync are complete; the other five large component matrices, dedicated
   RSS/counters, and thread scaling remain. The lead stopped the exhaustive
   campaign on 2026-08-24 to begin focused optimization from
   `docs/dev/native-port/PERFORMANCE_AUDIT_PRELIMINARY_RESULTS.md`; partial
   large fixed-RHS samples are not accepted aggregate evidence.
4. **Historical-result reconciliation and profiles — incomplete.** Reproduce
   the historical favorable sample and current slowdown; isolate one variable
   at a time and profile stable medium/large cells.
5. **Independent optimization hypotheses — incomplete.** Prototype only after
   measured ceilings exist; do not stack changes before attribution.
6. **Final report, full validation, and ranked roadmap — incomplete.**

### Timing-process protocol

The first medium matrix and the first three complete large rounds launched a
fresh Julia process for every observation.  That is stricter than the audit
plan requires and repeats two full-solve warmups for every timed solve.  Large
adaptive cells made that implementation prohibitively expensive: ten rounds
would execute 2,160 solves for 720 observations.

Resumed timing matrices therefore keep one clean, persistent Julia process per
implementation.  Competing implementations are never loaded in the same
process.  Each backend process performs two unmeasured warmups the first time
it sees a fixture/size/measurement tuple, after which the Python orchestrator
continues to request exactly one observation at a time in its seeded,
randomized round-robin order.  Setup is still recreated for every observation,
and timing convergence still requires 10--30 observations per cell.  Session
restart repeats the two warmups before accepting another observation.
The runner explicitly finalizes discarded Rust `NativeSim` handles, forces full
GC before and between warmups, and recycles a backend session above 6 GiB
post-sample RSS.  This is necessary on the 46-GiB capture host: the first
persistent large run retained discarded warmup native buffers and the kernel
OOM-killed it at 45.2 GiB; the worst single `free_env_kerr/rust` request still
peaks at 33.3 GiB.  All JSON written before the kill remains valid and is reused
by `--resume`.

The orchestrator prints a flushed `starting` and `completed` line around every
fixture/backend request. Long unattended jobs can therefore be held in one
blocking executor wait: completion wakes the caller without filesystem
polling, while a timeout or interruption identifies the exact active cell in
the captured output.

`Sys.maxrss()` is process-cumulative, so persistent-session observations are
not used as per-cell peak-RSS samples.  Peak RSS is summarized from the saved
fresh-process observations and from the dedicated RSS/counter pass; allocation
measurements remain per observation.

The `field_sync` and `fixed_rhs` component probes are microbenchmarks. Their
inner repetition count is calibrated independently for each fixture/backend
request to target at least 20 ms of timed work (unless
`AMALTHEA_AUDIT_REPETITIONS` explicitly fixes the count). This keeps the
reported value per operation while preventing scheduler/timer noise from
dominating sub-microsecond and low-microsecond cells. The first uncalibrated
medium `field_sync` pass used 20 operations per observation and left 13/72
cells non-converged after 30 samples; it is retained as
`matrix-field_sync-medium-underresolved.json` and is diagnostic only.

`fixed_rhs` may invoke the same Raman RHS hundreds of times on an unchanged
state to obtain a stable latency sample. That synthetic repetition is not a
valid trajectory correctness test: the resident Raman scratch evolves enough
for the final repeated stage to drift even though a fresh RHS/step and the full
solve pass their strict gates. The matrix therefore records the final-batch
field difference as `timed_batch_final_field_relative_error`, but gates the
timing with the frozen fixture's independently measured strict single-step
error and tolerance from `correctness-SIZE.json`.

`fixed_step` times exactly one complete step per observation by default. An
initial medium trial batched five steps and showed per-step latencies from
0.24 ms to 1.27 s; batching therefore made the most expensive observations up
to five times longer without adding independent samples. The randomized
10--30 observation convergence rule supplies the replication instead. Set
`AMALTHEA_AUDIT_REPETITIONS` only for an explicit batching diagnostic.

`dense_output` and `result_copy` use the same per-request 20 ms calibration as
the other microbenchmarks. The first medium dense-output trial fixed the batch
at 200 calls even though an interpolation computes two additional RK stages;
one measured modal call already took about 7 ms. Adaptive calibration retains
batching for cheap copies while allowing an expensive interpolation to stand
alone. The interrupted two-cell trial is preserved under
`matrix-dense_output-medium-200-call-batch/` and is diagnostic only.

`result_copy` first materializes the valid terminal result with
`interpolate(stepper, flength)` outside the timed interval, then times copies
of that stable result array. The first medium attempt copied `stepper.yn`
directly; after an adaptive solve that buffer is the accepted interval's
right-hand proposal rather than the requested terminal result, and on the
resident path it is also a Julia-side synchronization buffer. Its large false
field mismatches and noisy timings are retained as
`matrix-result_copy-medium-raw-yn.json` and are not performance evidence.
The subsequent corrected-field attempt timed allocating `copy(result)` and
left 28/70 backend cells non-converged at 30 samples because repeated
allocation/GC dominated the microbenchmark. The production `solve(...;
output=true)` seam assigns into preallocated `yout`, so the accepted probe
times `copyto!` into a preallocated destination. Allocation totals remain
available from every full-solve `@timed` sample. The allocating-copy attempt is
retained as `matrix-result_copy-medium-allocating-copy.json` and is diagnostic
only.

The adaptive-solve sweep additionally excludes `free_real_zdependent` through
`--exclude-fixture`.  Its fixed-step correctness gate passes, but the medium
adaptive trajectory reproducibly differs from the Julia oracle by
`1.45314e-2` despite matching four accepted and zero rejected steps.  Saved
timings for that cell are diagnostic only and cannot support a performance
claim until the z-dependent adaptive/dense-output seam is corrected.

Large adaptive timing found two more adaptive-only trajectory failures:
`modal_real_raman_thg` uses 9 Julia versus 10 Rust accepted steps and differs by
`2.04562e-6` against `1e-6`; `modal_real_raman_nothg` likewise uses 9 versus 10
steps and differs by `1.91107e-6` against `1.5e-6`.  Two correctness-valid
Julia radial-plasma cells exhausted 30 observations without meeting the 3%
relative-MAD gate (`radial_real_ppt` 5.11%, `radial_real_adk` 4.28%).  Their raw
timings are preserved in `matrix-adaptive_solve-large-correctness-admissible.json`
but all four fixture pairs are excluded from the accepted aggregate.

## Frozen commits

- Amalthea: `73e32dcf45d93f11136d419faeae3b3641c9577d`
- Upstream Luna.jl: `0a52ffbba6d5dd6820bb3dc3c300b8b38d724214`

The upstream remote was fetched on 2026-08-11 and still resolved to the same
commit used by the 2026-07-02 fork review. Commit pinning is intentional: a
later upstream change starts a new baseline rather than silently changing this
one.

`upstream/Project.toml` and `upstream/Manifest.toml` freeze the dependency
resolution used for the pinned upstream checkout (manifest SHA-256
`aa492c867b702a27916b760037dd39ccf42c6d6d794bbb2d8e7343c7e9f640ef`).
Recreate its isolated source tree and depot with:

```bash
git archive --format=tar --output=/tmp/amalthea-upstream-0a52ffb.tar \
  0a52ffbba6d5dd6820bb3dc3c300b8b38d724214
mkdir -p /tmp/amalthea-upstream-0a52ffb
tar -xf /tmp/amalthea-upstream-0a52ffb.tar \
  -C /tmp/amalthea-upstream-0a52ffb
cp test/performance_audit/upstream/Manifest.toml \
  /tmp/amalthea-upstream-0a52ffb/Manifest.toml
JULIA_DEPOT_PATH=/tmp/luna-upstream-depot:/home/diego/.julia \
  julia --startup-file=no --project=/tmp/amalthea-upstream-0a52ffb \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

## Reproduce checkpoint 1

Build the acceptance artifact through the package-build path so the repository
`target-cpu=native` Cargo config is neutralized:

```bash
AMALTHEA_RUST_SKIP_DOWNLOAD=1 AMALTHEA_CUDA_BUILD=off RUSTFLAGS='' \
  julia --startup-file=no --project deps/build.jl
```

Then capture and validate:

```bash
python3 test/performance_audit/capture_baseline.py
python3 test/performance_audit/validate_inventory.py
```

`results/baseline.json` contains the exact artifact checksum, project/manifest
checksums, Julia/Rust/FFTW/BLAS versions, CPU microcode/topology, memory,
governor/turbo state, affinity, `perf_event_paranoid`, relevant environment
variables, and dirty-source validation. Documentation and audit-harness edits
may be dirty; runtime source and dependency metadata may not be.

## Inventory rules

`workloads.toml` is derived from the `NativeIneligible` guard and setter
branches in `src/RK45.jl`; `NATIVE_SUPPORT_MATRIX.md` and native equivalence
tests are cross-checks, not the authority. Each entry identifies its source
guard and oracle-test provenance, small/medium/large shape, threadability, and
upstream status.

The inventory deliberately enumerates control-flow classes instead of a full
Cartesian product. For example, Kerr+PPT and Kerr+ADK are distinct because
they select different native rate kernels, while the same constant-density
Kerr path is not repeated for every gas species. Measurement variants such as
adaptive versus fixed step, setup included/excluded, and dense-output cadence
are orthogonal sweeps declared under `[measurement]`.

The initial derived-document discrepancies were resolved by executable
small-fixture probes:

- modal and free-space mixtures execute resident plain-Kerr mixture branches
  on both grids;
- radial EnvGrid mixtures execute the grid-independent radial mixture branch.

All eight geometry/grid mixture fixtures constructed a CPU resident stepper,
passed the single-step tier, passed fixed-solve agreement below `1e-6`, and
proved a Kerr effect above `1e-8` at small size.

The pinned-upstream raw-terminal-state probe admits 46 cells at small and
medium size and the 35 large cells that also pass the fork gate. All are below
the `1e-6` equivalence tier. Worst relative errors are
`7.902832940604446e-11`, `2.7645567126914224e-13`, and
`7.796588377966324e-14` respectively.
`free_real_zdependent` is unavailable because pinned Luna has no
`LinearOps.make_linop_free_gradient`. The probe deliberately compares raw
fixed-step terminal state: pinned upstream's known deferred-FSAL dense-output
defect otherwise creates a false `2.25e-6` modal mismatch.

## Apple Silicon quick diagnostic

From an Apple Silicon checkout, run:

```bash
python3 test/performance_audit/run_apple_quick_test.py \
  --output test/performance_audit/results/apple-quick.json
```

The runner emits the JSON file and a sibling Markdown summary in 5–10 minutes.
It records the M-chip and performance/efficiency cores, macOS/Julia/Rust,
FFTW library, Julia BLAS provider, relevant thread environment, and 1/2/4-thread
rotational-Raman/radial-QDHT timings. It also checks exact modal callback
threading and a two-worker exact-once scan with one thread per worker.

The normal portable CPU artifact is built first and saved. One diagnostic
`target-cpu=native`, thin-LTO, one-codegen-unit artifact is then measured, and
the portable artifact is restored in `finally` even on failure. Treat
`lto_recommendation=candidate only` literally: production LTO still requires a
separate ≥5% local end-to-end gain and the portability gates. On non-Apple
hosts, `--allow-non-apple --dry-run` validates output/schema only; it is not
NEON, Accelerate, or M-chip evidence.

Two low-level `TransModeAvg` EnvGrid plasma constructions are retained in the
inventory as explicit audit exclusions. Their Julia oracle calls
`PlasmaScalar!` with a complex envelope and throws `InexactError`; the public
API correctly does not offer those combinations. They therefore cannot be
timed as supported resident branches. At medium size, four-mode modal
single-step errors range from `2.20e-10` to `8.90e-8` for non-Raman branches,
above the documented modal `1e-10` tier. Timing orchestration reads the size's
correctness JSON and automatically excludes every failed cell.

## Historical and public benchmark probes

The historical Phase-C record does not include its literal command. Its
durable description freezes a 125 μm × 15 cm helium capillary at 1 bar,
800 nm, 30 fs, 1 μJ, `saveN=50`, default Kerr+PPT plasma, a fixed RNG seed,
one warmup, and ten accepted steps. The audit recreates that best-known input
and labels it `phase_c_reconstruction`; it must not be presented as a
bit-for-bit reproduction of omitted parameters. The current public benchmark
is fully specified by `test/benchmark_julia_vs_native.jl` and is labelled
`readme_v103`: 125 μm × 3 m, 1 bar He, 800 nm, 10 fs, 120 μJ, RealGrid,
Kerr+PPT, no Raman or shot noise.

`run_public_benchmark.jl` runs either configuration in a fresh process through
one backend only, performs two warmups, asserts the selected stepper, and emits
JSON plus the final field. `run_upstream_public_benchmark.jl` applies the same
public call to the pinned upstream project. This keeps the primary comparison
process-isolated; the original `test/benchmark_julia_vs_native.jl` is also run
unchanged to reproduce the published same-process five-trial number.

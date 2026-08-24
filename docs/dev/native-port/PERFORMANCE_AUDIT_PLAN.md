# Deep Performance Audit: Make Amalthea Faster Than Luna.jl

## Purpose and persistence

This document is the durable resume point for a deep, evidence-backed audit of
the resident CPU-native backend. The audit must explain why Amalthea is slower
than Julia/Luna in affected workloads, identify any additional slowdowns, and
rank changes that can make the installed portable backend faster without
weakening numerical correctness or portability.

The audit compares three independent baselines:

1. Amalthea's installed-default portable Rust resident CPU backend.
2. Amalthea's retained Julia oracle.
3. A pinned checkout of actual upstream Luna.jl for configurations supported by
   both projects.

Quantitative conclusions initially apply only to the current AMD Ryzen 5 5600X
host. The harness and recorded metadata must make later ARM, Intel, and other
AMD runs reproducible.

The investigation produces
`docs/dev/native-port/PERFORMANCE_AUDIT_REPORT.md`, with reproducible harnesses
and machine-readable results under `test/performance_audit/`. Temporary
instrumentation and isolated optimization prototypes are allowed during the
audit, but production optimizations are not merged until the report identifies
and ranks them.

To resume in a new conversation, use:

> Read `AGENTS.md` and
> `docs/dev/native-port/PERFORMANCE_AUDIT_PLAN.md`, then execute the performance
> audit from its first incomplete checkpoint.

## Benchmark design

- Freeze the Amalthea commit, upstream Luna commit, Julia/Rust/FFTW versions,
  manifest, kernel, CPU microcode, governor, turbo state, memory, and all
  thread-related environment variables.
- Run upstream Luna, Amalthea Julia, and Amalthea Rust in separate clean Julia
  processes and projects. Do not load competing implementations into the same
  process.
- Treat the installed portable package build as the acceptance baseline.
  Measure `target-cpu=native`, release LTO, and reduced codegen units only as
  controlled diagnostic variants.
- Derive the exhaustive workload inventory from the native eligibility guards
  and execution branches, then cross-check it against the support matrix and
  existing native tests. Cover every distinct eligible resident CPU path:
  - Mode-averaged, radial, modal, and free-space geometries.
  - RealGrid and EnvGrid.
  - Kerr variants, polarization counts, THG branches, full/modal
    representations, plasma, PPT/ADK, SDO and FFT Raman, shot noise, mixtures,
    and supported z-dependent behavior.
  - Upstream-compatible cases receive three-way comparisons. Fork-only cases
    compare Rust with the internal Julia oracle and are marked unavailable
    upstream.
- Give each branch small, medium, and large fixtures. Avoid redundant Cartesian
  combinations when they execute identical control flow.
- Measure separately:
  - Initialization, FFT planning, and setup.
  - Fixed-step RHS and complete-step latency.
  - Adaptive full-solve wall time.
  - Accepted/rejected steps and RHS evaluations.
  - Dense output and result-copying overhead.
  - Allocations, peak RSS, cycles, instructions, IPC, cache misses, and branch
    misses.
- Primary measurements use one Julia thread, one FFTW thread, one BLAS/OMP
  thread, and CPU affinity to one physical core. Threaded radial, modal, and
  free-space paths additionally use 1, 2, 4, and 6 physical cores.
- Use fixed seeds and identical physical inputs. Run two unmeasured warmups,
  then at least 10 randomized round-robin samples, extending to 30 until median
  relative MAD is at most 3% and the bootstrap 95% confidence-interval
  half-width is at most 5%.
- Gate every timed fixture on correctness:
  - Strict single-step agreement at the applicable testing tier.
  - Full-solve agreement at no worse than the documented `1e-6` floor.
  - A non-vacuity check proving the exercised physical feature changes the
    oracle by more than the asserted tolerance.

## Root-cause investigation

- Reproduce both the historical claimed `~3.5x` advantage and the current
  observed `0.885x` result. Change one variable at a time to determine which
  factors reversed the outcome:
  - Portable versus host-native Rust code generation.
  - Fixed versus adaptive integration.
  - Setup included versus excluded.
  - FFTW planning mode, wisdom, and thread count.
  - Cold versus warm plasma and Raman state.
  - Output and dense-sampling frequency.
  - Native worker count.
  - Accepted/rejected step and RHS-call divergence.
- Profile stable medium and large fixtures with `perf stat` and sampled call
  stacks. Generate comparable Julia and Rust flame graphs where permissions
  permit.
- If sampling cannot resolve internal native stages, add temporary low-overhead
  timers around RHS components, FFTs, linear propagation, nonlinear responses,
  error estimation, FFI crossings, copying, and output synchronization. Revert
  temporary instrumentation after exporting raw results unless it proves
  generally useful and safe.
- Inspect every matrix cell for additional slowdowns, including:
  - Portable code-generation losses and missed SIMD.
  - FFT planning or execution differences despite both paths using FFTW.
  - Redundant FFTs, memory passes, allocations, or conversions.
  - Per-call Rayon pool construction or poor parallel scaling.
  - Julia-to-Rust field copying and unnecessary `yn` synchronization.
  - FFI overhead on small grids.
  - Adaptive-controller differences that alter step counts.
  - Dense-output and callback seams.
  - Plasma/Raman scans, z-dependent setters, and branch-specific scratch
    handling.
  - Cache, memory-bandwidth, NUMA, and false-sharing effects.
- Calculate an Amdahl-law ceiling for every suspected optimization. Do not
  recommend an optimization as strategically important unless profiling
  supports a plausible end-to-end gain of at least 5%, or it removes a measured
  branch-specific regression.
- Benchmark optimization hypotheses individually against the frozen baseline.
  Do not stack prototypes until each gain is independently attributed.

## Suggested-change evaluation

Rank recommendations by measured gain, implementation effort, correctness risk,
portability, and maintainability. Explicitly investigate:

- Runtime CPU feature dispatch with portable scalar/SSE baselines and
  architecture-specific AVX2/AVX-512 or ARM NEON kernels, so installed artifacts
  can approach host-native compilation without sacrificing portability.
- A production release profile using thin LTO and one codegen unit, including
  build-time and artifact-size effects.
- Persistent or reused worker pools if thread-pool construction appears in hot
  paths.
- Reduced field copying and deferred Julia synchronization across native steps
  where output, rejection, and lifetime semantics permit it.
- FFT plan reuse, in-place transforms, removal of redundant transforms, and
  planning-policy parity.
- Fusion of dominant memory-bound passes only where profiles show sufficient
  ceiling.
- Exact adaptive-controller parity when differing decisions cause extra Rust
  RHS evaluations.
- Automatic Julia fallback for small workloads where fixed FFI or native setup
  costs cannot be amortized within the 5% regression limit.

For each recommendation, report measured before/after results, affected
branches, expected portable gain, numerical impact, implementation outline,
risks, and required tests. Retain a "rejected or deferred" section explaining
why plausible ideas such as mixed precision, wholesale FFT replacement, or
low-impact loop rewrites do not meet the evidence threshold.

## Report and acceptance criteria

The final report must contain:

- A direct explanation of why Amalthea is currently slower in each affected
  workload, with time percentages and uncertainty.
- An exhaustive performance table and slowdown heatmap covering all eligible
  resident CPU execution branches.
- A separate upstream Luna comparison table limited to genuinely equivalent
  public configurations.
- A reconciliation of the historical `~3.5x` claim with current measurements.
- An inventory of every measured regression greater than 5%.
- Ranked quick wins, medium-term changes, architectural changes, and rejected
  ideas.
- A cumulative roadmap showing how independently measured gains can plausibly
  reach the target without double-counting overlapping speedups.
- Reproduction commands and raw JSON results.
- Explicit limitations: current-host claims are not generalized to ARM or other
  CPUs until rerun there.

Success means:

- The installed-default portable Rust backend reaches at least a `1.20x`
  geometric-mean speedup over the Amalthea Julia oracle across medium and large
  representative cases for every eligible branch class.
- The common upstream subset and flagship public benchmark each reach at least
  `1.20x` over both the internal oracle and pinned upstream Luna.jl.
- No medium or large representative branch is more than 5% slower than its
  Julia oracle.
- Small workloads are either within 5% or automatically use the Julia path with
  dispatch overhead remaining within that limit.
- All correctness gates remain satisfied, and gains survive fresh-process
  reruns.
- If evidence shows the target is unattainable without an unacceptable
  numerical, portability, or maintenance tradeoff, the report must say so
  directly and identify the measured maximum credible speedup.

## Validation and assumptions

- Run the existing Rust equivalence and FFI safety tests affected by any
  prototype, plus the full resident CPU-native test group before accepting
  report results.
- CUDA and legacy opt-in per-kernel offloads are outside the benchmark target.
  Shared CPU kernels may be profiled only when they explain resident-backend
  behavior.
- Do not use CUDA availability as a prerequisite for the audit.
- Preserve upstream attribution and clearly distinguish upstream Luna,
  Amalthea's Julia fork path, and Amalthea's resident Rust path.
- Append a detailed `PORT_LOG.md` entry for the completed investigation,
  including exact commands, commits, tolerances, results, unresolved risks, and
  the recommended first implementation task.

# Preliminary Resident CPU Performance Results

> **Evidence cutoff:** 2026-08-24. This is an optimization starting point, not
> the completed exhaustive audit. The long-running matrix campaign was stopped
> after the lead chose to proceed from the evidence already collected. Only
> correctness- and stability-accepted matrices are used below. Partial samples
> from the interrupted large `fixed_rhs` matrix are excluded.

## Executive result

The installed portable Rust resident backend is faster in aggregate, but it
does not yet meet the audit's performance contract.

- Across 66 accepted medium and large adaptive-solve fixture/size pairs, Rust
  is `1.11913x` faster than the internal Julia oracle by geometric mean. The
  target is `1.20x`.
- Twenty-seven of those 66 pairs are more than 5% slower in Rust. The target is
  zero medium/large regressions over 5%.
- The accepted small adaptive set is `1.08860x` faster in aggregate, but 18 of
  45 pairs regress by more than 5%. Small-workload automatic fallback remains
  justified by the evidence.
- A complete medium fixed solve is `1.38159x` faster in aggregate, while one
  complete fixed step is only `1.00658x` and adaptive solve is `1.10906x`.
  The resident backend can win over a trajectory, but fixed costs and
  branch-specific stage costs erase much of that advantage.

The first optimization work should therefore target field synchronization and
the slow radial RealGrid/Raman/shot-noise RHS branches. Dense interpolation and
preallocated result copying are already faster or near parity and are poor
first targets.

## Frozen comparison

The measurements compare process-isolated implementations on the AMD Ryzen 5
5600X capture host:

1. Portable installed-default Rust resident backend at Amalthea commit
   `73e32dcf45d93f11136d419faeae3b3641c9577d`.
2. The retained Julia oracle at the same commit and manifest.
3. Pinned upstream Luna.jl commit
   `0a52ffbba6d5dd6820bb3dc3c300b8b38d724214` for equivalence checks.

The Rust artifact is the portable package build, not a `target-cpu=native`
diagnostic build. Primary timing uses one physical core and one Julia, FFTW,
BLAS, and OMP thread. Every accepted timing cell has 10--30 randomized samples,
at most 3% relative MAD, at most 5% bootstrap 95% confidence-interval
half-width, and a post-timing correctness gate.

## Coverage and correctness

The workload inventory defines 49 distinct resident CPU branch fixtures across
mode-averaged, radial, modal, and free-space geometries; RealGrid and EnvGrid;
Kerr, PPT/ADK plasma, multiple Raman paths, shot noise, mixtures, modal
representations, polarization counts, and supported z dependence.

- All 49 fixtures were constructed and exercised at small, medium, and large
  sizes: 147 correctness attempts in total.
- Small admits 47/49 fixtures. Medium and large each admit 36/49.
- Two manually constructed mode-averaged EnvGrid plasma fixtures
  (`modeavg_env_ppt` and `modeavg_env_adk`) have no valid retained Julia oracle:
  the oracle throws `InexactError`, and the public interface does not expose
  these combinations.
- Eleven scaled non-Raman modal fixtures fail the frozen strict single-step
  tier at medium/large size. They were exercised but are not performance
  evidence.
- Pinned upstream equivalence passes 46/46 common fixtures at small and medium
  size and 35/35 correctness-admitted common fixtures at large size. This
  proves physical equivalence, not upstream timing performance.

Known output-seam exclusions remain explicit. `free_real_zdependent` fails
adaptive/fixed interpolation despite a correct raw fixed terminal state.
`modal_real_tapered` has the same pattern at small size. Two large modal Raman
adaptive runs differ through one extra Rust accepted step and narrowly exceed
their numerical tiers. No tolerance was loosened to retain these cells.

## Accepted end-to-end timing

Ratios are Julia median time divided by Rust median time. Values above one mean
Rust is faster.

| Size | Accepted pairs | Geometric-mean ratio | Rust regressions >5% |
|---|---:|---:|---:|
| Small adaptive solve | 45 | `1.08860x` | 18 |
| Medium adaptive solve | 35 | `1.10906x` | 15 |
| Large adaptive solve | 31 | `1.13060x` | 12 |
| Medium + large adaptive solve | 66 | `1.11913x` | 27 |

For the accepted medium+large set, geometry-level ratios are:

| Geometry | Geometric-mean ratio |
|---|---:|
| Mode-averaged | `1.28429x` |
| Radial | `0.97223x` |
| Free-space | `1.00760x` |
| Modal | `1.54077x` |

The modal number covers only the two medium Raman fixtures admitted by the
strict correctness gate; it must not be generalized to the excluded modal
branches.

The worst accepted medium/large adaptive regressions are:

| Fixture | Size | Julia/Rust ratio | Accepted/rejected steps differ? |
|---|---|---:|---|
| `radial_real_kerr` | Medium | `0.59901x` | No |
| `radial_real_shotnoise` | Large | `0.63109x` | No |
| `modeavg_real_raman_rotational` | Medium | `0.63135x` | No |
| `modeavg_real_kerr` | Medium | `0.70735x` | No |
| `modeavg_real_shotnoise` | Medium | `0.71239x` | No |
| `free_real_raman_nothg_rotational` | Medium | `0.71437x` | No |
| `radial_real_shotnoise` | Medium | `0.72399x` | No |
| `radial_real_mixture` | Medium | `0.72661x` | No |
| `radial_real_raman_nothg_rotational` | Medium | `0.73529x` | No |
| `modeavg_real_kerr` | Large | `0.75116x` | No |

Because accepted/rejected counts match in these cells, adaptive-controller
divergence cannot explain their slowdown. Their costs are inside the executed
stages and synchronization seams.

## Component localization

### Small and medium

| Component | Small ratio | Medium ratio | Preliminary interpretation |
|---|---:|---:|---|
| Setup | `0.99348x` | `0.90942x` | Rust setup is slower, especially on medium free/radial cases. |
| Field synchronization | `0.45356x` | `0.68219x` | The clearest fixed-cost regression. |
| Fixed RHS | `1.21796x` | `1.19741x` | Aggregate is near target, but branch regressions are severe. |
| One complete fixed step | `1.05762x` | `1.00658x` | Stage-level gains are mostly lost at the complete-step seam. |
| Complete fixed solve | `1.42550x` | `1.38159x` | Resident execution amortizes fixed costs over a trajectory. |
| Dense output | `1.44746x` | `1.50780x` | Not a first optimization target. |
| Preallocated result copy | `0.98128x` | `0.99773x` | Near parity; allocation-free copy is not the main loss. |

At medium size, the geometry split localizes the RHS problem further:

| Component | Mode-averaged | Radial | Modal | Free-space |
|---|---:|---:|---:|---:|
| Fixed RHS | `1.60205x` | `0.81405x` | `1.40053x` | `1.17835x` |
| One fixed step | `1.19610x` | `0.77898x` | `1.36717x` | `0.98361x` |
| Complete fixed solve | `1.67766x` | `1.12887x` | `1.46189x` | `1.28030x` |
| Field synchronization | `0.51315x` | `0.91613x` | `0.19847x` | `0.97482x` |
| Setup | `0.99665x` | `0.88299x` | `1.08171x` | `0.75477x` |

The five slowest medium fixed-RHS fixtures are `radial_real_kerr` (`0.56532x`),
`radial_real_mixture` (`0.56534x`),
`radial_real_raman_nothg_rotational` (`0.60643x`),
`modeavg_real_raman_rotational` (`0.62414x`), and
`radial_real_shotnoise` (`0.63671x`). These fixtures provide a compact first
profiling set instead of rerunning the full matrix.

### Completed large components

Only large setup and field synchronization were completed before the campaign
was stopped:

| Component | Accepted pairs | Geometric-mean ratio | Rust regressions >5% |
|---|---:|---:|---:|
| Setup | 36 | `0.97582x` | 16 |
| Field synchronization | 35 | `0.85421x` | 15 |

The excluded field-sync pair is `modeavg_env_mixture`, whose Rust cell reached
30 samples without satisfying both stability gates. The full diagnostic data
is retained.

## Ranked optimization starting points

These are evidence-backed directions, not claims that the root cause or gain
has already been proven.

### 1. Remove or defer avoidable field synchronization

**Evidence:** field synchronization is `0.45356x`, `0.68219x`, and `0.85421x`
at small, medium, and large size. Medium mode-averaged synchronization is
`0.51315x`; medium modal is `0.19847x`. Preallocated terminal copying is near
parity, so the problem is not generic memory copying alone.

**Code to inspect first:** `CpuNativeSim::set_field` in
`amalthea/src/native.rs` copies Julia's field into `sim.field` and then clones
that field before dispatching the stage-0 RHS. `native_resync_field` and the
accepted-step/output synchronization paths should be mapped for redundant
copies. A prototype should preserve rejection, callback, windowing, and object
lifetime semantics and measure end-to-end gain, not only copy bandwidth.

### 2. Profile and optimize radial RealGrid RHS/step execution

**Evidence:** medium radial fixed RHS is `0.81405x` and a complete radial step
is `0.77898x`; `radial_real_kerr` and `radial_real_mixture` are each about
`0.565x` in the RHS probe. Matching step counts rule out the controller in the
worst end-to-end regressions.

**First question:** determine whether QDHT transforms, FFT layout/transposes,
temporary field clones, serial memory passes, or one-thread Rayon/worker
overhead dominates `rhs_radial`. Instrument only
`radial_real_kerr`, `radial_real_mixture`, and
`radial_real_raman_nothg_rotational` initially.

### 3. Isolate rotational Raman and shot-noise costs

**Evidence:** the worst mode-averaged RHS regression is rotational Raman at
`0.62414x`; radial shot noise is `0.63671x`, and both remain prominent in
adaptive solves. Compare their component timings against the corresponding
plain-Kerr fixture to isolate the incremental scan/noise cost.

### 4. Reuse expensive setup state where ownership permits

**Evidence:** medium setup is `0.90942x`; medium free-space setup is `0.75477x`
and radial setup is `0.88299x`. Large setup approaches parity, so this is
primarily an amortization and small/medium workload problem.

**Candidates:** FFT plan reuse, persistent worker resources, and avoiding
reconstruction of immutable geometry/response tables. Build-profile changes
should be tested separately so their gain is not confused with reuse.

### 5. Add conservative automatic Julia fallback for small work

**Evidence:** 18/45 small adaptive branches regress by more than 5%, and small
field synchronization is `0.45356x`. The existing plan explicitly permits a
fallback when fixed native costs cannot be amortized. Dispatch must be based on
stable shape/branch thresholds and remain under the 5% overhead limit.

### Not first targets

- Dense output: already `1.45--1.51x` faster in accepted small/medium sets.
- Preallocated terminal copy: approximately parity.
- Controller parity for the worst accepted regressions: their step and
  rejection counts already match. Controller parity still matters for the two
  excluded large modal Raman trajectories, but it is not the broad slowdown.
- Stacked `target-cpu=native`, LTO, and codegen-unit changes: useful later as
  independent build variants, but they cannot localize the current branch and
  synchronization losses.

## What remains unknown

The stopped exhaustive work means this report does not yet contain:

- Accepted large fixed-RHS, fixed-step, fixed-solve, dense-output, or
  result-copy matrices.
- Dedicated peak-RSS measurements, hardware counters, call-stack profiles, or
  flame graphs.
- 1/2/4/6-core scaling for threaded radial, modal, and free-space branches.
- Timed pinned-upstream tables or the flagship public-benchmark comparison.
- One-variable reconciliation of the historical `~3.5x` claim and the earlier
  `0.885x` observation.
- Independently measured before/after optimization prototypes or Amdahl-law
  ceilings.
- Evidence that the final `1.20x`/no-regressions acceptance contract is met.

Optimization work should therefore use focused A/B tests on the five-fixture
profiling set above. Any production change still needs the strict single-step
and full-solve correctness gates, followed by a fresh-process end-to-end rerun
of every affected branch.

## Superseding focused optimization result — 2026-08-24

The five-fixture A/B requested above has now been executed for the selected
ownership/QDHT/Raman unit. With the identical ten-sample, core-2, one-thread
protocol, native fixed-step medians improved 30.3% (mode-averaged rotational
Raman), 47.6% (radial Kerr), 48.0% (radial mixture), 36.8% (radial rotational
Raman), and 46.2% (radial shot noise). Native adaptive solves improved 31.1%,
49.6%, 49.7%, 37.9%, and 48.2% in the same order. Per-step Julia-visible
allocation is now 96 bytes in all five cells.

All final fields pass the frozen `1e-6` gate without tolerance changes.
Explicit field-sync timings are unchanged within noise, locating the gain in
removed no-op transfers, stage-vector clones, configured-BLAS QDHT, and Raman
SIMD rather than altered transfer semantics. The files
`/tmp/amalthea-focused-{baseline,after}-fixed-step.json` and
`/tmp/amalthea-focused-after-adaptive-solve.json` are local capture artifacts;
the exact reproduction commands are retained in `PLANS.md` §15.8 and the
2026-08-24 `PORT_LOG.md` entry.

This closes the selected production unit, not the exhaustive audit. Large
fixed-component matrices, full topology scaling, counters, and the Apple-host
run remain outside this checkpoint.

## Evidence and reproduction

- Frozen plan: [`PERFORMANCE_AUDIT_PLAN.md`](PERFORMANCE_AUDIT_PLAN.md)
- Detailed evolving report: [`PERFORMANCE_AUDIT_REPORT.md`](PERFORMANCE_AUDIT_REPORT.md)
- Harness protocol: [`../../../test/performance_audit/README.md`](../../../test/performance_audit/README.md)
- Workload inventory: [`../../../test/performance_audit/workloads.toml`](../../../test/performance_audit/workloads.toml)
- Combined accepted rows: [`../../../test/performance_audit/results/matrix-analysis.json`](../../../test/performance_audit/results/matrix-analysis.json)
- Frozen baseline metadata: [`../../../test/performance_audit/results/baseline.json`](../../../test/performance_audit/results/baseline.json)

Canonical per-matrix JSON files live in
`test/performance_audit/results/matrix-<measurement>-<size>.json`. Directories
or files carrying `correctness-admissible`, `underresolved`, `raw-yn`, or
`allocating-copy` suffixes are retained diagnostics and are not accepted
aggregate evidence.

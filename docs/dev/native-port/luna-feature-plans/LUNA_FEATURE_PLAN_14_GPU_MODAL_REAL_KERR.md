# Luna feature plan 14 — CUDA modal RealGrid Kerr

Status: complete 2026-08-04; standing CUDA CI remains strongly preferred.

## Outcome

A bounded modal Kerr surface—RealGrid, constant-radius eligible Marcatili/
Zeisberger/Vincetti mode collections, constant linop, `full=false|true`, and
`npol=1|2`—runs through a resident CUDA backend while retaining libcubature's
adaptive node placement.

## Architecture decision

Do not invent a different quadrature rule. Keep the same host
`libcubature` binary and adaptive batches so node placement and stopping
criteria remain comparable. For each callback batch, copy only node
coordinates to CUDA, evaluate field synthesis/FFT/Kerr/projection on device,
and copy the small `fval` batch back; the resident spectral state and large
scratch arrays must not return to the host. Document this bounded control-data
exception to the general traffic budget and benchmark it.

## Mathematical and layout contract

For each cubature node `(r, θ)`, modal synthesis is the same one used by the
CPU native path. For mode `m` and selected polarization `p`, define

```
q_m = n_m - 1                         (HE), or 1 (TE/TM)
B_m(r) = J_{q_m}(unm_m*r/a) / sqrt(N_m)
e_{m,p}(r,θ) = B_m(r) * A_{kind_m,p}(n_m, ϕ_m, θ)
Er_ω,p(r,θ) = Σ_m Em_ω,m * e_{m,p}(r,θ)
```

`A` is the closed-form `(x,y)` angular factor from
`amalthea/src/native.rs::mode_angle_xy`. Julia transfers `N_m`, `unm_m`,
`a`, `kind_m`, `n_m`, and `ϕ_m`; CUDA must not recompute mode normalization,
dispersion, or effective area. Zeisberger and Vincetti wrappers use their
inner Marcatili profile; their distinct dispersion is already baked into
Julia's transferred `linop`.

For RealGrid, each polarization/node batch is a contiguous
`n_time_over` cuFFT series. The device pipeline is:

```
zero-pad and scale Er_ω → Er_ωo       scale_fwd=(No/2)/(N/2) in r2c lengths
inverse batched D2Z/Z2D → Er(t)
scale by 1/n_time_over
P(t) = kerr_fac*E(t)^3                 npol=1
Pp(t) = kerr_fac*(Ex²+Ey²)*Ep(t)       npol=2
P(t) *= towin(t)
forward batched D2Z → P_ωo
crop and scale by (Nω-1)/(Nωo-1), then multiply nlfac(ω)
Prm_ω,m = jac(r) * Σ_p P_ω,p * e_{m,p}(r,θ)
```

For `full=false`, `θ=0` and `jac(r)=2πr`; for `full=true`, θ is a genuine
second cubature coordinate and `jac(r)=r`. The output is packed in
libcubature's point-major `(npt, 2*Nω*nmodes)` real/imaginary layout, exactly
matching Julia's `reinterpret(Float64, Prmω)` result. Device buffers use
column-major batch storage: `series = node*npol + p`, with time/spectrum index
fastest. The resident modal state remains in `ystage_d`; no modal field copy is
made for an RHS evaluation.

The CUDA Bessel evaluator must implement the CPU native `jn` contract: exact
`J0`/`J1` branches, integer-order downward Miller recurrence for `q>=2`, and
the same sign/zero behavior. The fixed-node test compares its point evaluator
against CPU native before any adaptive cubature comparison.

## Resource and traffic contract

`native_set_modal_params` stages every metadata buffer, the batched inverse and
forward RealGrid cuFFT plans, and bounded node/scratch buffers before replacing
the active modal setup. Any invalid dimension, null pointer, allocation, copy,
plan, or libcubature-load failure leaves the previous setup untouched. A
`modal_batch_capacity` of 32 is the initial bounded staging choice; a larger
libcubature callback is processed in contiguous sub-batches without changing
node placement or output ordering.

For each callback sub-batch of `b` nodes, the only host/device traffic is:

```
host → device: 2*b Float64 node coordinates (r,θ; θ=0 is still explicit)
device → host: b * 2*Nω*nmodes Float64 output values
```

The resident modal spectrum, mode metadata, FFT scratch, Kerr scratch, window,
and normalization remain on the device. The test records callback count,
transferred bytes, and wall time for `full=false` and `full=true`; these are a
bounded control-data exception, not evidence for an automatic dispatch
threshold. `AMALTHEA_NATIVE_GPU=auto` remains false for modal.

## Verification rationale

The direct fixed-node check uses nonsymmetric mode coefficients, distinct mode
orders, nontrivial `ϕ`, and nonzero energy in both polarization components. It
compares the CUDA point evaluator with CPU native at the same supplied nodes,
so it tests mode synthesis, Bessel order, polarization selectors, transform
scaling, Kerr cross-coupling, normalization, projection, and Jacobian
independently of libcubature's adaptive decisions. The expected single-stage
target is the modal method tier (`~1e-10`) because Rust's Bessel evaluator is
not required to be bit-identical to Julia's `SpecialFunctions.besselj`; the
fixed-step trajectory target is `<1e-6`.

The adaptive tests cover one mode and two modes, both cubature branches,
`npol=1` and genuinely energized `npol=2`, plus a nonzero HE11→HE12 transfer
control. A rejected RK trial must preserve the resident field bitwise and a
retry must agree with CPU native/Julia. The Julia Kerr-on/Kerr-off control must
exceed the asserted comparison tolerance by at least two orders of magnitude.

## Implementation

1. Implement transactional CUDA `set_modal_params`, including mode metadata,
   polarization selectors, normalization factors, nonlinear prefactors,
   dimensions, and c2r plans/scratch.
2. Preserve Julia column-major mode/time/polarization layout. Add literal
   layout tests using nonsymmetric mode and polarization data.
3. Implement device kernels for mode-field synthesis at `(r,theta)`, inverse
   FFT, scalar/vector Kerr, window/normalization, forward FFT, and modal
   projection including the polar Jacobian.
4. Use stable Bessel formulas matching the CPU native path; do not recompute
   normalization or dispersion on CUDA.
5. Support both cubature callbacks (`full=false` radial and `full=true`
   two-dimensional) and both polarization counts. Tests must energize both
   polarizations so vector cross-coupling is non-vacuous.
6. Broaden eligibility only for the implemented constant-radius,
   constant-linop modal Kerr surface. Keep tapered/z-dependent modes,
   Raman/plasma/noise/mixtures, and `StepIndexMode` excluded.
7. Keep `:auto` false.

## Acceptance

- Device point-evaluator vs CPU native at fixed supplied nodes before testing
  adaptive cubature.
- One-mode and two-mode cases, `full=false` and `full=true`, `npol=1` and a
  genuinely two-polarization `npol=2` case.
- Nonzero HE11→HE12 transfer control, direct stage comparison at the modal
  method tier, fixed-step `<1e-6`, reject/retry, and adaptive trajectory.
- Transactional setup/lifecycle failures on real CUDA.
- Record host/device callback bytes and wall time to ensure the design is not
  dominated by transfers; no auto threshold follows from this benchmark.

Run strict CUDA, CPU modal/cubature/threading tests, focused Julia item, Rust
group, and `git diff --check`. Update docs and append `PORT_LOG.md`.

## Verification record (2026-08-04)

Implemented the transactional CUDA modal setup and resident point-evaluator
pipeline in `amalthea/src/cuda_native.rs`, with the four device kernels in
`amalthea/src/kernels.cu` and kernel handles loaded by `amalthea/src/cuda.rs`.
`src/RK45.jl` now admits only this constant-radius RealGrid/Kerr surface under
explicit `AMALTHEA_NATIVE_GPU=on`; `:auto` remains false.

The strict focused CUDA item passed **37/37** on the host RTX 5060 Ti. Fixed
nodes and direct FSAL stages for all four surfaces (`full=false|true`,
`npol=1|2`) matched CPU native at `1.11e-15`–`1.41e-15`. The two-mode
`full=false`, `npol=1` fixed solve matched at `4.07e-16`, transferred
`8.1872e4` host→device bytes and `1.678376e8` device→host bytes over 1,204
device callback batches, and took 0.310 s on GPU versus 0.693 s on CPU for
the recorded case. HE11→HE12 transfer was `8.49e-6`; Kerr on/off was
`2.53e-2`; the rejected adaptive trial preserved state and matched CPU error.
The test records the traffic/wall-time data but intentionally establishes no
automatic dispatch threshold.

## Non-goals

Tapered/z-dependent modes, plasma, Raman, shot noise, mixtures,
`StepIndexMode`, or replacing cubature.

#!/usr/bin/env julia

using Amalthea
import LinearAlgebra: norm

Amalthea.set_fftw_mode(Symbol(get(ENV, "AMALTHEA_AUDIT_FFTW_MODE", "estimate")))
Amalthea.set_fftw_threads(1)

include(joinpath(@__DIR__, "fixtures.jl"))

length(ARGS) in (1, 2) || error("usage: check_fixture.jl FIXTURE_ID [small|medium|large]")
fixture_id = ARGS[1]
size = length(ARGS) == 2 ? Symbol(ARGS[2]) : :small

function relative_error(actual, expected)
    denominator = norm(expected)
    denominator == 0 && return norm(actual - expected)
    norm(actual - expected) / denominator
end

function make_stepper(fixture, backend::Symbol; fixed::Bool=false)
    kwargs = fixed ?
        (; rtol=1e-6, atol=1e-10, max_dt=fixture.dt, min_dt=fixture.dt) :
        (; rtol=1e-6, atol=1e-10)
    if backend === :julia
        return RK45.PreconStepper(
            fixture.transform, fixture.linop, copy(fixture.Eω), 0.0, fixture.dt; kwargs...)
    elseif backend === :rust
        rust_kwargs = hasproperty(fixture, :requires_flength) && fixture.requires_flength ?
            (; kwargs..., flength=fixture.flength) : kwargs
        stepper = withenv("AMALTHEA_NATIVE_GPU" => "off",
                          "AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                          "AMALTHEA_USE_RUST_IONISATION" => "1") do
            RK45.RustNativeStepper(
                fixture.transform, fixture.linop, copy(fixture.Eω), 0.0, fixture.dt;
                rust_kwargs..., native_threads=1)
        end
        RK45._native_backend(stepper) === :cpu || error("fixture selected non-CPU backend")
        return stepper
    end
    error("unknown backend $backend")
end

fixture = apply_audit_overrides!(build_fixture(Amalthea, fixture_id, size))

single_rel = let
    single_julia = make_stepper(fixture, :julia)
    single_rust = make_stepper(fixture, :rust)
    RK45.step!(single_julia)
    RK45.step!(single_rust)
    relative_error(single_rust.yn, single_julia.yn)
end
GC.gc()
single_tolerance = fixture_id == "modeavg_env_raman_sdo" ? 5e-6 :
                   fixture_id == "modal_real_tapered" ? 1e-9 :
                   occursin("raman", fixture_id) ? 2e-7 :
                   occursin("zdependent", fixture_id) ? 1e-9 :
                   fixture.geometry === :modal ? 1e-10 : 1e-12
single_rel < single_tolerance || error(
    "single-step error $single_rel exceeds $single_tolerance")

oracle_field = let
    fixed_julia = make_stepper(fixture, :julia; fixed=true)
    RK45.solve(fixed_julia, fixture.flength)
    copy(fixed_julia.yn)
end
GC.gc()
native_field = let
    fixed_rust = make_stepper(fixture, :rust; fixed=true)
    RK45.solve(fixed_rust, fixture.flength)
    copy(fixed_rust.yn)
end
GC.gc()
solve_rel = relative_error(native_field, oracle_field)
solve_tolerance = fixture_id == "modeavg_real_zdependent" ? 1e-3 :
                  fixture_id == "modeavg_env_raman_sdo" ? 2e-5 :
                  fixture_id == "modal_real_raman_nothg" ? 1.5e-6 : 1e-6
solve_rel < solve_tolerance || error(
    "fixed-solve error $solve_rel exceeds $solve_tolerance")

if occursin("vector", fixture_id)
    # A cylindrically symmetric single mode is rotationally invariant, so a
    # full solve at phi=0 versus phi=pi/4 can be identical even though the
    # vector cross terms are nonzero. Probe the exact Julia-oracle response
    # formula with both components populated and compare it with uncoupled
    # scalar Kerr; this is the discriminating physical effect.
    if fixture.grid_kind === :real
        field = [1.0 2.0; -0.5 0.75]
        coupled = zeros(Base.size(field))
        uncoupled = zeros(Base.size(field))
        Amalthea.Nonlinear.KerrVector!(coupled, field, 1.0)
        for column in axes(field, 2)
            Amalthea.Nonlinear.KerrScalar!(
                view(uncoupled, :, column), view(field, :, column), 1.0)
        end
    else
        field = ComplexF64[1.0+0.5im 2.0-0.25im; -0.5+0.2im 0.75+0.4im]
        coupled = zeros(ComplexF64, Base.size(field))
        uncoupled = zeros(ComplexF64, Base.size(field))
        Amalthea.Nonlinear.KerrVectorEnv!(coupled, field, 1.0)
        for column in axes(field, 2)
            Amalthea.Nonlinear.KerrScalarEnv!(
                view(uncoupled, :, column), view(field, :, column), 1.0)
        end
    end
    effect_rel = relative_error(coupled, uncoupled)
else
    control = apply_audit_overrides!(
        build_fixture(Amalthea, fixture_id, size; feature_enabled=false))
    control_field = let
        control_julia = make_stepper(control, :julia; fixed=true)
        RK45.solve(control_julia, control.flength)
        copy(control_julia.yn)
    end
    effect_rel = relative_error(oracle_field, control_field)
end
if occursin("full", fixture_id)
    fixture.transform.full || error("full-representation fixture did not select full=true")
end
effect_tolerance = occursin("shotnoise", fixture_id) || occursin("full", fixture_id) ?
                   0.0 : 1e-8
effect_rel > effect_tolerance || error(
    "feature effect $effect_rel is not non-vacuous above $effect_tolerance")

println("fixture=", fixture_id)
println("size=", size)
println("field_length=", length(fixture.Eω))
println("single_step_relative_error=", single_rel)
println("single_step_tolerance=", single_tolerance)
println("fixed_solve_relative_error=", solve_rel)
println("fixed_solve_tolerance=", solve_tolerance)
println("feature_effect_relative=", effect_rel)
println("feature_effect_tolerance=", effect_tolerance)
println("backend=cpu")

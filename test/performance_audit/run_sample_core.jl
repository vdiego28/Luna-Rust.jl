module PerformanceAuditSampleCore

using Amalthea
import LinearAlgebra: norm

include(joinpath(@__DIR__, "fixtures.jl"))

const SUPPORTED_MEASUREMENTS = (
    "setup", "field_sync", "fixed_rhs", "fixed_step", "fixed_solve",
    "fixed_solve_raw", "adaptive_solve", "dense_output", "result_copy",
)

function json_string(value::AbstractString)
    "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
end

function run_one(
    fixture_id::AbstractString,
    size_text::AbstractString,
    backend_text::AbstractString,
    measurement::AbstractString,
    output_path::AbstractString;
    warmups::Integer=2,
    process_mode::AbstractString="fresh",
)
    size = Symbol(size_text)
    backend = Symbol(backend_text)
    backend in (:julia, :rust) || error("backend must be julia or rust")
    measurement in SUPPORTED_MEASUREMENTS ||
        error("unsupported measurement $measurement")
    warmups >= 0 || error("warmups must be nonnegative")

    function make_stepper(fixture; fixed::Bool=false)
        kwargs = fixed ?
            (; rtol=1e-6, atol=1e-10, max_dt=fixture.dt, min_dt=fixture.dt) :
            (; rtol=1e-6, atol=1e-10)
        if backend === :julia
            RK45.PreconStepper(
                fixture.transform, fixture.linop, copy(fixture.Eω), 0.0,
                fixture.dt; kwargs...)
        else
            rust_kwargs = hasproperty(fixture, :requires_flength) &&
                          fixture.requires_flength ?
                (; kwargs..., flength=fixture.flength) : kwargs
            stepper = withenv(
                "AMALTHEA_NATIVE_GPU" => "off",
                "AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                "AMALTHEA_USE_RUST_IONISATION" => "1",
            ) do
                RK45.RustNativeStepper(
                    fixture.transform, fixture.linop, copy(fixture.Eω), 0.0,
                    fixture.dt; rust_kwargs..., native_threads=Threads.nthreads())
            end
            RK45._native_backend(stepper) === :cpu ||
                error("non-CPU backend selected")
            stepper
        end
    end

    function setup_backend(; fixed::Bool=false)
        fixture = apply_audit_overrides!(
            build_fixture(Amalthea, fixture_id, size))
        stepper = make_stepper(fixture; fixed)
        fixture, stepper
    end

    function counted_solve!(stepper, finish)
        accepted = 0
        rejected = 0
        while stepper.tn <= finish
            if RK45.step!(stepper)
                accepted += 1
            else
                rejected += 1
            end
        end
        accepted, rejected
    end

    function calibrated_repetitions(operation!)
        if haskey(ENV, "AMALTHEA_AUDIT_REPETITIONS")
            return parse(Int, ENV["AMALTHEA_AUDIT_REPETITIONS"])
        end
        target_seconds = parse(Float64, get(
            ENV, "AMALTHEA_AUDIT_TARGET_SECONDS", "0.02"))
        probe_seconds = @elapsed operation!()
        calibration_seconds = min(target_seconds / 10, 0.002)
        if probe_seconds < calibration_seconds
            calibration_repetitions = clamp(
                ceil(Int, calibration_seconds / max(probe_seconds, 1e-9)),
                1, 100_000)
            probe_seconds = @elapsed for _ in 1:calibration_repetitions
                operation!()
            end
            probe_seconds /= calibration_repetitions
        end
        estimate = ceil(Int, target_seconds / max(probe_seconds, 1e-9))
        clamp(estimate, 1, 1_000_000)
    end

    function execute()
        if measurement == "setup"
            timed = @timed setup_backend()
            fixture, stepper = timed.value
            return timed, fixture, stepper, 0, 0, 1
        end

        fixed = measurement in ("fixed_step", "fixed_solve", "fixed_solve_raw")
        fixture, stepper = setup_backend(; fixed)
        if measurement in ("field_sync", "fixed_rhs")
            operation! = if backend === :julia
                if measurement == "fixed_rhs"
                    () -> begin
                        stepper.fbar!(
                            stepper.ks[1], stepper.yn, stepper.t, stepper.t)
                    end
                else
                    () -> begin
                        copyto!(stepper.yn, fixture.Eω)
                    end
                end
            else
                if measurement == "fixed_rhs"
                    () -> begin
                        GC.@preserve stepper begin
                            rc = ccall(
                                (:set_field, RK45._LIBAMALTHEA_RK45), Cint,
                                (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                                stepper._handle.ptr, pointer(stepper.yn),
                                length(stepper.yn))
                            RK45.check_ffi(rc, "set_field")
                        end
                    end
                else
                    () -> begin
                        GC.@preserve stepper begin
                            rc = ccall(
                                (:native_resync_field, RK45._LIBAMALTHEA_RK45),
                                Cint,
                                (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                                stepper._handle.ptr, pointer(stepper.yn),
                                length(stepper.yn))
                            RK45.check_ffi(rc, "native_resync_field")
                        end
                    end
                end
            end
            repetitions = calibrated_repetitions(operation!)
            timed = @timed for _ in 1:repetitions
                operation!()
            end
            return timed, fixture, stepper, 0, 0, repetitions
        end
        if measurement == "fixed_step"
            repetitions = parse(Int, get(
                ENV, "AMALTHEA_AUDIT_REPETITIONS", "1"))
            timed = @timed begin
                accepted = 0
                rejected = 0
                for _ in 1:repetitions
                    if RK45.step!(stepper)
                        accepted += 1
                    else
                        rejected += 1
                    end
                end
                accepted, rejected
            end
            accepted, rejected = timed.value
            return timed, fixture, stepper, accepted, rejected, repetitions
        end
        if measurement in ("dense_output", "result_copy")
            accepted, rejected = counted_solve!(stepper, fixture.flength)
            result = measurement == "result_copy" ?
                RK45.interpolate(stepper, fixture.flength) : nothing
            destination = measurement == "result_copy" ? similar(result) : nothing
            operation! = measurement == "dense_output" ?
                (() -> RK45.interpolate(stepper, fixture.flength)) :
                (() -> copyto!(destination, result))
            repetitions = calibrated_repetitions(operation!)
            timed = @timed begin
                value = nothing
                for _ in 1:repetitions
                    value = operation!()
                end
                value
            end
            return timed, fixture, stepper, accepted, rejected, repetitions
        end
        timed = @timed counted_solve!(stepper, fixture.flength)
        accepted, rejected = timed.value
        timed, fixture, stepper, accepted, rejected, 1
    end

    # A persistent session cycles through unrelated, potentially multi-GiB
    # fixtures.  Collect the previous request before allocating this one and
    # collect each discarded warmup so NativeSim finalizers and FFT plans do
    # not accumulate until the process is OOM-killed.
    function discard_warmup!()
        warmup_result = execute()
        warmup_stepper = warmup_result[3]
        if backend === :rust
            finalize(warmup_stepper._handle)
        end
        nothing
    end

    GC.gc(true)
    for _ in 1:warmups
        discard_warmup!()
        GC.gc(true)
    end
    timed, fixture, stepper, accepted, rejected, repetitions = execute()

    field = if measurement == "fixed_rhs"
        if backend === :julia
            copy(stepper.ks[1])
        else
            rhs = similar(stepper.yn)
            GC.@preserve stepper rhs begin
                rc = ccall(
                    (:get_ks_stage, RK45._LIBAMALTHEA_RK45), Cint,
                    (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                    stepper._handle.ptr, 0, pointer(rhs), length(rhs))
                RK45.check_ffi(rc, "get_ks_stage")
            end
            rhs
        end
    elseif measurement in ("dense_output", "result_copy")
        timed.value
    elseif measurement in ("fixed_solve", "adaptive_solve")
        RK45.interpolate(stepper, fixture.flength)
    else
        stepper.yn
    end

    mkpath(dirname(output_path))
    field_path = output_path * ".field.bin"
    open(field_path, "w") do io
        write(io, reinterpret(Float64, vec(field)))
    end

    pairs = [
        "schema_version" => "2",
        "fixture" => json_string(fixture_id),
        "size" => json_string(size_text),
        "backend" => json_string(backend_text),
        "measurement" => json_string(measurement),
        "process_mode" => json_string(process_mode),
        "warmups_performed" => string(warmups),
        "elapsed_seconds" => repr(timed.time / repetitions),
        "allocated_bytes" => repr(timed.bytes / repetitions),
        "gc_seconds" => repr(timed.gctime / repetitions),
        "measurement_repetitions" => string(repetitions),
        "peak_rss_bytes" => string(Sys.maxrss()),
        "accepted_steps" => string(accepted),
        "rejected_steps" => string(rejected),
        "derived_rhs_evaluations" => string(
            accepted + rejected == 0 ? 0 : 1 + 6 * (accepted + rejected)),
        "field_length" => string(length(field)),
        "field_shape" => "[" * join(Base.size(field), ",") * "]",
        "input_field_shape" => "[" * join(Base.size(fixture.Eω), ",") * "]",
        "n_time" => string(length(fixture.grid.t)),
        "n_time_oversampled" => string(length(fixture.grid.to)),
        "n_spectral" => string(length(fixture.grid.ω)),
        "field_norm" => repr(norm(field)),
        "final_t" => repr(stepper.t),
        "final_tn" => repr(stepper.tn),
        "field_path" => json_string(field_path),
        "julia_threads" => string(Threads.nthreads()),
        "native_threads" => string(backend === :rust ? Threads.nthreads() : 0),
    ]
    json = "{\n" *
           join(("  \"$key\": $value" for (key, value) in pairs), ",\n") *
           "\n}\n"
    open(output_path, "w") do io
        write(io, json)
    end
    if backend === :rust
        finalize(stepper._handle)
    end
    timed = fixture = stepper = field = nothing
    GC.gc(true)
    output_path
end

end # module

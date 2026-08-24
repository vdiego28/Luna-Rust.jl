module PerformanceAuditUpstreamSampleCore

using Luna
import LinearAlgebra: norm

include(joinpath(@__DIR__, "fixtures.jl"))

const SUPPORTED_MEASUREMENTS = (
    "setup", "field_sync", "fixed_rhs", "fixed_step", "fixed_solve",
    "fixed_solve_raw", "adaptive_solve", "dense_output", "result_copy",
)

json_string(value::AbstractString) =
    "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""

function run_one(
    fixture_id::AbstractString,
    size_text::AbstractString,
    measurement::AbstractString,
    output_path::AbstractString;
    warmups::Integer=2,
    process_mode::AbstractString="fresh",
)
    size = Symbol(size_text)
    measurement in SUPPORTED_MEASUREMENTS ||
        error("unsupported measurement $measurement")
    warmups >= 0 || error("warmups must be nonnegative")

    function setup_upstream(; fixed::Bool=false)
        fixture = apply_audit_overrides!(build_fixture(Luna, fixture_id, size))
        kwargs = fixed ?
            (; rtol=1e-6, atol=1e-10, max_dt=fixture.dt, min_dt=fixture.dt) :
            (; rtol=1e-6, atol=1e-10)
        stepper = Luna.RK45.PreconStepper(
            fixture.transform, fixture.linop, copy(fixture.Eω), 0.0,
            fixture.dt; kwargs...)
        fixture, stepper
    end

    function counted_solve!(stepper, finish)
        accepted = 0
        rejected = 0
        while stepper.tn <= finish
            if Luna.RK45.step!(stepper)
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
            timed = @timed setup_upstream()
            fixture, stepper = timed.value
            return timed, fixture, stepper, 0, 0, 1
        end
        fixed = measurement in ("fixed_step", "fixed_solve", "fixed_solve_raw")
        fixture, stepper = setup_upstream(; fixed)
        if measurement in ("field_sync", "fixed_rhs")
            operation! = if measurement == "fixed_rhs"
                () -> begin
                    stepper.fbar!(
                        stepper.ks[1], stepper.yn, stepper.t, stepper.t)
                end
            else
                () -> begin
                    copyto!(stepper.yn, fixture.Eω)
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
                    if Luna.RK45.step!(stepper)
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
                Luna.RK45.interpolate(stepper, fixture.flength) : nothing
            destination = measurement == "result_copy" ? similar(result) : nothing
            operation! = measurement == "dense_output" ?
                (() -> Luna.RK45.interpolate(stepper, fixture.flength)) :
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

    function discard_warmup!()
        execute()
        nothing
    end

    GC.gc(true)
    for _ in 1:warmups
        discard_warmup!()
        GC.gc(true)
    end
    timed, fixture, stepper, accepted, rejected, repetitions = execute()
    field = measurement == "fixed_rhs" ? copy(stepper.ks[1]) :
            measurement in ("dense_output", "result_copy") ? timed.value :
            measurement in ("fixed_solve", "adaptive_solve") ?
            Luna.RK45.interpolate(stepper, fixture.flength) : stepper.yn

    mkpath(dirname(output_path))
    field_path = output_path * ".field.bin"
    open(field_path, "w") do io
        write(io, reinterpret(Float64, vec(field)))
    end

    pairs = [
        "schema_version" => "2",
        "fixture" => json_string(fixture_id),
        "size" => json_string(size_text),
        "backend" => json_string("upstream"),
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
    ]
    json = "{\n" *
           join(("  \"$key\": $value" for (key, value) in pairs), ",\n") *
           "\n}\n"
    open(output_path, "w") do io
        write(io, json)
    end
    timed = fixture = stepper = field = nothing
    GC.gc(true)
    output_path
end

end # module

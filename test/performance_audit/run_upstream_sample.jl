#!/usr/bin/env julia

using Luna

include(joinpath(@__DIR__, "run_upstream_sample_core.jl"))
using .PerformanceAuditUpstreamSampleCore

length(ARGS) == 4 || error(
    "usage: run_upstream_sample.jl FIXTURE SIZE MEASUREMENT OUTPUT_JSON")
fixture_id, size_text, measurement, output_path = ARGS

Luna.set_fftw_mode(Symbol(get(ENV, "AMALTHEA_AUDIT_FFTW_MODE", "estimate")))
Luna.set_fftw_threads(1)
warmups = parse(Int, get(ENV, "AMALTHEA_AUDIT_WARMUPS", "2"))
PerformanceAuditUpstreamSampleCore.run_one(
    fixture_id, size_text, measurement, output_path;
    warmups, process_mode="fresh")
println(output_path)

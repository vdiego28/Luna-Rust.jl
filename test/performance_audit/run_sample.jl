#!/usr/bin/env julia

using Amalthea

include(joinpath(@__DIR__, "run_sample_core.jl"))
using .PerformanceAuditSampleCore

length(ARGS) == 5 || error(
    "usage: run_sample.jl FIXTURE SIZE BACKEND MEASUREMENT OUTPUT_JSON")
fixture_id, size_text, backend_text, measurement, output_path = ARGS

Amalthea.set_fftw_mode(Symbol(get(
    ENV, "AMALTHEA_AUDIT_FFTW_MODE", "estimate")))
Amalthea.set_fftw_threads(1)
warmups = parse(Int, get(ENV, "AMALTHEA_AUDIT_WARMUPS", "2"))
PerformanceAuditSampleCore.run_one(
    fixture_id, size_text, backend_text, measurement, output_path;
    warmups, process_mode="fresh")
println(output_path)

#!/usr/bin/env julia

using Amalthea

include(joinpath(@__DIR__, "run_sample_core.jl"))
using .PerformanceAuditSampleCore

length(ARGS) == 1 || error("usage: run_sample_session.jl BACKEND")
backend_text = only(ARGS)
backend_text in ("julia", "rust") || error("backend must be julia or rust")

Amalthea.set_fftw_mode(Symbol(get(
    ENV, "AMALTHEA_AUDIT_FFTW_MODE", "estimate")))
Amalthea.set_fftw_threads(1)
warmed = Set{Tuple{String, String, String}}()

println("__AUDIT_READY__")
flush(stdout)
for line in eachline(stdin)
    parts = split(chomp(line), '\t'; keepempty=true)
    if length(parts) != 4
        println("__AUDIT_ERROR__\tinvalid request")
        flush(stdout)
        continue
    end
    fixture_id, size_text, measurement, output_path = parts
    key = (fixture_id, size_text, measurement)
    warmups = key in warmed ? 0 : 2
    try
        PerformanceAuditSampleCore.run_one(
            fixture_id, size_text, backend_text, measurement, output_path;
            warmups, process_mode="persistent")
        push!(warmed, key)
        println("__AUDIT_OK__\t", output_path)
    catch error
        showerror(stderr, error, catch_backtrace())
        println(stderr)
        println("__AUDIT_ERROR__\t", output_path)
    end
    flush(stdout)
    flush(stderr)
end

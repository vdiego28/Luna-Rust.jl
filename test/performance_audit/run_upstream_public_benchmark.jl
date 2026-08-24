#!/usr/bin/env julia

using Luna
import LinearAlgebra: norm
import Logging: NullLogger, with_logger
import Random: MersenneTwister

Luna.set_fftw_mode(Symbol(get(ENV, "AMALTHEA_AUDIT_FFTW_MODE", "estimate")))
Luna.set_fftw_threads(1)

length(ARGS) == 2 || error(
    "usage: run_upstream_public_benchmark.jl phase_c_reconstruction|readme_v103 OUTPUT_JSON")
config_name, output_path = ARGS

const CONFIGS = Dict(
    "phase_c_reconstruction" => (
        radius=125e-6, flength=0.15, gas=:He, pressure=1.0,
        λ0=800e-9, λlims=(200e-9, 4e-6), trange=1e-12,
        energy=1e-6, τfwhm=30e-15, saveN=50,
    ),
    "readme_v103" => (
        radius=125e-6, flength=3.0, gas=:He, pressure=1.0,
        λ0=800e-9, λlims=(150e-9, 4e-6), trange=1e-12,
        energy=120e-6, τfwhm=10e-15, saveN=2,
    ),
)
haskey(CONFIGS, config_name) || error("unknown config $config_name")
config = CONFIGS[config_name]

function public_call()
    with_logger(NullLogger()) do
        Luna.prop_capillary(
            config.radius, config.flength, config.gas, config.pressure;
            config.λ0, config.λlims, config.trange, config.energy,
            config.τfwhm, config.saveN, raman=false, plasma=true,
            kerr=true, shotnoise=false, rng=MersenneTwister(0),
            status_period=Inf,
        )
    end
end

public_call()
public_call()
GC.gc()
timed = @timed public_call()
field = timed.value["Eω"][:, end]
field_path = output_path * ".field.bin"
mkpath(dirname(output_path))
open(field_path, "w") do io
    write(io, reinterpret(Float64, vec(field)))
end

json_string(value::AbstractString) =
    "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
pairs = [
    "schema_version" => "1",
    "config" => json_string(config_name),
    "backend" => json_string("upstream"),
    "elapsed_seconds" => repr(timed.time),
    "allocated_bytes" => string(timed.bytes),
    "gc_seconds" => repr(timed.gctime),
    "peak_rss_bytes" => string(Sys.maxrss()),
    "field_length" => string(length(field)),
    "field_norm" => repr(norm(field)),
    "field_path" => json_string(field_path),
    "julia_threads" => string(Threads.nthreads()),
]
open(output_path, "w") do io
    write(io, "{\n" * join(("  \"$key\": $value" for (key, value) in pairs), ",\n") * "\n}\n")
end
println(output_path)

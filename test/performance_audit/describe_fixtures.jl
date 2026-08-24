#!/usr/bin/env julia

using Amalthea
using TOML

Amalthea.set_fftw_mode(:estimate)
Amalthea.set_fftw_threads(1)

include(joinpath(@__DIR__, "fixtures.jl"))

inventory = TOML.parsefile(joinpath(@__DIR__, "workloads.toml"))["fixture"]
output_path = length(ARGS) >= 2 && ARGS[1] == "--output" ? ARGS[2] : nothing
resume = "--resume" in ARGS
size_args = isnothing(output_path) ? filter(!=("--resume"), ARGS) : filter(!=("--resume"), ARGS[3:end])
sizes = isempty(size_args) ? AUDIT_SIZES : Tuple(Symbol(arg) for arg in size_args)

function json_string(value::AbstractString)
    "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
end

function describe(fixture, size)
    pairs = [
        "fixture" => json_string(fixture.id),
        "size" => json_string(String(size)),
        "field_shape" => "[" * join(Base.size(fixture.Eω), ",") * "]",
        "field_length" => string(length(fixture.Eω)),
        "n_time" => string(length(fixture.grid.t)),
        "n_time_oversampled" => string(length(fixture.grid.to)),
        "n_spectral" => string(length(fixture.grid.ω)),
    ]
    "{" * join(("\"$key\":$value" for (key, value) in pairs), ",") * "}"
end

completed = Set{Tuple{String,Symbol}}()
if resume && !isnothing(output_path) && isfile(output_path)
    for line in eachline(output_path)
        fixture_match = match(r"\"fixture\":\"([^\"]+)\"", line)
        size_match = match(r"\"size\":\"([^\"]+)\"", line)
        if !isnothing(fixture_match) && !isnothing(size_match)
            push!(completed, (fixture_match.captures[1], Symbol(size_match.captures[1])))
        end
    end
end

io = isnothing(output_path) ? stdout : open(output_path, resume ? "a" : "w")
try
    for size in sizes, item in inventory
        (item["id"], size) in completed && continue
        fixture = build_fixture(Amalthea, item["id"], size)
        println(io, describe(fixture, size))
        flush(io)
        fixture = nothing
        GC.gc()
    end
finally
    io === stdout || close(io)
end

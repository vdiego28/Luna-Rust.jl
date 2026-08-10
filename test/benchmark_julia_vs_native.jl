# Reproducible backend comparison for the public README.
#
# This deliberately compares Amalthea's retained Luna-compatible Julia oracle
# with its resident native CPU backend in one checkout. It is not an
# independently installed latest-Luna.jl comparison: keeping package version,
# physics setup, dependencies, and output code identical isolates the backend
# cost. Run from the repository root with:
#
#     julia --project test/benchmark_julia_vs_native.jl
#
# Set AMALTHEA_BENCHMARK_TRIALS to change the default three timed repetitions.

using Amalthea
import Amalthea: set_fftw_mode
import LinearAlgebra: norm
import Logging: NullLogger, with_logger
import Random: MersenneTwister
import Statistics: median

set_fftw_mode(:estimate)

const TRIALS = parse(Int, get(ENV, "AMALTHEA_BENCHMARK_TRIALS", "3"))
TRIALS > 0 || error("AMALTHEA_BENCHMARK_TRIALS must be a positive integer")

const WORKLOAD = (
    radius = 125e-6,
    flength = 3.0,
    gas = :He,
    pressure = 1.0,
    λ0 = 800e-9,
    λlims = (150e-9, 4e-6),
    trange = 1e-12,
    energy = 120e-6,
    τfwhm = 10e-15,
    saveN = 2,
    raman = false,
    plasma = true,
    kerr = true,
    shotnoise = false,
)

function run_backend(native::Bool)
    toggle = native ? "1" : "0"
    GC.gc()
    result = withenv(
        "AMALTHEA_USE_RUST_NATIVE" => toggle,
        "AMALTHEA_NATIVE_GPU" => "off",
        "AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
    ) do
        with_logger(NullLogger()) do
            @timed prop_capillary(
                WORKLOAD.radius,
                WORKLOAD.flength,
                WORKLOAD.gas,
                WORKLOAD.pressure;
                WORKLOAD.λ0,
                WORKLOAD.λlims,
                WORKLOAD.trange,
                WORKLOAD.energy,
                WORKLOAD.τfwhm,
                WORKLOAD.saveN,
                WORKLOAD.raman,
                WORKLOAD.plasma,
                WORKLOAD.kerr,
                WORKLOAD.shotnoise,
                rng = MersenneTwister(0),
                status_period = Inf,
            )
        end
    end

    expected = native ? RK45.RustNativeStepper : RK45.PreconStepper
    RK45._LAST_STEPPER_TYPE[] <: expected || error(
        "Requested $(native ? "native" : "Julia") path selected " *
        "$(RK45._LAST_STEPPER_TYPE[]) instead of $expected",
    )
    return result.time, result.value["Eω"][:, end]
end

# Warm both implementations so JIT compilation and first-use initialization
# are excluded from the timed samples.
run_backend(false)
run_backend(true)

julia_times = Float64[]
native_times = Float64[]
julia_field = Ref{Any}(nothing)
native_field = Ref{Any}(nothing)
for _ in 1:TRIALS
    julia_time, field = run_backend(false)
    julia_field[] = field
    native_time, field = run_backend(true)
    native_field[] = field
    push!(julia_times, julia_time)
    push!(native_times, native_time)
end

julia_s = median(julia_times)
native_s = median(native_times)
speedup = julia_s / native_s
relative_error = norm(native_field[] - julia_field[]) / norm(julia_field[])
relative_error < 1e-6 || error(
    "Backend outputs differ by $relative_error, outside the 1e-6 full-solve tier",
)

function git_revision()
    try
        return readchomp(`git rev-parse --short HEAD`)
    catch
        return "unknown"
    end
end

println("Amalthea Julia-oracle vs resident-native CPU benchmark")
println("Scope: same Amalthea checkout; not an independently installed Luna.jl release")
println("Revision: ", git_revision())
println("Julia: ", VERSION)
println("Host: ", Sys.KERNEL, " ", Sys.ARCH, ", CPU=", Sys.CPU_NAME,
        ", Julia threads=", Threads.nthreads())
println("Workload: 125 μm × 3 m He HCF, 1 bar, 800 nm, 10 fs, 120 μJ, ",
        "RealGrid, Kerr+PPT plasma, Raman/shot noise off")
println("Statistic: median of $TRIALS complete warmed solves")
println("Julia oracle samples (s): ", join(round.(julia_times; digits=6), ", "))
println("Native CPU samples (s): ", join(round.(native_times; digits=6), ", "))
println("Julia oracle median (s): ", round(julia_s; digits=6))
println("Native CPU median (s): ", round(native_s; digits=6))
println("Speedup: ", round(speedup; digits=3), "×")
println("Final-field relative error: ", relative_error)

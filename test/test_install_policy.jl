using TestItems

@testitem "Native installer platform policy" tags=[:rust] begin
    import Test: @test, @testset

    include(joinpath(@__DIR__, "..", "deps", "build_platforms.jl"))

    @testset "published release triples" begin
        @test _target_triple(:Linux, :x86_64) == "x86_64-unknown-linux-gnu"
        @test _target_triple(:Linux, :aarch64) == "aarch64-unknown-linux-gnu"
        @test _target_triple(:Darwin, :aarch64) == "aarch64-apple-darwin"
        @test _target_triple(:NT, :x86_64) == "x86_64-pc-windows-msvc"
    end

    @testset "unsupported hosts fall back to source" begin
        @test _target_triple(:Darwin, :x86_64) === nothing
        @test _target_triple(:NT, :aarch64) === nothing
        @test _target_triple(:Linux, :armv7l) === nothing
        @test _target_triple(:FreeBSD, :x86_64) === nothing
    end

    @test _target_triple() == _target_triple(Sys.KERNEL, Sys.ARCH)

    @testset "CPU-only prebuilt selection" begin
        @test _cpu_prebuilt_allowed("off", "0")
        @test _cpu_prebuilt_allowed(" OFF ", "0")
        @test !_cpu_prebuilt_allowed("auto", "0")
        @test !_cpu_prebuilt_allowed("required", "0")
        @test !_cpu_prebuilt_allowed("off", "1")
        @test !_cpu_prebuilt_allowed("invalid", "0")
    end
end

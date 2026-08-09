using TestItems

@testitem "Native-Rust CUDA modal EnvGrid Kerr (Plan 15)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, Fields, LinearOps, PhysData, Nonlinear, Capillary, Modes
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import LinearAlgebra: norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA modal EnvGrid tests require the Rust library")
        @test_skip "Rust library not found"
    else
        gas = :Ar
        pressure = 1.0
        λ0 = 800e-9
        τfwhm = 20e-15
        radius = 125e-6
        flength = 0.05
        dt = 0.001
        grid = Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), 0.5e-12)
        density = PhysData.density(gas, pressure)
        densityfun(z) = density
        γ3 = PhysData.γ3_gas(gas)

        function make_case(; full=false, npol=1, two_modes=true, γ=γ3, energy=5e-6)
            modes = if npol == 2
                (Capillary.MarcatiliMode(radius, gas, pressure;
                                          kind=:HE, n=1, m=1, ϕ=π/4),)
            elseif two_modes
                (Capillary.MarcatiliMode(radius, gas, pressure;
                                          kind=:HE, n=1, m=1),
                 Capillary.MarcatiliMode(radius, gas, pressure;
                                          kind=:HE, n=1, m=2))
            else
                (Capillary.MarcatiliMode(radius, gas, pressure;
                                          kind=:HE, n=1, m=1),)
            end
            responses = (Nonlinear.Kerr_env(γ),)
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, responses, input, modes, npol == 2 ? :xy : :y;
                               full)
            end
            # Keep both c2c halves populated and make the complex data
            # intentionally asymmetric. This prevents a RealGrid-style
            # half-spectrum shortcut from passing by accident.
            Eω = copy(Eω)
            for m in axes(Eω, 2), i in axes(Eω, 1)
                Eω[i, m] *= cis(0.11 + 0.00037 * i + 0.17 * m)
            end
            (; Eω, transform, linop, modes, npol, full, n=length(Eω),
               n_spec=size(Eω, 1), n_modes=size(Eω, 2))
        end

        function make_stepper(c; gpu=false, field=c.Eω, time=dt, max_dt=time, min_dt=time)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => gpu ? "1" : "0",
                    "AMALTHEA_NATIVE_GPU" => gpu ? "on" : "off") do
                RustNativeStepper(c.transform, c.linop, copy(field), 0.0, time;
                                  rtol=1e-6, atol=1e-10,
                                  max_dt=max_dt, min_dt=min_dt)
            end
        end

        function get_stage(s, c, idx=0)
            out = zeros(ComplexF64, c.n)
            rc = ccall((:get_ks_stage, libpath), Cint,
                       (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                       s._handle.ptr, Csize_t(idx), out, Csize_t(c.n))
            rc == 0 || error("get_ks_stage failed with rc=$rc")
            out
        end

        function modal_nodes(s, c, coords)
            fdim = 2 * c.n_spec * c.n_modes
            npt = c.full ? length(coords) ÷ 2 : length(coords)
            out = zeros(Float64, npt * fdim)
            rc = ccall((:native_debug_modal_eval_nodes, libpath), Cint,
                       (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ptr{Float64}, Csize_t),
                       s._handle.ptr, coords, Csize_t(npt), out, Csize_t(length(out)))
            rc == 0 || error("native_debug_modal_eval_nodes failed with rc=$rc")
            out
        end

        cases = [make_case(full=false, npol=1, two_modes=true),
                 make_case(full=true, npol=1, two_modes=true),
                 make_case(full=false, npol=2),
                 make_case(full=true, npol=2)]

        @testset "Eligibility and explicit dispatch" begin
            for c in cases
                @test RK45._gpu_kernel_supports(c.transform, c.linop)
                withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                        "AMALTHEA_NATIVE_GPU" => "on") do
                    @test RK45._gpu_native_eligible(c.transform, c.linop, c.n)
                end
                withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                        "AMALTHEA_NATIVE_GPU" => "auto") do
                    @test !RK45._gpu_native_eligible(c.transform, c.linop, c.n)
                end
            end
        end

        probe_error = try
            probe = make_stepper(cases[1]; gpu=true)
            RK45._native_backend(probe) === :cuda ||
                error("explicit modal EnvGrid CUDA request selected $(RK45._native_backend(probe))")
            nothing
        catch err
            err
        end
        if probe_error !== nothing
            require_cuda && error("CUDA modal EnvGrid setup failed: $probe_error")
            @test_skip "CUDA GPU/toolkit not available: $(sprint(showerror, probe_error))"
        else
            @testset "Fixed-node device evaluator and direct stages" begin
                for c in cases
                    s_cpu = make_stepper(c; gpu=false)
                    s_gpu = make_stepper(c; gpu=true)
                    rs = [0.13 * radius, 0.41 * radius, 0.77 * radius]
                    coords = c.full ? reduce(vcat, ([r, θ] for (r, θ) in
                                      zip(rs, (0.2, 0.9, 2.1)))) : rs
                    p_cpu = modal_nodes(s_cpu, c, coords)
                    p_gpu = modal_nodes(s_gpu, c, coords)
                    @test norm(p_cpu) > 1e-20
                    point_rel = norm(p_gpu - p_cpu) / norm(p_cpu)
                    println("CUDA modal EnvGrid $(c.full ? "full" : "radial") " *
                            "npol=$(c.npol) point rel: $point_rel")
                    @test point_rel < 5e-8

                    k_cpu = get_stage(s_cpu, c)
                    k_gpu = get_stage(s_gpu, c)
                    @test norm(k_cpu) > 1e-20
                    stage_rel = norm(k_gpu - k_cpu) / norm(k_cpu)
                    println("CUDA modal EnvGrid $(c.full ? "full" : "radial") " *
                            "npol=$(c.npol) stage rel: $stage_rel")
                    @test stage_rel < 5e-8
                end

                # A physical pulse has negligible amplitude in the retained
                # upper c2c half, so phase-only perturbations can hide a bad
                # oversampling map. Populate only that half at full scale.
                c = cases[1]
                half = c.n_spec ÷ 2
                high_only = zeros(ComplexF64, size(c.Eω))
                amp = maximum(abs, c.Eω)
                for m in axes(high_only, 2), i in (half + 1):c.n_spec
                    high_only[i, m] = amp * ((0.4 + 0.013 * i + 0.09 * m) +
                                              im * (-0.3 + 0.007 * i - 0.04 * m))
                end
                @test iszero(norm(@view high_only[1:half, :]))
                @test norm(@view high_only[(half + 1):end, :]) > 0
                high_cpu = make_stepper(c; gpu=false, field=high_only)
                high_gpu = make_stepper(c; gpu=true, field=high_only)
                k_cpu_high = get_stage(high_cpu, c)
                k_gpu_high = get_stage(high_gpu, c)
                @test norm(k_cpu_high) > 1e-20
                high_rel = norm(k_gpu_high - k_cpu_high) / norm(k_cpu_high)
                println("CUDA modal EnvGrid high-half-only stage rel: $high_rel")
                @test high_rel < 5e-8
            end

            @testset "Fixed solve, transfer, and nonlinear non-vacuousness" begin
                c = cases[1]
                s_cpu = make_stepper(c; gpu=false)
                s_gpu = make_stepper(c; gpu=true)
                solve(s_cpu, flength)
                solve(s_gpu, flength)
                solve_rel = norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn)
                println("CUDA modal EnvGrid fixed-solve rel: $solve_rel")
                @test solve_rel < 1e-6

                he12_frac = sum(abs2, s_cpu.yn[:, 2]) / sum(abs2, s_cpu.yn)
                println("CUDA modal EnvGrid HE11→HE12 fraction: $he12_frac")
                @test he12_frac > 1e-6

                c_linear = make_case(full=false, npol=1, two_modes=true, γ=0.0)
                s_nl = PreconStepper(c.transform, c.linop, copy(c.Eω), 0.0, dt;
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                s_linear = PreconStepper(c_linear.transform, c_linear.linop,
                                          copy(c_linear.Eω), 0.0, dt;
                                          rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                solve(s_nl, flength)
                solve(s_linear, flength)
                effect = norm(s_nl.yn - s_linear.yn) / norm(s_linear.yn)
                println("Julia modal EnvGrid Kerr on/off rel: $effect")
                @test effect > 1e-6
            end

            @testset "Rejected adaptive step preserves state and retry agrees" begin
                hot = make_case(full=false, npol=1, two_modes=true, energy=5e-3)
                trial_dt = 0.1
                s_cpu = make_stepper(hot; gpu=false, time=trial_dt,
                                     max_dt=trial_dt, min_dt=0.0)
                s_gpu = make_stepper(hot; gpu=true, time=trial_dt,
                                     max_dt=trial_dt, min_dt=0.0)
                before = copy(s_gpu.yn)
                accepted_cpu = step!(s_cpu)
                accepted_gpu = step!(s_gpu)
                @test accepted_gpu == accepted_cpu == false
                @test s_gpu.yn == before
                @test (isnan(s_gpu.err) && isnan(s_cpu.err)) ||
                      isapprox(s_gpu.err, s_cpu.err; rtol=1e-8)

                adaptive = make_case(full=false, npol=1, two_modes=true, energy=5e-6)
                adaptive_cpu = make_stepper(adaptive; gpu=false, time=0.01,
                                            max_dt=0.01, min_dt=0.0)
                adaptive_gpu = make_stepper(adaptive; gpu=true, time=0.01,
                                            max_dt=0.01, min_dt=0.0)
                solve(adaptive_cpu, 0.005)
                solve(adaptive_gpu, 0.005)
                adaptive_rel = norm(adaptive_gpu.yn - adaptive_cpu.yn) /
                               norm(adaptive_cpu.yn)
                println("CUDA modal EnvGrid adaptive-solve rel: $adaptive_rel")
                @test adaptive_rel < 1e-6
            end
        end
    end
end

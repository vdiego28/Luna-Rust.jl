using TestItems

@testitem "Native-Rust CUDA modal RealGrid SDO Raman (Plan 16)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, Fields, LinearOps, PhysData, Nonlinear, Capillary, Raman
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import LinearAlgebra: norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("Plan 16 CUDA modal Raman tests require the Rust library")
        @test_skip "Rust library not found"
    else
        gas = :N2
        pressure = 1.0
        λ0 = 800e-9
        τfwhm = 20e-15
        radius = 125e-6
        flength = 0.05
        dt = 0.001
        density = PhysData.density(gas, pressure)
        densityfun(z) = density
        γ3 = PhysData.γ3_gas(gas)

        function make_case(; grid_kind=:real, thg=true, rotation=false,
                           vibration=true, full=false, npol=1, energy=5e-6,
                           with_raman=true)
            grid = grid_kind === :real ?
                Grid.RealGrid(flength, λ0, (400e-9, 2000e-9), 0.5e-12) :
                Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), 0.5e-12)
            modes = npol == 2 ?
                (Capillary.MarcatiliMode(radius, gas, pressure;
                                         kind=:HE, n=1, m=1, ϕ=π/4),) :
                (Capillary.MarcatiliMode(radius, gas, pressure;
                                         kind=:HE, n=1, m=1),
                 Capillary.MarcatiliMode(radius, gas, pressure;
                                         kind=:HE, n=1, m=2))
            responses = if !with_raman
                (grid_kind === :real ? Nonlinear.Kerr_field(γ3) :
                                       Nonlinear.Kerr_env(γ3),)
            else
                rr = Raman.raman_response(grid.to, gas;
                                          rotation=rotation, vibration=vibration)
                if grid_kind === :real
                    (Nonlinear.Kerr_field(γ3),
                     Nonlinear.RamanPolarField(grid.to, rr; thg=thg))
                else
                    (Nonlinear.Kerr_env(γ3), Nonlinear.RamanPolarEnv(grid.to, rr))
                end
            end
            linop = LinearOps.make_const_linop(grid, modes, grid.referenceλ)
            input = Fields.GaussField(λ0=λ0, τfwhm=τfwhm, energy=energy)
            components = npol == 2 ? :xy : :y
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, densityfun, responses, input, modes, components;
                               full)
            end
            (; Eω, transform, linop, grid, modes, npol, full,
               n=length(Eω), n_spec=size(Eω, 1), n_modes=size(Eω, 2),
               rotation, vibration, thg, with_raman)
        end

        function make_stepper(c; gpu=false, field=c.Eω, time=dt,
                              max_dt=time, min_dt=time)
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
                       s._handle.ptr, coords, Csize_t(npt), out,
                       Csize_t(length(out)))
            rc == 0 || error("native_debug_modal_eval_nodes failed with rc=$rc")
            out
        end

        vibration_case = make_case(thg=true, rotation=false, vibration=true,
                                   full=false)
        rotational_case = make_case(thg=false, rotation=true, vibration=false,
                                     full=true)

        @testset "Plan 16 eligibility boundaries" begin
            for c in (vibration_case, rotational_case)
                @test c.transform.ts.npol == 1
                @test length(Raman.flatten_sdo_oscillators(c.transform.resp[2].r)) ==
                      (c.rotation ? 49 : 1)
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

            env = make_case(grid_kind=:env, rotation=false, vibration=true)
            @test !RK45._gpu_kernel_supports(env.transform, env.linop)

            vector = make_case(npol=2, rotation=false, vibration=true)
            @test !RK45._gpu_kernel_supports(vector.transform, vector.linop)
        end

        probe_error = try
            probe = make_stepper(vibration_case; gpu=true)
            RK45._native_backend(probe) === :cuda ||
                error("Plan 16 explicit dispatch selected $(RK45._native_backend(probe))")
            nothing
        catch err
            err
        end
        if probe_error !== nothing
            require_cuda && error("Plan 16 CUDA modal Raman setup failed: $probe_error")
            @test_skip "CUDA GPU/toolkit not available: $(sprint(showerror, probe_error))"
        else
            @testset "Distinct-node point and direct-stage agreement" begin
                for c in (vibration_case, rotational_case)
                    s_cpu = make_stepper(c; gpu=false)
                    s_gpu = make_stepper(c; gpu=true)
                    rs = [0.11 * radius, 0.37 * radius, 0.73 * radius, 0.89 * radius]
                    coords = c.full ? reduce(vcat, ([r, θ] for (r, θ) in
                                      zip(rs, (0.2, 0.8, 1.7, 2.6)))) : rs
                    p_cpu = modal_nodes(s_cpu, c, coords)
                    p_gpu = modal_nodes(s_gpu, c, coords)
                    fdim = 2 * c.n_spec * c.n_modes
                    @test norm(p_cpu) > 1e-20
                    @test norm(p_cpu[1:fdim] - p_cpu[(fdim + 1):(2 * fdim)]) > 1e-20
                    point_rel = norm(p_gpu - p_cpu) / norm(p_cpu)
                    println("Plan 16 $(c.rotation ? "rotational" : "vibrational") " *
                            "point rel: $point_rel")
                    @test point_rel < 1e-6

                    k_cpu = get_stage(s_cpu, c)
                    k_gpu = get_stage(s_gpu, c)
                    stage_rel = norm(k_gpu - k_cpu) / norm(k_cpu)
                    println("Plan 16 $(c.rotation ? "rotational" : "vibrational") " *
                            "stage rel: $stage_rel")
                    @test stage_rel < 1e-6
                end
            end

            @testset "Fixed solve and Raman non-vacuity" begin
                # The rotational 49-oscillator case is covered by the direct
                # and stage checks above. Keep the trajectory checks on the
                # one-oscillator case so the strict CUDA regression remains a
                # practical CI test rather than spending tens of minutes in
                # the CPU oracle's adaptive cubature loop.
                c = vibration_case
                s_cpu = make_stepper(c; gpu=false)
                s_gpu = make_stepper(c; gpu=true)
                solve(s_cpu, flength)
                solve(s_gpu, flength)
                solve_rel = norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn)
                println("Plan 16 vibrational fixed-solve rel: $solve_rel")
                @test solve_rel < 1e-6

                no_raman = make_case(thg=c.thg, rotation=c.rotation,
                                     vibration=c.vibration, full=c.full,
                                     with_raman=false)
                s_on = PreconStepper(c.transform, c.linop, copy(c.Eω), 0.0, dt;
                                     rtol=1e-6, atol=1e-10,
                                     max_dt=dt, min_dt=dt)
                s_off = PreconStepper(no_raman.transform, no_raman.linop,
                                      copy(no_raman.Eω), 0.0, dt;
                                      rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                solve(s_on, flength)
                solve(s_off, flength)
                effect = norm(s_on.yn - s_off.yn) / norm(s_off.yn)
                println("Plan 16 vibrational Julia Raman effect: $effect")
                @test effect > 1e-6
            end

            @testset "Rejected state, retry, and adaptive trajectory" begin
                hot = make_case(thg=true, rotation=false, vibration=true,
                                full=false, energy=5e-3)
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

                retried_cpu = step!(s_cpu)
                retried_gpu = step!(s_gpu)
                @test retried_gpu == retried_cpu
                @test norm(s_gpu.yn - s_cpu.yn) / max(norm(s_cpu.yn), 1e-30) < 1e-6

                adaptive = make_case(thg=false, rotation=false, vibration=true,
                                     full=false)
                adaptive_cpu = make_stepper(adaptive; gpu=false, time=0.01,
                                            max_dt=0.01, min_dt=0.0)
                adaptive_gpu = make_stepper(adaptive; gpu=true, time=0.01,
                                            max_dt=0.01, min_dt=0.0)
                solve(adaptive_cpu, 0.005)
                solve(adaptive_gpu, 0.005)
                adaptive_rel = norm(adaptive_gpu.yn - adaptive_cpu.yn) /
                               norm(adaptive_cpu.yn)
                println("Plan 16 adaptive-solve rel: $adaptive_rel")
                @test adaptive_rel < 1e-6
            end
        end
    end
end

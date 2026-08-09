using TestItems

@testitem "Native-Rust CUDA free-space RealGrid SDO Raman" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData,
                     Nonlinear, Raman, Ionisation
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import LinearAlgebra: norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA free-space Raman tests require the Rust library")
        @test_skip "Rust library not found"
    else
        gas = :N2
        pressure = 1.5
        λ0 = 800e-9
        τfwhm = 15e-15
        flength = 0.02
        radius = 1.5e-3
        n_x = 10
        n_y = 8
        dt = 0.002

        function make_free(; rotation=false, vibration=true, thg=true,
                           energy=6e-5)
            grid = Grid.RealGrid(flength, λ0, (150e-9, 2000e-9), 0.5e-12)
            xygrid = Grid.FreeGrid(radius, n_x, radius, n_y)
            density = PhysData.density(gas, pressure)
            densityfun(z) = density
            rr = Raman.raman_response(grid.to, gas;
                                       rotation=rotation, vibration=vibration)
            raman = Nonlinear.RamanPolarField(grid.to, rr; thg=thg)
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), raman)
            normfun = NonlinearRHS.const_norm_free(
                grid, xygrid, PhysData.ref_index_fun(gas, pressure))
            input = Fields.GaussGaussField(λ0=λ0, τfwhm=τfwhm,
                                            energy=energy, w0=150e-6,
                                            propz=-0.1)
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, xygrid, densityfun, normfun,
                               responses, input)
            end
            linop = LinearOps.make_const_linop(
                grid, xygrid, PhysData.ref_index_fun(gas, pressure))
            n_cols = n_x * n_y
            (; grid, xygrid, Eω, transform, linop, density, rr, raman,
               n=length(Eω), n_time=length(grid.t),
               n_time_over=length(grid.to), n_cols,
               n_spec=length(Eω) ÷ n_cols, n_x, n_y)
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

        function set_field(s, field)
            ccall((:set_field, libpath), Cint,
                  (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                  s._handle.ptr, field, Csize_t(length(field)))
        end

        vib = make_free(rotation=false, vibration=true, thg=true)
        rot = make_free(rotation=true, vibration=false, thg=true)
        both = make_free(rotation=true, vibration=true, thg=true)

        @testset "Eligibility, flattening, and rejection boundaries" begin
            @test vib.transform isa Amalthea.NonlinearRHS.TransFree
            @test vib.n_x != vib.n_y
            @test vib.n_cols == n_x * n_y
            @test length(Raman.flatten_sdo_oscillators(vib.rr)) == 1
            @test length(Raman.flatten_sdo_oscillators(rot.rr)) > 1
            @test length(Raman.flatten_sdo_oscillators(both.rr)) <= 64
            @test RK45._gpu_kernel_supports(vib.transform, vib.linop)
            @test RK45._gpu_kernel_supports(rot.transform, rot.linop)
            @test RK45._gpu_kernel_supports(both.transform, both.linop)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_native_eligible(vib.transform, vib.linop, vib.n)
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(vib.transform, vib.linop, vib.n)
            end

            env_raman = Nonlinear.RamanPolarEnv(vib.grid.to, vib.rr)
            env_transform = NonlinearRHS.TransFree(
                vib.grid, vib.xygrid, vib.transform.FT,
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), env_raman),
                z -> vib.density,
                NonlinearRHS.const_norm_free(
                    vib.grid, vib.xygrid,
                    PhysData.ref_index_fun(gas, pressure)))
            @test !RK45._gpu_kernel_supports(env_transform, vib.linop)

            adk = Ionisation.IonRateADK(
                PhysData.ionisation_potential(gas); threshold=true)
            plasma = Nonlinear.PlasmaCumtrapz(
                vib.grid.to, vib.grid.to, adk,
                PhysData.ionisation_potential(gas))
            mixed = NonlinearRHS.TransFree(
                vib.grid, vib.xygrid, vib.transform.FT,
                (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), vib.raman, plasma),
                z -> vib.density,
                NonlinearRHS.const_norm_free(
                    vib.grid, vib.xygrid,
                    PhysData.ref_index_fun(gas, pressure)))
            @test !RK45._gpu_kernel_supports(mixed, vib.linop)
        end

        # Construction is the hardware gate. Dispatch and rejection checks
        # above remain useful on CPU-only hosts; strict mode makes an absent or
        # broken CUDA driver a hard failure instead of a vacuous pass.
        gpu_error = try
            s_probe = make_stepper(vib; gpu=true)
            RK45._native_backend(s_probe) === :cuda ||
                error("explicit CUDA request selected $(RK45._native_backend(s_probe))")
            nothing
        catch err
            err
        end
        if gpu_error !== nothing
            require_cuda && error("CUDA free-space Raman setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available: $(sprint(showerror, gpu_error))"
        else
            @testset "Direct stages: nonsquare columns and N2 oscillator families" begin
                for (name, base) in (("vibration", vib), ("rotation", rot),
                                     ("rotation+vibration", both))
                    for thg in (true, false)
                        c = thg ? base : make_free(
                            rotation=name != "vibration",
                            vibration=name != "rotation", thg=false)
                        s_cpu = make_stepper(c; gpu=false)
                        s_gpu = make_stepper(c; gpu=true)
                        probe = copy(c.Eω)
                        for col in 1:c.n_cols, i in 1:c.n_spec
                            j = (col - 1) * c.n_spec + i
                            probe[j] *= (1 + 0.013 * col + 0.005 * i) +
                                        im * (0.004 * col - 0.001 * i)
                        end
                        @test set_field(s_cpu, probe) == 0
                        @test set_field(s_gpu, probe) == 0
                        k_cpu = get_stage(s_cpu, c)
                        k_gpu = get_stage(s_gpu, c)
                        @test maximum(abs.(k_cpu)) > 1e-12
                        rel = norm(k_gpu - k_cpu) / max(norm(k_cpu), 1e-30)
                        println("CUDA free Raman $name thg=$thg stage rel: $rel")
                        @test rel < 1e-8
                    end
                end
            end

            @testset "Fixed solve and Julia non-vacuity" begin
                for thg in (true, false)
                    c = thg ? vib : make_free(thg=false)
                    s_cpu = make_stepper(c; gpu=false)
                    s_gpu = make_stepper(c; gpu=true)
                    solve(s_cpu, flength)
                    solve(s_gpu, flength)
                    rel = norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn)
                    println("CUDA free Raman thg=$thg fixed-solve rel: $rel")
                    @test rel < 1e-6

                    no_raman = NonlinearRHS.TransFree(
                        c.grid, c.xygrid, c.transform.FT,
                        (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
                        z -> c.density,
                        NonlinearRHS.const_norm_free(
                            c.grid, c.xygrid,
                            PhysData.ref_index_fun(gas, pressure)))
                    s_on = PreconStepper(c.transform, c.linop, copy(c.Eω),
                                         0.0, dt; rtol=1e-6, atol=1e-10,
                                         max_dt=dt, min_dt=dt)
                    s_off = PreconStepper(no_raman, c.linop, copy(c.Eω),
                                          0.0, dt; rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)
                    solve(s_on, flength)
                    solve(s_off, flength)
                    effect = norm(s_on.yn - s_off.yn) / norm(s_off.yn)
                    println("Julia free Raman thg=$thg on/off rel: $effect")
                    @test effect > 1e-6
                end
            end

            @testset "Rejected adaptive step preserves state" begin
                c = make_free(thg=false, energy=6e-3)
                trial_dt = 0.1
                s_cpu = make_stepper(c; gpu=false, time=trial_dt,
                                     max_dt=trial_dt, min_dt=0.0)
                s_gpu = make_stepper(c; gpu=true, time=trial_dt,
                                     max_dt=trial_dt, min_dt=0.0)
                before = copy(s_gpu.yn)
                accepted_cpu = step!(s_cpu)
                accepted_gpu = step!(s_gpu)
                @test accepted_gpu == accepted_cpu == false
                @test s_gpu.yn == before
                @test (isnan(s_gpu.err) && isnan(s_cpu.err)) ||
                      isapprox(s_gpu.err, s_cpu.err; rtol=1e-10)
            end
        end
    end
end

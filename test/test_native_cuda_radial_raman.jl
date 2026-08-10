using TestItems

@testitem "Native-Rust CUDA radial RealGrid SDO Raman" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Raman
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import Hankel
    import LinearAlgebra: I, norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA radial Raman tests require the Rust library")
        @test_skip "Rust library not found"
    else
        gas = :N2
        pressure = 1.5
        λ0 = 800e-9
        τfwhm = 15e-15
        flength = 0.02
        radius = 4e-3
        n_r = 8
        dt = 0.002

        function make_radial(; rotation=false, vibration=true, thg=true,
                             energy=6e-5)
            grid = Grid.RealGrid(flength, λ0, (150e-9, 2000e-9), 0.25e-12)
            q = Hankel.QDHT(radius, n_r, dim=2)
            density = PhysData.density(gas, pressure)
            densityfun(z) = density
            rr = Raman.raman_response(grid.to, gas;
                                      rotation=rotation, vibration=vibration)
            raman = Nonlinear.RamanPolarField(grid.to, rr; thg=thg)
            responses = (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)), raman)
            linop = LinearOps.make_const_linop(
                grid, q, PhysData.ref_index_fun(gas, pressure))
            normfun = NonlinearRHS.const_norm_radial(
                grid, q, PhysData.ref_index_fun(gas, pressure))
            input = Fields.GaussGaussField(λ0=λ0, τfwhm=τfwhm,
                                            energy=energy, w0=150e-6,
                                            propz=-0.1)
            Eω, transform, _ = with_logger(NullLogger()) do
                Amalthea.setup(grid, q, densityfun, normfun, responses, input)
            end
            (; grid, q, Eω, transform, linop, density, raman, rr,
               n_time=length(grid.t), n_time_over=length(grid.to),
               n=nothing)
        end

        function finish_config(c)
            merge(c, (; n=length(c.Eω), n_spec=length(c.Eω) ÷ n_r))
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

        function set_radial(ptr, c, T, sf, si, window, kfac, M)
            ccall((:native_set_radial_params, libpath), Cint,
                  (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Ptr{Float64},
                   Float64, Float64, Ptr{Float64}, Float64,
                   Ptr{Float64}, Ptr{Float64}),
                  ptr, Csize_t(c.n_time), Csize_t(c.n_time_over), Csize_t(n_r),
                  T, sf, si, window, kfac, real.(M), imag.(M))
        end

        function set_raman(ptr, c; thg=true)
            Rs = Raman.flatten_sdo_oscillators(c.rr)
            omegas = Float64[ri.Ω for ri in Rs]
            gammas = Float64[1 / ri.τ2ρ(c.density) for ri in Rs]
            couplings = Float64[ri.K for ri in Rs]
            ccall((:native_set_raman_params, libpath), Cint,
                  (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                   Csize_t, Float64, Float64, Cint),
                  ptr, omegas, gammas, couplings, Csize_t(length(Rs)),
                  c.raman.dt, c.density, Cint(thg))
        end

        vib = finish_config(make_radial(rotation=false, vibration=true))
        rot = finish_config(make_radial(rotation=true, vibration=false))

        @testset "Eligibility and oscillator capacity" begin
            @test vib.transform isa Amalthea.NonlinearRHS.TransRadial
            @test rot.transform isa Amalthea.NonlinearRHS.TransRadial
            @test length(Raman.flatten_sdo_oscillators(vib.rr)) == 1
            @test length(Raman.flatten_sdo_oscillators(rot.rr)) == 49
            @test RK45._gpu_kernel_supports(vib.transform, vib.linop)
            @test RK45._gpu_kernel_supports(rot.transform, rot.linop)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_native_eligible(vib.transform, vib.linop, vib.n)
                @test RK45._gpu_native_eligible(rot.transform, rot.linop, rot.n)
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(vib.transform, vib.linop, vib.n)
                @test !RK45._gpu_native_eligible(rot.transform, rot.linop, rot.n)
            end
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
            require_cuda && error("CUDA radial Raman setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available: $(sprint(showerror, gpu_error))"
        else
            @testset "Direct stages: vibration and N2 rotation, both THG modes" begin
                for (name, base) in (("vibration", vib), ("rotation", rot))
                    for thg in (true, false)
                        c = name == "vibration" && thg ? vib : finish_config(
                            make_radial(rotation=name == "rotation",
                                        vibration=name == "vibration", thg=thg))
                        s_cpu = make_stepper(c; gpu=false)
                        s_gpu = make_stepper(c; gpu=true)
                        rel = norm(get_stage(s_gpu, c) - get_stage(s_cpu, c)) /
                              max(norm(get_stage(s_cpu, c)), 1e-30)
                        println("CUDA radial Raman $name thg=$thg stage rel: $rel")
                        @test rel < 1e-10
                    end
                end
            end

            @testset "Fixed solve and Julia non-vacuity" begin
                for thg in (true, false)
                    c = finish_config(make_radial(thg=thg))
                    s_cpu = make_stepper(c; gpu=false)
                    s_gpu = make_stepper(c; gpu=true)
                    solve(s_cpu, flength)
                    solve(s_gpu, flength)
                    rel = norm(s_gpu.yn - s_cpu.yn) / norm(s_cpu.yn)
                    println("CUDA radial Raman thg=$thg fixed-solve rel: $rel")
                    @test rel < 1e-6

                    no_raman = NonlinearRHS.TransRadial(
                        c.grid, c.q, c.transform.FT,
                        (Nonlinear.Kerr_field(PhysData.γ3_gas(gas)),),
                        z -> c.density,
                        NonlinearRHS.const_norm_radial(
                            c.grid, c.q, PhysData.ref_index_fun(gas, pressure)))
                    s_on = PreconStepper(c.transform, c.linop, copy(c.Eω), 0.0, dt;
                                         rtol=1e-6, atol=1e-10,
                                         max_dt=dt, min_dt=dt)
                    s_off = PreconStepper(no_raman, c.linop, copy(c.Eω), 0.0, dt;
                                          rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)
                    solve(s_on, flength)
                    solve(s_off, flength)
                    effect = norm(s_on.yn - s_off.yn) / norm(s_off.yn)
                    println("Julia radial Raman thg=$thg on/off rel: $effect")
                    @test effect > 1e-6
                end
            end

            @testset "Column isolation" begin
                c = vib
                s_cpu = make_stepper(c; gpu=false)
                s_gpu = make_stepper(c; gpu=true)
                T_identity = Matrix{Float64}(I, n_r, n_r)
                window = ones(Float64, c.n_time_over)
                M = ones(ComplexF64, c.n)
                @test set_radial(s_cpu._handle.ptr, c, T_identity, 1.0, 1.0,
                                 window, 0.0, M) == 0
                @test set_radial(s_gpu._handle.ptr, c, T_identity, 1.0, 1.0,
                                 window, 0.0, M) == 0
                @test set_raman(s_cpu._handle.ptr, c; thg=false) == 0
                @test set_raman(s_gpu._handle.ptr, c; thg=false) == 0
                isolated = zeros(ComplexF64, c.n)
                # Use a deliberately strong single-column DC sentinel: the
                # Raman coupling is tiny in SI units, so a laboratory-scale
                # field would make the isolation assertion vacuous.
                isolated[1] = 2e14
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_cpu._handle.ptr, isolated, Csize_t(c.n)) == 0
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_gpu._handle.ptr, isolated, Csize_t(c.n)) == 0
                k_cpu = reshape(get_stage(s_cpu, c), c.n_spec, n_r)
                k_gpu = reshape(get_stage(s_gpu, c), c.n_spec, n_r)
                @test norm(k_gpu - k_cpu) / max(norm(k_cpu), 1e-30) < 1e-10
                @test maximum(abs.(k_gpu[:, 2:end])) < 1e-12
                @test maximum(abs.(k_gpu[:, 1])) > 1e-12
            end

            @testset "Rejected adaptive step preserves state" begin
                c = finish_config(make_radial(thg=false, energy=6e-3))
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

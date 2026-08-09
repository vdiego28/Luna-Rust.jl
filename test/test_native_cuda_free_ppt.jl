using TestItems

@testitem "Native-Rust CUDA free-space RealGrid PPT plasma (Plan 19)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear, Ionisation
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import LinearAlgebra: norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA Plan 19 tests require the Rust library")
        @test_skip "Rust library not found"
    else
        gas = :Ar; pres = 1.5; τ = 15e-15; λ0 = 800e-9
        w0 = 150e-6; energy = 6e-5; L = 0.02; R = 1.5e-3
        Nx = 10; Ny = 8
        grid = Grid.RealGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
        xygrid = Grid.FreeGrid(R, Nx, R, Ny)
        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        γ3 = PhysData.γ3_gas(gas)
        ionpot = PhysData.ionisation_potential(gas)
        ionrate = withenv("AMALTHEA_USE_RUST_IONISATION" => "1") do
            Ionisation.IonRatePPTCached(gas, λ0)
        end
        plasma = Nonlinear.PlasmaCumtrapz(grid.to, grid.to, ionrate, ionpot)
        responses = (Nonlinear.Kerr_field(γ3), plasma)
        inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy,
                                         w0=w0, propz=-0.1)
        Eω, transform, _ = with_logger(NullLogger()) do
            Amalthea.setup(grid, xygrid, densityfun,
                           NonlinearRHS.const_norm_free(grid, xygrid,
                                                       PhysData.ref_index_fun(gas, pres)),
                           responses, inputs)
        end
        @assert transform isa NonlinearRHS.TransFree
        linop = LinearOps.make_const_linop(grid, xygrid,
                                           PhysData.ref_index_fun(gas, pres))

        t0 = 0.0
        dt = 0.0005
        n = length(Eω)
        n_time = length(grid.t)
        n_time_over = length(grid.to)
        n_cols = Nx * Ny
        @test n % n_cols == 0
        n_spec = n ÷ n_cols
        m = (grid.ωwin .* (-im .* grid.ω)) ./ (2 .* transform.normfun(0.0))
        ion_ptr = ionrate.rust_handle.ptr

        make_cpu(field=Eω, rhs=transform, time_step=dt;
                 max_dt=time_step, min_dt=time_step) =
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                    "AMALTHEA_NATIVE_GPU" => "off",
                    "AMALTHEA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(rhs, linop, copy(field), t0, time_step;
                                  rtol=1e-6, atol=1e-10, max_dt, min_dt)
            end
        make_gpu(field=Eω, rhs=transform, time_step=dt;
                 max_dt=time_step, min_dt=time_step) =
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on",
                    "AMALTHEA_USE_RUST_IONISATION" => "1") do
                RustNativeStepper(rhs, linop, copy(field), t0, time_step;
                                  rtol=1e-6, atol=1e-10, max_dt, min_dt)
            end
        set_field(ptr, field) = ccall(
            (:set_field, libpath), Cint,
            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t), ptr, field, Csize_t(length(field)))
        set_free(ptr, towin, kfac, mval) = ccall(
            (:native_set_free_params, libpath), Cint,
            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Cuint,
             Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64}),
            ptr, Csize_t(n_time), Csize_t(n_time_over), Csize_t(Ny), Csize_t(Nx),
            Cuint(64), towin, kfac, real.(mval), imag.(mval))
        set_plasma(ptr) = ccall(
            (:native_set_plasma_params, libpath), Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
            ptr, ion_ptr, ionpot, PhysData.e_ratio, plasma.preionfrac,
            plasma.δt, dens0)
        getk(s, i=0) = begin
            out = zeros(ComplexF64, n)
            rc = ccall((:get_ks_stage, libpath), Cint,
                       (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                       s._handle.ptr, Csize_t(i), out, Csize_t(n))
            rc == 0 || error("get_ks_stage failed rc=$rc")
            out
        end

        @testset "Eligibility boundaries and explicit-only policy" begin
            @test RK45._gpu_kernel_supports(transform, linop)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(transform, linop, n)
            end

            egrid = Grid.EnvGrid(L, λ0, (150e-9, 2000e-9), 0.5e-12)
            enorm = NonlinearRHS.const_norm_free(egrid, xygrid,
                                                 PhysData.ref_index_fun(gas, pres))
            elinop = LinearOps.make_const_linop(egrid, xygrid,
                                                PhysData.ref_index_fun(gas, pres))
            eω, etransform, _ = with_logger(NullLogger()) do
                Amalthea.setup(egrid, xygrid, densityfun, enorm,
                               (Nonlinear.Kerr_env(γ3),), inputs)
            end
            @test RK45._gpu_kernel_supports(etransform, elinop)
            env_plasma = Nonlinear.PlasmaCumtrapz(egrid.to, egrid.to, ionrate, ionpot)
            etransform_plasma = NonlinearRHS.TransFree(
                egrid, xygrid, etransform.FT,
                (Nonlinear.Kerr_env(γ3), env_plasma), densityfun, enorm)
            @test !RK45._gpu_kernel_supports(etransform_plasma, elinop)
        end

        local gpu_error
        local gpu_available = true
        local s_gpu
        try
            s_gpu = make_gpu()
        catch e
            gpu_available = false
            gpu_error = e
        end
        if !gpu_available
            require_cuda && error("CUDA Plan 19 setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
        else
            @test RK45._native_backend(s_gpu) === :cuda

            s_cpu = make_cpu()
            @testset "Joint 3-D PPT stage and non-square layout" begin
                k_cpu = getk(s_cpu)
                k_gpu = getk(s_gpu)
                @test maximum(abs.(k_cpu)) > 100 * 1e-12
                rel_stage = norm(k_gpu - k_cpu) / norm(k_cpu)
                println("Free-space Plan 19 PPT stage rel: ", rel_stage)
                @test rel_stage < 1e-8

                # Distinct complex factors per spectral bin and spatial column
                # make both the joint layout and the plasma path observable.
                Eω_probe = copy(Eω)
                for c in 1:n_cols, i in 1:n_spec
                    j = (c - 1) * n_spec + i
                    Eω_probe[j] *= (1 + 0.013 * c + 0.005 * i) +
                                    im * (0.004 * c - 0.001 * i)
                end
                @test set_field(s_gpu._handle.ptr, Eω_probe) == 0
                @test set_field(s_cpu._handle.ptr, Eω_probe) == 0
                rel_probe = norm(getk(s_gpu) - getk(s_cpu)) / norm(getk(s_cpu))
                println("Free-space Plan 19 asymmetric PPT stage rel: ", rel_probe)
                @test rel_probe < 1e-8
                @test set_field(s_gpu._handle.ptr, Eω) == 0
                @test set_field(s_cpu._handle.ptr, Eω) == 0
            end

            @testset "Non-vacuous plasma effect" begin
                energy_strong = 3e-4
                inputs_strong = Fields.GaussGaussField(
                    λ0=λ0, τfwhm=τ, energy=energy_strong, w0=w0, propz=-0.1)
                Eω_strong, transform_strong, _ = with_logger(NullLogger()) do
                    Amalthea.setup(grid, xygrid, densityfun,
                                   transform.normfun, responses, inputs_strong)
                end
                no_plasma = NonlinearRHS.TransFree(
                    grid, xygrid, transform_strong.FT,
                    (Nonlinear.Kerr_field(γ3),), densityfun, transform.normfun)
                s_on = PreconStepper(transform_strong, linop, copy(Eω_strong), t0, dt;
                                     rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                s_off = PreconStepper(no_plasma, linop, copy(Eω_strong), t0, dt;
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                solve(s_on, L)
                solve(s_off, L)
                effect = norm(s_on.yn - s_off.yn) / norm(s_off.yn)
                println("Free-space Plan 19 Julia plasma share: ", effect)
                @test effect > 1e-6

                s_gpu_strong = make_gpu(Eω_strong, transform_strong)
                solve(s_gpu_strong, L)
                rel_native = norm(s_gpu_strong.yn - s_on.yn) / norm(s_on.yn)
                println("Free-space Plan 19 strong native-vs-Julia rel: ", rel_native)
                @test rel_native < 1e-6
                @test norm(s_gpu_strong.yn - s_off.yn) / norm(s_off.yn) > 1e-6
            end

            @testset "Fixed solve, adaptive trajectory, and rejection" begin
                s_cpu_fixed = make_cpu()
                s_gpu_fixed = make_gpu()
                solve(s_cpu_fixed, L)
                solve(s_gpu_fixed, L)
                rel_fixed = norm(s_gpu_fixed.yn - s_cpu_fixed.yn) / norm(s_cpu_fixed.yn)
                println("Free-space Plan 19 fixed solve rel: ", rel_fixed)
                @test rel_fixed < 1e-6

                s_cpu_adapt = make_cpu(Eω, transform, 0.01; max_dt=0.01, min_dt=0.0)
                s_gpu_adapt = make_gpu(Eω, transform, 0.01; max_dt=0.01, min_dt=0.0)
                solve(s_cpu_adapt, L)
                solve(s_gpu_adapt, L)
                rel_adapt = norm(s_gpu_adapt.yn - s_cpu_adapt.yn) / norm(s_cpu_adapt.yn)
                println("Free-space Plan 19 adaptive solve rel: ", rel_adapt)
                @test rel_adapt < 1e-6

                hot = 1e4 .* Eω
                s_cpu_reject = make_cpu(hot, transform, 0.1; max_dt=0.2, min_dt=0.0)
                s_gpu_reject = make_gpu(hot, transform, 0.1; max_dt=0.2, min_dt=0.0)
                before = copy(s_gpu_reject.yn)
                @test !step!(s_cpu_reject)
                @test !step!(s_gpu_reject)
                @test s_gpu_reject.yn == before
                @test (isnan(s_gpu_reject.err) && isnan(s_cpu_reject.err)) ||
                      isapprox(s_gpu_reject.err, s_cpu_reject.err; rtol=1e-10)
                @test (isnan(s_gpu_reject.dtn) && isnan(s_cpu_reject.dtn)) ||
                      isapprox(s_gpu_reject.dtn, s_cpu_reject.dtn; rtol=1e-10)
            end

            @testset "Failed plasma replacement is transactional" begin
                s_tx = make_gpu()
                ones_towin = ones(Float64, n_time_over)
                ones_m = ones(Float64, n)
                @test set_free(s_tx._handle.ptr, ones_towin, 0.0, ones_m) == 0
                @test set_field(s_tx._handle.ptr, Eω) == 0
                @test set_plasma(s_tx._handle.ptr) == 0
                k_before = getk(s_tx)
                bad_rc = ccall(
                    (:native_set_plasma_params, libpath), Cint,
                    (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Float64, Float64, Float64, Float64),
                    s_tx._handle.ptr, Ptr{Cvoid}(C_NULL), ionpot,
                    PhysData.e_ratio, plasma.preionfrac, plasma.δt, dens0)
                @test bad_rc != 0
                @test norm(getk(s_tx) - k_before) / max(norm(k_before), 1e-30) < 1e-12
            end
        end
    end
end

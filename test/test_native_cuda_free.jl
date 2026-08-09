using TestItems

@testitem "Native-Rust CUDA free-space RealGrid Kerr (Plan 17)" tags=[:rust] begin
    import Test: @test, @test_skip, @testset
    using Amalthea
    import Amalthea: Grid, NonlinearRHS, Fields, LinearOps, PhysData, Nonlinear
    using Amalthea.RK45: PreconStepper, RustNativeStepper, step!, solve
    import LinearAlgebra: norm
    import Logging: with_logger, NullLogger

    libpath = RK45._LIBAMALTHEA_RK45
    require_cuda = get(ENV, "AMALTHEA_REQUIRE_CUDA_TESTS", "0") == "1"
    if !isfile(libpath)
        require_cuda && error("CUDA tests are required, but the Rust library was not found")
        @test_skip "Rust library not found"
    else
        gas = :Ar; pres = 1.0; τ = 20e-15; λ0 = 800e-9
        w0 = 60e-6; energy = 1e-9; L = 0.004; R = 300e-6
        Nx = 8; Ny = 6
        grid = Grid.RealGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
        xygrid = Grid.FreeGrid(R, Nx, R, Ny)
        dens0 = PhysData.density(gas, pres)
        densityfun(z) = dens0
        γ3 = PhysData.γ3_gas(gas)
        inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τ, energy=energy, w0=w0)

        function make_case(γ)
            responses = (Nonlinear.Kerr_field(γ),)
            linop = LinearOps.make_const_linop(grid, xygrid,
                                                PhysData.ref_index_fun(gas, pres))
            normfun = NonlinearRHS.const_norm_free(grid, xygrid,
                                                   PhysData.ref_index_fun(gas, pres))
            with_logger(NullLogger()) do
                Amalthea.setup(grid, xygrid, densityfun, normfun, responses, inputs)
            end
        end

        Eω, transform, FT = make_case(γ3)
        @assert transform isa NonlinearRHS.TransFree
        linop = LinearOps.make_const_linop(grid, xygrid, PhysData.ref_index_fun(gas, pres))
        t0 = 0.0
        dt = 0.0005
        n = length(Eω)
        n_time = length(grid.t)
        n_time_over = length(grid.to)
        n_cols = Nx * Ny
        @test n % n_cols == 0
        n_spec = n ÷ n_cols
        m = (grid.ωwin .* (-im .* grid.ω)) ./
            (2 .* transform.normfun(0.0))

        @testset "Eligibility boundaries and explicit-only policy" begin
            @test RK45._gpu_kernel_supports(transform, linop)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "on") do
                @test RK45._gpu_native_eligible(transform, linop, n)
            end
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(transform, linop, n)
            end

            egrid = Grid.EnvGrid(L, λ0, (400e-9, 2000e-9), 0.5e-12)
            elinop = LinearOps.make_const_linop(egrid, xygrid,
                                                PhysData.ref_index_fun(gas, pres))
            enormfun = NonlinearRHS.const_norm_free(egrid, xygrid,
                                                    PhysData.ref_index_fun(gas, pres))
            eω, etransform, _ = with_logger(NullLogger()) do
                Amalthea.setup(egrid, xygrid, densityfun, enormfun,
                               (Nonlinear.Kerr_env(γ3),), inputs)
            end
            @test etransform isa NonlinearRHS.TransFree
            # Plan 18 adds the EnvGrid c2c free-space slice; it remains
            # explicit-only even though pure eligibility is now true.
            @test RK45._gpu_kernel_supports(etransform, elinop)
            withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                    "AMALTHEA_NATIVE_GPU" => "auto") do
                @test !RK45._gpu_native_eligible(etransform, elinop, length(eω))
            end
        end

        local gpu_error
        gpu_available = true
        withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "1",
                "AMALTHEA_NATIVE_GPU" => "on") do
            local s_gpu
            try
                s_gpu = RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                          rtol=1e-6, atol=1e-10,
                                          max_dt=dt, min_dt=dt)
            catch e
                gpu_available = false
                gpu_error = e
                return
            end

            @test RK45._native_backend(s_gpu) === :cuda
            @test RK45._gpu_native_eligible(transform, linop, n)

            getk(s, i) = begin
                k = zeros(ComplexF64, n)
                rc = ccall((:get_ks_stage, libpath), Cint,
                           (Ptr{Cvoid}, Csize_t, Ptr{ComplexF64}, Csize_t),
                           s._handle.ptr, Csize_t(i), k, Csize_t(n))
                rc == 0 || error("get_ks_stage failed rc=$rc")
                k
            end

            s_cpu = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                            "AMALTHEA_NATIVE_GPU" => "off") do
                RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                  rtol=1e-6, atol=1e-10,
                                  max_dt=dt, min_dt=dt)
            end

            @testset "Joint 3-D stage and non-square layout" begin
                k_cpu = getk(s_cpu, 0)
                k_gpu = getk(s_gpu, 0)
                println("Free-space Plan 17 max|k0| CPU/GPU: ",
                        maximum(abs.(k_cpu)), " / ", maximum(abs.(k_gpu)))
                @test maximum(abs.(k_cpu)) > 1e-12
                @test norm(k_gpu - k_cpu) / norm(k_cpu) < 1e-12

                # Deliberately nonsymmetric spectral data prevents a y/x axis
                # swap from hiding behind the symmetric Gaussian input.
                Eω_probe = copy(Eω)
                for c in 1:n_cols, i in 1:n_spec
                    j = (c - 1) * n_spec + i
                    Eω_probe[j] *= (1 + 0.011 * c + 0.007 * i) +
                                    im * (0.003 * c - 0.002 * i)
                end
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_gpu._handle.ptr, Eω_probe, Csize_t(n)) == 0
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_cpu._handle.ptr, Eω_probe, Csize_t(n)) == 0
                k_cpu_probe = getk(s_cpu, 0)
                k_gpu_probe = getk(s_gpu, 0)
                @test norm(k_gpu_probe - k_cpu_probe) / norm(k_cpu_probe) < 1e-12
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_gpu._handle.ptr, Eω, Csize_t(n)) == 0
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_cpu._handle.ptr, Eω, Csize_t(n)) == 0
            end

            @testset "Non-vacuous Kerr effect" begin
                Eω_off, transform_off, _ = make_case(0.0)
                linop_off = LinearOps.make_const_linop(grid, xygrid,
                                                        PhysData.ref_index_fun(gas, pres))
                s_on = PreconStepper(transform, linop, copy(Eω), t0, dt;
                                     rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                s_off = PreconStepper(transform_off, linop_off, copy(Eω_off), t0, dt;
                                      rtol=1e-6, atol=1e-10, max_dt=dt, min_dt=dt)
                solve(s_on, L)
                solve(s_off, L)
                effect = norm(s_on.yn - s_off.yn) / norm(s_on.yn)
                println("Free-space Plan 17 Julia Kerr share: ", effect)
                @test effect > 1e-10
            end

            @testset "Fixed solve and adaptive trajectory" begin
                s_cpu_fixed = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                      "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                      rtol=1e-6, atol=1e-10,
                                      max_dt=dt, min_dt=dt)
                end
                s_gpu_fixed = RustNativeStepper(transform, linop, copy(Eω), t0, dt;
                                                rtol=1e-6, atol=1e-10,
                                                max_dt=dt, min_dt=dt)
                solve(s_cpu_fixed, L)
                solve(s_gpu_fixed, L)
                rel_fixed = norm(s_gpu_fixed.yn - s_cpu_fixed.yn) / norm(s_cpu_fixed.yn)
                println("Free-space Plan 17 fixed solve rel: ", rel_fixed)
                @test rel_fixed < 1e-6

                s_cpu_adapt = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                      "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, copy(Eω), t0, 0.01;
                                      rtol=1e-6, atol=1e-10,
                                      max_dt=0.01, min_dt=0.0)
                end
                s_gpu_adapt = RustNativeStepper(transform, linop, copy(Eω), t0, 0.01;
                                                rtol=1e-6, atol=1e-10,
                                                max_dt=0.01, min_dt=0.0)
                solve(s_cpu_adapt, L)
                solve(s_gpu_adapt, L)
                rel_adapt = norm(s_gpu_adapt.yn - s_cpu_adapt.yn) / norm(s_cpu_adapt.yn)
                println("Free-space Plan 17 adaptive solve rel: ", rel_adapt)
                @test rel_adapt < 1e-6
            end

            @testset "Rejected step and transactional setup" begin
                # A valid second setup is committed, then an invalid setup is
                # rejected. The active replacement must remain usable after
                # the rejection; this exercises both plan ownership and the
                # no-partial-mutation boundary.
                ones_towin = ones(Float64, n_time_over)
                ones_m = ones(Float64, n)
                @test ccall((:native_set_free_params, libpath), Cint,
                            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Cuint,
                             Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64}),
                            s_gpu._handle.ptr, Csize_t(n_time), Csize_t(n_time_over),
                            Csize_t(Ny), Csize_t(Nx), Cuint(64), ones_towin, 0.37,
                            ones_m, zeros(n)) == 0
                @test ccall((:native_set_free_params, libpath), Cint,
                            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Cuint,
                             Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64}),
                            s_gpu._handle.ptr, Csize_t(n_time), Csize_t(n_time_over),
                            Csize_t(0), Csize_t(Nx), Cuint(64),
                            Ptr{Float64}(C_NULL), 1.0,
                            Ptr{Float64}(C_NULL), Ptr{Float64}(C_NULL)) != 0
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_gpu._handle.ptr, Eω, Csize_t(n)) == 0
                @test ccall((:native_set_free_params, libpath), Cint,
                            (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Cuint,
                             Ptr{Float64}, Float64, Ptr{Float64}, Ptr{Float64}),
                            s_gpu._handle.ptr, Csize_t(n_time), Csize_t(n_time_over),
                            Csize_t(Ny), Csize_t(Nx), Cuint(64), grid.towin,
                            dens0 * PhysData.ε_0 * γ3, real.(m), imag.(m)) == 0
                @test ccall((:set_field, libpath), Cint,
                            (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t),
                            s_gpu._handle.ptr, Eω, Csize_t(n)) == 0

                Eω_hot = 1e4 .* Eω
                s_cpu_reject = withenv("AMALTHEA_USE_RUST_CUDA_NATIVE" => "0",
                                       "AMALTHEA_NATIVE_GPU" => "off") do
                    RustNativeStepper(transform, linop, Eω_hot, t0, 0.1;
                                      rtol=1e-6, atol=1e-10, max_dt=0.2, min_dt=0.0)
                end
                s_gpu_reject = RustNativeStepper(transform, linop, Eω_hot, t0, 0.1;
                                                 rtol=1e-6, atol=1e-10,
                                                 max_dt=0.2, min_dt=0.0)
                before = copy(s_gpu_reject.yn)
                @test !step!(s_cpu_reject)
                @test !step!(s_gpu_reject)
                @test s_gpu_reject.yn == before
                @test (isnan(s_gpu_reject.err) && isnan(s_cpu_reject.err)) ||
                      isapprox(s_gpu_reject.err, s_cpu_reject.err; rtol=1e-10)
                @test isapprox(s_gpu_reject.dtn, s_cpu_reject.dtn; rtol=1e-10)
            end
        end
        if !gpu_available
            require_cuda && error("CUDA tests are required, but GPU setup failed: $gpu_error")
            @test_skip "CUDA GPU/toolkit not available on this machine: $gpu_error"
        end
    end
end

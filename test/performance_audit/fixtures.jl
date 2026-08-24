"""Runnable fixture constructors shared by audit correctness and timing tools."""

import Logging: NullLogger, with_logger
import Hankel
import Random: MersenneTwister

const AUDIT_SIZES = (:small, :medium, :large)

function apply_audit_overrides!(fixture)
    if fixture.geometry === :modal
        if haskey(ENV, "AMALTHEA_AUDIT_MODAL_RTOL")
            fixture.transform.rtol = parse(Float64, ENV["AMALTHEA_AUDIT_MODAL_RTOL"])
        end
        if haskey(ENV, "AMALTHEA_AUDIT_MODAL_MAXEVALS")
            fixture.transform.mfcn = parse(Int, ENV["AMALTHEA_AUDIT_MODAL_MAXEVALS"])
        end
    end
    fixture
end

function _check_size(size::Symbol)
    size in AUDIT_SIZES || error("unknown audit size $size; expected one of $AUDIT_SIZES")
    size
end

function _time_window(size::Symbol, grid::Symbol)
    _check_size(size)
    values = grid === :real ? (0.125e-12, 0.5e-12, 2.0e-12) :
             grid === :env  ? (0.5e-12, 2.0e-12, 8.0e-12) :
             error("unknown grid $grid")
    values[findfirst(==(size), AUDIT_SIZES)]
end

function _spatial_size(size::Symbol, values)
    _check_size(size)
    values[findfirst(==(size), AUDIT_SIZES)]
end

function _modeavg_fixture(A, size::Symbol; envelope::Bool=false, kerr::Bool=true)
    radius = 125e-6
    flength = 0.15
    gas = :He
    pressure = 1.0
    λ0 = 800e-9
    λlims = (400e-9, 2000e-9)
    trange = _time_window(size, envelope ? :env : :real)
    args = (radius, flength, gas, pressure)
    kw = (; λ0, λlims, trange, envelope, raman=false, plasma=false, kerr,
           shotnoise=false, energy=5e-6, τfwhm=30e-15, saveN=2)
    Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
        A.Interface.prop_capillary_args(args...; kw...)
    end
    (; id=envelope ? "modeavg_env_kerr" : "modeavg_real_kerr",
       Eω, grid, linop, transform, FT, output, flength, dt=0.005,
       geometry=:modeavg, grid_kind=envelope ? :env : :real)
end

function _modeavg_response_fixture(A, id::AbstractString, size::Symbol;
                                   feature_enabled::Bool=true)
    envelope = occursin("_env_", id)
    grid_kind = envelope ? :env : :real
    λ0 = 800e-9
    radius = 125e-6
    flength = occursin("zdependent", id) ? 0.5 : 0.05
    gas = occursin("raman", id) ? :N2 :
          occursin("adk", id) || occursin("ppt", id) ? :He : :Ar
    pressure = occursin("zdependent", id) && feature_enabled ? (0.5, 5.0) : 1.0
    trange = _time_window(size, grid_kind)
    energy = occursin("adk", id) || occursin("ppt", id) ? 1.6e-3 : 5e-6

    # Public constructors are deliberately used where possible so the same
    # fixture is admissible for the pinned upstream comparison.
    if (!envelope && (occursin("ppt", id) || occursin("adk", id))) ||
       occursin("shotnoise", id) ||
       occursin("zdependent", id) || id == "modeavg_real_raman_sdo_thg"
        plasma = feature_enabled && occursin("ppt", id) ? :PPT :
                 feature_enabled && occursin("adk", id) ? :ADK : false
        raman = feature_enabled && id == "modeavg_real_raman_sdo_thg"
        shotnoise = feature_enabled && occursin("shotnoise", id)
        kw = (; λ0, λlims=(150e-9, 4e-6), trange, envelope,
              raman, rotation=false, vibration=raman, plasma, kerr=true,
              shotnoise, energy, τfwhm=30e-15, saveN=2)
        setup_call() = A.Interface.prop_capillary_args(
            radius, flength, gas, pressure; kw..., rng=MersenneTwister(20260811))
        Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
            (occursin("ppt", id) || occursin("adk", id)) ?
                withenv(setup_call, "AMALTHEA_USE_RUST_IONISATION" => "1") : setup_call()
        end
        return (; id, Eω, grid, linop, transform, FT, output, flength,
                 dt=occursin("adk", id) ? 0.005 : 0.01,
                 geometry=:modeavg, grid_kind,
                 feature=occursin("ppt", id) || occursin("adk", id) ? :plasma :
                         occursin("raman", id) ? :raman :
                         occursin("shotnoise", id) ? :shotnoise : :z_dependent,
                 requires_flength=occursin("zdependent", id))
    end

    # The Hilbert (thg=false), rotational, and envelope-Raman branches need
    # explicit response construction; Interface couples some of these flags.
    grid = envelope ?
        A.Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), trange) :
        A.Grid.RealGrid(flength, λ0, (400e-9, 2000e-9), trange)
    mode = A.Capillary.MarcatiliMode(radius, gas, pressure; kind=:HE, n=1, m=1)
    density = A.PhysData.density(gas, pressure)
    densityfun(z) = density
    kerr = envelope ? A.Nonlinear.Kerr_env(A.PhysData.γ3_gas(gas)) :
                      A.Nonlinear.Kerr_field(A.PhysData.γ3_gas(gas))
    responses = if feature_enabled && (occursin("ppt", id) || occursin("adk", id))
        ionrate = occursin("adk", id) ?
            A.Ionisation.IonRateADK(gas; threshold=true, cycle_average=false) :
            withenv("AMALTHEA_USE_RUST_IONISATION" => "1") do
                A.Ionisation.IonRatePPTCached(gas, λ0)
            end
        plasma = A.Nonlinear.PlasmaCumtrapz(
            grid.to, grid.to, ionrate, A.PhysData.ionisation_potential(gas))
        (kerr, plasma)
    elseif feature_enabled
        rotation = occursin("rotational", id)
        rr = A.Raman.raman_response(grid.to, gas;
                                    rotation, vibration=!rotation)
        raman = envelope ? A.Nonlinear.RamanPolarEnv(grid.to, rr) :
                 A.Nonlinear.RamanPolarField(
                     grid.to, rr; thg=!occursin("nothg", id))
        (kerr, raman)
    else
        (kerr,)
    end
    linop, βfun!, _, _ = A.LinearOps.make_const_linop(grid, mode, grid.referenceλ)
    aeff(z) = A.Modes.Aeff(mode; z)
    input = A.Fields.GaussField(λ0=λ0, τfwhm=20e-15, energy=5e-6)
    Eω, transform, FT = with_logger(NullLogger()) do
        A.setup(grid, densityfun, responses, input, βfun!, aeff)
    end
    (; id, Eω, grid, linop, transform, FT, output=nothing, flength, dt=0.01,
       geometry=:modeavg, grid_kind,
       feature=occursin("ppt", id) || occursin("adk", id) ? :plasma : :raman)
end

function _modeavg_sio2_fixture(A, size::Symbol; feature_enabled::Bool=true)
    γ = 0.1
    β2 = -1e-26
    soliton_order = 4.0
    τ0 = 280e-15
    τfwhm = (2log(1 + sqrt(2))) * τ0
    fr = 0.18
    power = soliton_order^2 * abs(β2) / ((1 - fr) * γ * τ0^2)
    flength = π * τ0^2 / abs(β2)
    kw = (; λ0=835e-9, τfwhm, power, pulseshape=:sech,
          λlims=[450e-9, 8000e-9], trange=_time_window(size, :env) * 2,
          raman=feature_enabled, ramanmodel=:SiO2, fr, shotnoise=false)
    Eω, grid, linop, transform, FT, output = with_logger(NullLogger()) do
        A.Interface.prop_gnlse_args(γ, flength, [0.0, 0.0, β2]; kw...)
    end
    (; id="modeavg_env_raman_sio2", Eω, grid, linop, transform, FT, output,
       flength, dt=flength / 2000, geometry=:modeavg, grid_kind=:env,
       feature=:raman_sio2)
end

function _radial_fixture(A, size::Symbol; kerr::Bool=true)
    gas = :Ar
    pressure = 1.2
    τ = 20e-15
    λ0 = 800e-9
    w0 = 40e-6
    energy = 1e-9
    flength = 0.02
    radius = 4e-3
    nr = _spatial_size(size, (8, 24, 48))
    grid = A.Grid.RealGrid(flength, λ0, (400e-9, 2000e-9),
                           _time_window(size, :real))
    q = Hankel.QDHT(radius, nr, dim=2)
    density = A.PhysData.density(gas, pressure)
    densityfun(z) = density
    responses = (A.Nonlinear.Kerr_field(kerr ? A.PhysData.γ3_gas(gas) : 0.0),)
    nfun = A.PhysData.ref_index_fun(gas, pressure)
    linop = A.LinearOps.make_const_linop(grid, q, nfun)
    normfun = A.NonlinearRHS.const_norm_radial(grid, q, nfun)
    inputs = A.Fields.GaussGaussField(
        λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.15)
    Eω, transform, FT = with_logger(NullLogger()) do
        A.setup(grid, q, densityfun, normfun, responses, inputs)
    end
    (; id="radial_real_kerr", Eω, grid, linop, transform, FT, output=nothing,
       flength, dt=0.001, geometry=:radial, grid_kind=:real)
end

function _radial_response_fixture(A, id::AbstractString, size::Symbol;
                                  feature_enabled::Bool=true)
    envelope = occursin("_env_", id)
    grid_kind = envelope ? :env : :real
    is_raman = occursin("raman", id)
    is_plasma = occursin("ppt", id) || occursin("adk", id)
    is_noise = occursin("shotnoise", id)
    gas = is_raman ? :N2 : occursin("adk", id) ? :He : :Ar
    pressure = is_raman || is_plasma ? 1.5 : 1.2
    τ = is_raman || is_plasma ? 15e-15 : 20e-15
    λ0 = 800e-9
    w0 = is_raman || is_plasma ? 150e-6 : 40e-6
    energy = occursin("adk", id) ? 1.6e-3 :
             occursin("ppt", id) ? 3e-4 :
             is_raman ? (envelope ? 3e-5 : 6e-5) :
             is_noise ? 1e-12 : 1e-9
    flength = 0.02
    nr = _spatial_size(size, (8, 24, 48))
    grid = envelope ?
        A.Grid.EnvGrid(flength, λ0, (150e-9, 2000e-9), _time_window(size, :env)) :
        A.Grid.RealGrid(flength, λ0, (150e-9, 2000e-9), _time_window(size, :real))
    # Keep the first radial sample at a comparable physical radius as Nr
    # changes; a fixed 4 mm aperture makes the Nr=8 plasma field vanish
    # between the axis and its first sampled point.
    radial_radius = 4e-3 * nr / 32
    q = Hankel.QDHT(radial_radius, nr, dim=2)
    density = A.PhysData.density(gas, pressure)
    densityfun(z) = density
    γ3 = feature_enabled || is_raman || is_plasma || is_noise ?
        A.PhysData.γ3_gas(gas) : 0.0
    kerr = _kerr_response(A, grid, γ3)
    responses = if feature_enabled && is_raman
        rotation = occursin("rotational", id)
        rr = A.Raman.raman_response(grid.to, gas;
                                    rotation, vibration=!rotation)
        raman = envelope ? A.Nonlinear.RamanPolarEnv(grid.to, rr) :
                 A.Nonlinear.RamanPolarField(
                     grid.to, rr; thg=!occursin("nothg", id))
        (kerr, raman)
    elseif feature_enabled && is_plasma
        ionrate = withenv("AMALTHEA_USE_RUST_IONISATION" => "1") do
            occursin("adk", id) ?
                A.Ionisation.IonRateADK(gas; threshold=true, cycle_average=false) :
                A.Ionisation.IonRatePPTCached(gas, λ0)
        end
        plasma = A.Nonlinear.PlasmaCumtrapz(
            grid.to, grid.to, ionrate, A.PhysData.ionisation_potential(gas))
        (kerr, plasma)
    else
        (kerr,)
    end
    refidx = A.PhysData.ref_index_fun(gas, pressure)
    linop = A.LinearOps.make_const_linop(grid, q, refidx)
    normfun = A.NonlinearRHS.const_norm_radial(grid, q, refidx)
    input = A.Fields.GaussGaussField(
        λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.1)
    noise_field = feature_enabled && is_noise ?
        A.Fields.generate_noise_field(grid; rng=MersenneTwister(20260811), nmodes=nr) : nothing
    Eω, transform, FT = with_logger(NullLogger()) do
        isnothing(noise_field) ?
            A.setup(grid, q, densityfun, normfun, responses, input) :
            A.setup(grid, q, densityfun, normfun, responses, input; noise_field)
    end
    feature = is_raman ? :raman : is_plasma ? :plasma : is_noise ? :shotnoise : :kerr
    (; id, Eω, grid, linop, transform, FT, output=nothing, flength, dt=0.0005,
       geometry=:radial, grid_kind, feature)
end

function _modal_fixture(A, size::Symbol; kerr::Bool=true)
    gas = :Ar
    pressure = 1.0
    τ = 20e-15
    λ0 = 800e-9
    radius = 125e-6
    energy = 5e-6
    flength = 0.02
    nmodes = _spatial_size(size, (2, 4, 8))
    grid = A.Grid.RealGrid(flength, λ0, (400e-9, 2000e-9),
                           _time_window(size, :real))
    modes = Tuple(A.Capillary.MarcatiliMode(radius, gas, pressure; m=i)
                  for i in 1:nmodes)
    density = A.PhysData.density(gas, pressure)
    densityfun(z) = density
    responses = (A.Nonlinear.Kerr_field(kerr ? A.PhysData.γ3_gas(gas) : 0.0),)
    linop = A.LinearOps.make_const_linop(grid, modes, grid.referenceλ)
    input = A.Fields.GaussField(λ0=λ0, τfwhm=τ, energy=energy)
    Eω, transform, FT = with_logger(NullLogger()) do
        A.setup(grid, densityfun, responses, input, modes, :y)
    end
    (; id="modal_real_scalar", Eω, grid, linop, transform, FT, output=nothing,
       flength, dt=0.001, geometry=:modal, grid_kind=:real)
end

function _modal_response_fixture(A, id::AbstractString, size::Symbol;
                                 feature_enabled::Bool=true)
    envelope = occursin("_env_", id)
    grid_kind = envelope ? :env : :real
    is_raman = occursin("raman", id)
    is_tapered = occursin("tapered", id)
    is_wrapper = occursin("wrappers", id)
    is_vector = occursin("vector", id)
    is_full = occursin("full", id)
    gas = is_raman ? :N2 : :Ar
    pressure = 1.0
    λ0 = 800e-9
    radius = 125e-6
    flength = 0.05
    grid = envelope ?
        A.Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), _time_window(size, :env)) :
        A.Grid.RealGrid(flength, λ0, (400e-9, 2000e-9), _time_window(size, :real))
    nmodes = _spatial_size(size, (2, 4, 8))
    modes = if is_tapered
        a0, aL = radius, 100e-6
        afun = feature_enabled ? (z -> a0 + (aL - a0) * z / flength) : (z -> a0)
        Tuple(A.Capillary.MarcatiliMode(afun, gas, pressure; m=i) for i in 1:nmodes)
    elseif is_wrapper && feature_enabled
        Tuple(A.Antiresonant.ZeisbergerMode(
            radius, gas, pressure; m=i, wallthickness=700e-9, loss=false)
              for i in 1:nmodes)
    elseif occursin("general", id) && feature_enabled
        Tuple(A.Capillary.MarcatiliMode(
            radius, gas, pressure; kind=:HE, n=2, m=i) for i in 1:nmodes)
    else
        Tuple(A.Capillary.MarcatiliMode(
            radius, gas, pressure;
            m=i, ϕ=is_vector && feature_enabled ? π / 4 : 0.0)
              for i in 1:nmodes)
    end
    density = A.PhysData.density(gas, pressure)
    densityfun(z) = density
    structural_feature = is_tapered || is_wrapper || is_vector || is_full ||
                         occursin("general", id)
    γ3 = feature_enabled || is_raman || structural_feature ?
          A.PhysData.γ3_gas(gas) : 0.0
    kerr = envelope ? A.Nonlinear.Kerr_env(γ3) : A.Nonlinear.Kerr_field(γ3)
    responses = if is_raman && feature_enabled
        rr = A.Raman.raman_response(grid.to, gas; rotation=false, vibration=true)
        raman = A.Nonlinear.RamanPolarField(
            grid.to, rr; thg=!occursin("nothg", id))
        (kerr, raman)
    else
        (kerr,)
    end
    linop = is_tapered ? A.LinearOps.make_linop(grid, modes, λ0) :
                         A.LinearOps.make_const_linop(grid, modes, grid.referenceλ)
    input = A.Fields.GaussField(
        λ0=λ0, τfwhm=20e-15, energy=is_raman ? 4e-6 : 5e-6)
    components = is_vector ? :xy : :y
    mfcn = _spatial_size(size, (500, 2000, 8000))
    Eω, transform, FT = with_logger(NullLogger()) do
        is_full && feature_enabled ?
            A.setup(grid, densityfun, responses, input, modes, components;
                    full=true, mfcn) :
            A.setup(grid, densityfun, responses, input, modes, components;
                    full=false)
    end
    feature = is_raman ? :raman : is_tapered ? :z_dependent :
              is_wrapper ? :wrapper_modes : is_vector ? :polarisation :
              is_full ? :full_representation : occursin("general", id) ?
              :general_modes : :kerr
    (; id, Eω, grid, linop, transform, FT, output=nothing, flength, dt=0.001,
       geometry=:modal, grid_kind, feature, requires_flength=is_tapered)
end

function _free_fixture(A, size::Symbol; kerr::Bool=true)
    gas = :Ar
    pressure = 1.0
    τ = 20e-15
    λ0 = 800e-9
    w0 = 60e-6
    energy = 1e-8
    flength = 0.005
    transverse_radius = 300e-6
    ny, nx = _spatial_size(size, ((6, 8), (16, 20), (32, 40)))
    grid = A.Grid.RealGrid(flength, λ0, (400e-9, 2000e-9),
                           _time_window(size, :real))
    xygrid = A.Grid.FreeGrid(transverse_radius, nx, transverse_radius, ny)
    density = A.PhysData.density(gas, pressure)
    densityfun(z) = density
    responses = (A.Nonlinear.Kerr_field(kerr ? A.PhysData.γ3_gas(gas) : 0.0),)
    nfun = A.PhysData.ref_index_fun(gas, pressure)
    linop = A.LinearOps.make_const_linop(grid, xygrid, nfun)
    normfun = A.NonlinearRHS.const_norm_free(grid, xygrid, nfun)
    inputs = A.Fields.GaussGaussField(
        λ0=λ0, τfwhm=τ, energy=energy, w0=w0)
    Eω, transform, FT = with_logger(NullLogger()) do
        A.setup(grid, xygrid, densityfun, normfun, responses, inputs)
    end
    (; id="free_real_kerr", Eω, grid, linop, transform, FT, output=nothing,
       flength, dt=0.0005, geometry=:free, grid_kind=:real)
end

function _free_response_fixture(A, id::AbstractString, size::Symbol;
                                feature_enabled::Bool=true)
    envelope = occursin("_env_", id)
    grid_kind = envelope ? :env : :real
    is_raman = occursin("raman", id)
    is_plasma = occursin("ppt", id) || occursin("adk", id)
    is_zdep = occursin("zdependent", id)
    gas = is_raman ? :N2 : occursin("adk", id) ? :He : :Ar
    pressure = is_raman || is_plasma ? 1.5 : 1.0
    λ0 = 800e-9
    τ = is_raman || is_plasma ? 15e-15 : 20e-15
    w0 = is_raman || is_plasma ? 150e-6 : 60e-6
    energy = occursin("adk", id) ? 1.6e-3 :
             occursin("ppt", id) ? 3e-4 : is_raman ? 6e-5 : 1e-8
    flength = is_zdep ? 0.01 : 0.005
    ny, nx = _spatial_size(size, ((6, 8), (16, 20), (32, 40)))
    grid = envelope ?
        A.Grid.EnvGrid(flength, λ0, (150e-9, 2000e-9), _time_window(size, :env)) :
        A.Grid.RealGrid(flength, λ0, (150e-9, 2000e-9), _time_window(size, :real))
    xygrid = A.Grid.FreeGrid(300e-6, nx, 300e-6, ny)
    density = A.PhysData.density(gas, pressure)
    densityfun(z) = density
    γ3 = feature_enabled || is_raman || is_plasma || is_zdep ?
        A.PhysData.γ3_gas(gas) : 0.0
    kerr = _kerr_response(A, grid, γ3)
    responses = if feature_enabled && is_raman
        rotation = occursin("rotational", id)
        rr = A.Raman.raman_response(grid.to, gas;
                                    rotation, vibration=!rotation)
        (kerr, A.Nonlinear.RamanPolarField(
            grid.to, rr; thg=!occursin("nothg", id)))
    elseif feature_enabled && is_plasma
        ionrate = withenv("AMALTHEA_USE_RUST_IONISATION" => "1") do
            occursin("adk", id) ?
                A.Ionisation.IonRateADK(gas; threshold=true, cycle_average=false) :
                A.Ionisation.IonRatePPTCached(gas, λ0)
        end
        plasma = A.Nonlinear.PlasmaCumtrapz(
            grid.to, grid.to, ionrate, A.PhysData.ionisation_potential(gas))
        (kerr, plasma)
    else
        (kerr,)
    end
    if is_zdep
        p1 = feature_enabled ? 1.5 : 0.5
        linop, densityfun = A.LinearOps.make_linop_free_gradient(
            grid, xygrid, gas, flength, 0.5, p1)
        normfun = A.NonlinearRHS.norm_free_gradient(grid, xygrid, gas, densityfun)
    else
        refidx = A.PhysData.ref_index_fun(gas, pressure)
        linop = A.LinearOps.make_const_linop(grid, xygrid, refidx)
        normfun = A.NonlinearRHS.const_norm_free(grid, xygrid, refidx)
    end
    input = A.Fields.GaussGaussField(
        λ0=λ0, τfwhm=τ, energy=energy, w0=w0, propz=-0.1)
    Eω, transform, FT = with_logger(NullLogger()) do
        A.setup(grid, xygrid, densityfun, normfun, responses, input)
    end
    feature = is_raman ? :raman : is_plasma ? :plasma :
              is_zdep ? :z_dependent : :kerr
    (; id, Eω, grid, linop, transform, FT, output=nothing, flength, dt=0.0005,
       geometry=:free, grid_kind, feature, requires_flength=is_zdep)
end

_kerr_response(A, grid, γ3) = grid isa A.Grid.RealGrid ?
    A.Nonlinear.Kerr_field(γ3) : A.Nonlinear.Kerr_env(γ3)

function _mixture_fixture(A, id::AbstractString, size::Symbol;
                          feature_enabled::Bool=true)
    parts = split(id, '_')
    geometry = Symbol(parts[1])
    grid_kind = Symbol(parts[2])
    grid_kind in (:real, :env) || error("invalid mixture grid in $id")
    gas = :Ar
    pressure = 1.0
    λ0 = 800e-9
    τ = 20e-15
    flength = geometry === :free ? 0.005 : 0.02
    grid = grid_kind === :real ?
        A.Grid.RealGrid(flength, λ0, (400e-9, 2000e-9), _time_window(size, :real)) :
        A.Grid.EnvGrid(flength, λ0, (400e-9, 2000e-9), _time_window(size, :env))
    half_pressure = A.PhysData.pressure(
        gas, A.PhysData.density(gas, pressure) / 2)
    half_density = A.PhysData.density(gas, half_pressure)
    densityfun(z) = [half_density, half_density]
    γ3 = A.PhysData.γ3_gas(gas)
    responses = feature_enabled ?
        ((_kerr_response(A, grid, γ3),), (_kerr_response(A, grid, γ3),)) :
        ((_kerr_response(A, grid, γ3),), (_kerr_response(A, grid, 0.0),))
    refidx = A.PhysData.ref_index_fun((gas, gas), (half_pressure, half_pressure))

    if geometry === :modeavg
        radius = 125e-6
        mode = A.Capillary.MarcatiliMode(
            radius, (gas, gas), (half_pressure, half_pressure); loss=false)
        aeff(z) = A.Modes.Aeff(mode; z)
        linop, βfun!, _, _ = A.LinearOps.make_const_linop(grid, mode, λ0)
        input = A.Fields.GaussField(λ0=λ0, τfwhm=τ, energy=5e-6)
        Eω, transform, FT = with_logger(NullLogger()) do
            A.setup(grid, densityfun, responses, input, βfun!, aeff)
        end
        return (; id, Eω, grid, linop, transform, FT, output=nothing,
                 flength, dt=0.001, geometry, grid_kind)
    elseif geometry === :radial
        nr = _spatial_size(size, (8, 24, 48))
        q = Hankel.QDHT(4e-3, nr, dim=2)
        linop = A.LinearOps.make_const_linop(grid, q, refidx)
        normfun = A.NonlinearRHS.const_norm_radial(grid, q, refidx)
        input = A.Fields.GaussGaussField(
            λ0=λ0, τfwhm=τ, energy=1e-9, w0=40e-6, propz=-0.15)
        Eω, transform, FT = with_logger(NullLogger()) do
            A.setup(grid, q, densityfun, normfun, responses, input)
        end
        return (; id, Eω, grid, linop, transform, FT, output=nothing,
                 flength, dt=0.001, geometry, grid_kind)
    elseif geometry === :modal
        nmodes = _spatial_size(size, (2, 4, 8))
        modes = Tuple(A.Capillary.MarcatiliMode(
            125e-6, (gas, gas), (half_pressure, half_pressure); m=i)
            for i in 1:nmodes)
        linop = A.LinearOps.make_const_linop(grid, modes, grid.referenceλ)
        input = A.Fields.GaussField(λ0=λ0, τfwhm=τ, energy=5e-6)
        Eω, transform, FT = with_logger(NullLogger()) do
            A.setup(grid, densityfun, responses, input, modes, :y)
        end
        return (; id, Eω, grid, linop, transform, FT, output=nothing,
                 flength, dt=0.001, geometry, grid_kind)
    elseif geometry === :free
        ny, nx = _spatial_size(size, ((6, 8), (16, 20), (32, 40)))
        xygrid = A.Grid.FreeGrid(300e-6, nx, 300e-6, ny)
        linop = A.LinearOps.make_const_linop(grid, xygrid, refidx)
        normfun = A.NonlinearRHS.const_norm_free(grid, xygrid, refidx)
        input = A.Fields.GaussGaussField(
            λ0=λ0, τfwhm=τ, energy=1e-8, w0=60e-6)
        Eω, transform, FT = with_logger(NullLogger()) do
            A.setup(grid, xygrid, densityfun, normfun, responses, input)
        end
        return (; id, Eω, grid, linop, transform, FT, output=nothing,
                 flength, dt=0.0005, geometry, grid_kind)
    end
    error("invalid mixture geometry in $id")
end

function build_fixture(A, id::AbstractString, size::Symbol;
                       feature_enabled::Bool=true)
    id == "modeavg_real_kerr" &&
        return _modeavg_fixture(A, size; kerr=feature_enabled)
    id == "modeavg_env_kerr" &&
        return _modeavg_fixture(A, size; envelope=true, kerr=feature_enabled)
    id == "modeavg_env_raman_sio2" &&
        return _modeavg_sio2_fixture(A, size; feature_enabled)
    startswith(id, "modeavg_") && !endswith(id, "_mixture") &&
        return _modeavg_response_fixture(A, id, size; feature_enabled)
    id == "radial_real_kerr" && return _radial_fixture(A, size; kerr=feature_enabled)
    startswith(id, "radial_") && !endswith(id, "_mixture") &&
        return _radial_response_fixture(A, id, size; feature_enabled)
    id == "modal_real_scalar" && return _modal_fixture(A, size; kerr=feature_enabled)
    startswith(id, "modal_") && !endswith(id, "_mixture") &&
        return _modal_response_fixture(A, id, size; feature_enabled)
    id == "free_real_kerr" && return _free_fixture(A, size; kerr=feature_enabled)
    startswith(id, "free_") && !endswith(id, "_mixture") &&
        return _free_response_fixture(A, id, size; feature_enabled)
    endswith(id, "_mixture") && return _mixture_fixture(A, id, size; feature_enabled)
    error("fixture $id does not have a runnable constructor yet")
end

available_fixture_ids() = (
    "modeavg_real_kerr",
    "modeavg_real_ppt",
    "modeavg_real_adk",
    "modeavg_real_raman_sdo_thg",
    "modeavg_real_raman_sdo_nothg",
    "modeavg_real_raman_rotational",
    "modeavg_real_shotnoise",
    "modeavg_real_zdependent",
    "modeavg_env_kerr",
    "modeavg_env_ppt",
    "modeavg_env_adk",
    "modeavg_env_raman_sdo",
    "modeavg_env_raman_sio2",
    "modeavg_env_shotnoise",
    "radial_real_kerr",
    "radial_real_ppt",
    "radial_real_adk",
    "radial_real_raman_thg",
    "radial_real_raman_nothg_rotational",
    "radial_real_shotnoise",
    "radial_env_kerr",
    "radial_env_raman",
    "radial_env_shotnoise",
    "modal_real_scalar",
    "modal_real_vector",
    "modal_real_full",
    "modal_real_general_modes",
    "modal_real_raman_thg",
    "modal_real_raman_nothg",
    "modal_real_tapered",
    "modal_real_wrappers",
    "modal_env_scalar",
    "modal_env_vector_full",
    "modal_env_wrappers",
    "free_real_kerr",
    "free_real_ppt",
    "free_real_adk",
    "free_real_raman_thg",
    "free_real_raman_nothg_rotational",
    "free_real_zdependent",
    "free_env_kerr",
    "modeavg_real_mixture",
    "modeavg_env_mixture",
    "radial_real_mixture",
    "radial_env_mixture",
    "modal_real_mixture",
    "modal_env_mixture",
    "free_real_mixture",
    "free_env_mixture",
)

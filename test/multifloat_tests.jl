using MultiFloats: MultiFloats, Float32x2, Float64x2
using IMEXRungeKutta: IMEXRungeKutta, IMEXProblem
using IMEXRungeKutta: IMEXSSP222, IMEXSSP3433, ARS443

# MultiFloats' double-floats as the state's and the time's type
# ("Requirements" and "Time and the step count" in `CODE.md`; added
# 2026-09-28). They are the software types with no hardware behind them:
#
#   Float64x2 is about 106 bits, finer than any hardware float, so any
#             coefficient, abscissa or `Δt` rounded through `Float64` on
#             the way shows as an error of 1e−16 where 1e−31 is due.
#   Float32x2 is about 46 bits, what Metal can have without `Float64`.
#             It promotes `Float64` *downward*, so it absorbs a `Float64`
#             leak silently; a `Float32` one shows as 1e−7 where 1e−14 is
#             due.
#
# MultiFloats has no conversion to `Int`, to `Float64`, or to any
# integer by rounding, and no `cos` or `sin`, so the right-hand sides
# here are rational in `u` and `t`.

const MULTIFLOATS = (Float32x2, Float64x2)
# The types as the testset names should say them.
mfname(::Type{Float32x2}) = "Float32x2"
mfname(::Type{Float64x2}) = "Float64x2"
mfname(::Type{Complex{T}}) where {T} = "Complex{$(mfname(T))}"

# `u′ = u(1 + t)/10 + t² + (t − 3)u`: an explicit part in `u` and `t`,
# and an implicit one in `u` and `t` whose stage solve has a closed form.
# Both depend on `t`, so that a mistimed stage shows (the two abscissae,
# `CLAUDE.md`). A callback converts the time to `T` itself, and `T(t)`
# has no method from a Float32x2 time to a `Float64` state: the
# package's own `convert_float` does it, and is the identity, allocating
# nothing, when the two types are one.
function f_mf!(du, u, p, t)
    T = real(eltype(du))
    c = IMEXRungeKutta.convert_float(T, t)
    @. du = u * (1 + c) / 10 + c * c
    return nothing
end
function solve_mf!(U, u★, γΔt, p, t)
    T = real(eltype(U))
    c = IMEXRungeKutta.convert_float(T, t)
    @. U = u★ / (1 - γΔt * (c - 3))
    return nothing
end
mf_problem(u0, tspan) = IMEXProblem(f_mf!, solve_mf!, u0, tspan)

# The relative error of `u` against the 256-bit reference `ref`.
relerr(u, ref) = maximum(abs.(BigFloat.(u) .- ref) ./ abs.(ref))

# `Int(n)` of the step count has no method for a MultiFloat: `init`
# refused any MultiFloat `tspan` or `dt` before 2026-09-28. With one, the
# tolerance of the step count must still keep a whole chunk from getting
# one more step by round-off ("The tolerance" in `CODE.md`).
@testset "A $(mfname(S)) time gets a step count, with no extra step from round-off" for
        S in MULTIFLOATS
    nsteps(t0, t1, dt) = init(mf_problem([S(1)], (t0, t1)), IMEXSSP222(); dt).nsteps
    Δt(t0, t1, dt) = init(mf_problem([S(1)], (t0, t1)), IMEXSSP222(); dt).dt
    @test nsteps(S(0), S(1), S(1) / 10) == 10
    @test Δt(S(0), S(1), S(1) / 10) === S(1) / 10
    @test nsteps(S(0), S(1), S(3) / 10) == 4
    @test nsteps(S(0), S(1), S(1) / 3 + 100 * eps(S)) == 3
    @test nsteps(S(0), S(1), S(1) / 3 - 100 * eps(S)) == 4
    # The chunk sweep of `interface_tests.jl`, in `S`.
    worst = zero(S)
    for T in (S(1) / 10, S(3) / 10, S(1) / 7, S(25) / 10000),
        k in (0, 1, 7, 99, 1000, 123456), m in 1:12

        t0, t1, dt = k * T, (k + 1) * T, T / m
        @test nsteps(t0, t1, dt) == m
        worst = max(worst, Δt(t0, t1, dt) / dt - 1)
    end
    # Measured 2026-09-28: 2.0e−10 for Float32x2 and 6.1e−28 for Float64x2,
    # at k = 123456; `eps(S)` times the `Float64` sweep's 1.1e−11/eps(Float64).
    @test worst < 1e5 * eps(S)
    # An integer `tspan` with a MultiFloat `dt` promotes to the MultiFloat.
    integ = init(mf_problem([S(1)], (0, 1)), IMEXSSP222(); dt = S(1) / 10)
    @test typeof(integ.t) === S && integ.nsteps == 10
end

# A coefficient pasted as a `Float64` literal, an abscissa summed from
# converted entries, or a `Δt` formed in `Float64` would leave the run a
# hardware float's error off the exact one, not its own type's
# ("Coefficient precision" in `CLAUDE.md`). The reference is the same run
# in 256-bit `BigFloat`, whose own error is 1e−77.
@testset "A $(mfname(S)) run carries its type's precision: a 256-bit run to a few eps" for
        S in MULTIFLOATS
    worst = 0.0
    for tab in all_tableaus()
        run(T) = solve(mf_problem(T[1, 2], (T(0), T(1))), tab; dt = T(1) / 7).u
        ref = at256(() -> run(BigFloat))
        got = run(S)
        @test eltype(got) === S
        err = relerr(got, ref) / BigFloat(eps(S))
        @test err ≤ 8
        worst = max(worst, Float64(err))
    end
    # Measured 2026-09-28: 2.1 eps for Float32x2, 3.3 for Float64x2.
    @info "A $(mfname(S)) run against 256 bits, in eps($(mfname(S)))" worst
end

# The state's type and the time's are separate for a MultiFloat too
# ("Two types" in `CODE.md`): a Float64x2 state with `Float64` time must
# not make the time a Float64x2, nor `γΔt` a `Float64`; and a `Float64`
# state with Float64x2 time must not narrow the time.
@testset "The types stay separate: $(mfname(S)) state with Float64 time, and back" for
        S in MULTIFLOATS
    integ = mock_integrator(IMEXSSP3433(), S[1, 2]; tspan = (0.0, 1.0), dt = 0.1)
    @test typeof(integ.t) === Float64 && typeof(integ.dt) === Float64
    solve!(integ)
    log = integ.p
    @test eltype(integ.u) === S
    @test eltype(log.γΔt) === S && eltype(log.t) === Float64
    @test integ.t === 1.0
    integ = solve(mf_problem([1.0, 2.0], (S(0), S(1))), IMEXSSP3433(); dt = S(1) / 10)
    @test typeof(integ.t) === S && eltype(integ.u) === Float64
    @test integ.t === S(1)
    # A complex state over the double-float: `T = real(eltype(u0))`.
    integ = solve(mf_problem(Complex{S}[1, 2im], (S(0), S(1))), IMEXSSP3433();
                  dt = S(1) / 7)
    ref = at256(() -> solve(mf_problem(Complex{BigFloat}[1, 2im],
                                       (BigFloat(0), BigFloat(1))),
                            IMEXSSP3433(); dt = BigFloat(1) / 7).u)
    @test eltype(integ.u) === Complex{S}
    @test maximum(abs.(Complex{BigFloat}.(integ.u) .- ref) ./ abs.(ref)) ≤ 8 * eps(S)
end

# MultiFloats' two types do not promote to each other
# (`promote_type(Float32x2, Float64x2)` is a `UnionAll`), so a `tspan` in
# one and a `dt` in the other has no time type; it must be refused as
# such, not fail later in a conversion.
@testset "A tspan and dt in different double-floats are refused, saying why" begin
    prob = mf_problem([1.0], (Float32x2(0), Float32x2(1)))
    err = try
        init(prob, IMEXSSP222(); dt = Float64x2(1) / 10)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("not one floating-point type", err.msg)
end

# A double-float has no fixed width, so "correctly rounded" in the sense
# of `prevfloat` and `nextfloat` (`tableau_tests.jl`) does not apply;
# the claim is half an ulp of the leading `eps(T)`. A conversion through
# `Float64`, or at the global precision, would miss it by far.
@testset "The coefficients in $(mfname(T)) are the 256-bit values, to eps/2" for
        T in MULTIFLOATS
    within(x, q) = at256(() -> abs(BigFloat(x) - q) ≤ BigFloat(eps(T)) / 2 * abs(q))
    for spec in TABLEAUS
        tab = spec.make()
        ref = reference_coefficients(tab)
        for Tt in (Float64, T)
            co = IMEXRungeKutta.coefficients(T, Tt, tab)
            for f in (:Ã, :b̃, :γ, :Ā, :b̄)
                @test eltype(co[f]) === T
                @test all(within.(co[f], ref[f]))
            end
            for f in (:c̃, :c)
                @test eltype(co[f]) === Tt
                Tt === T && @test all(within.(co[f], ref[f]))
            end
            for prec in (64, 1024)
                co′ = setprecision(() -> IMEXRungeKutta.coefficients(T, Tt, tab),
                                   BigFloat, prec)
                @test all(f -> co′[f] == co[f], keys(co))
            end
        end
    end
end

# A MultiFloat that reached `step!` as an abstract type, or a conversion
# per step, would allocate and could be type-unstable; and a by-owner
# kernel that differed from the broadcast would change the bits
# ("The result does not depend on the path" in `CODE.md`). By owner at
# more than one thread a step allocates its sticky tasks, whatever the
# element type, and is held to `owner_tests.jl`'s bound.
@testset "$(mfname(T)): step! is inferred, allocates as Float64 does, same on every path" for
        T in (Float32x2, Float64x2, Complex{Float64x2})
    S = real(T)
    for tab in all_tableaus()
        u0 = T[1, 2, 3]
        for partition in (nothing, :even)
            integ = init(mf_problem(copy(u0), (S(0), S(1))), tab; dt = S(1) / 70,
                         partition)
            @test (@inferred step!(integ)) === nothing
            CHECK_BOUNDS_FORCED && continue
            if partition === nothing || NT == 1
                @test step_allocations(integ) == 0
            else
                bytes = owner_step_allocations(integ)
                @test bytes ≤ owner_launches(tab) * (128 + 768 * NT)
            end
        end
        # The owner path's own problem, with its limiters, in `Float64`
        # time; `random_state` is in `T`.
        v0 = random_state(T, NSTATE)
        ref = owner_run(tab, v0, nothing)
        for partition in (:even, irregular_partition(NSTATE))
            got = owner_run(tab, v0, partition)
            @test all(n -> bits(got[n]) == bits(ref[n]), eachindex(ref))
        end
    end
end

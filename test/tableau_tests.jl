using IMEXRungeKutta: IMEXRungeKutta, IMEXTableau
using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP2332, IMEXSSP3332, IMEXSSP3433
using IMEXRungeKutta: ARS222, ARS443

# The properties of each named tableau, as measured in step 1 and recorded
# in `CODE.md` ("Tableaus", "Measured properties"). `stiffly_accurate` is
# (implicit, explicit); `ssp` is the SSP coefficient of the explicit part.
const TABLEAUS = [
    (make = IMEXSSP222, name = "SSP2(2,2,2)", R = BigFloat, order = 2,
     stiffly_accurate = (false, false), ssp = 1,
     solves = Bool[1, 1], explicit_used = Bool[1, 1], implicit_used = Bool[1, 1],
     scratch = 5),
    (make = IMEXSSP2322, name = "SSP2(3,2,2)", R = Rational{BigInt}, order = 2,
     stiffly_accurate = (true, false), ssp = 1,
     solves = Bool[1, 1, 1], explicit_used = Bool[0, 1, 1],
     implicit_used = Bool[1, 1, 1], scratch = 6),
    (make = IMEXSSP2332, name = "SSP2(3,3,2)", R = Rational{BigInt}, order = 2,
     stiffly_accurate = (true, false), ssp = 2,
     solves = Bool[1, 1, 1], explicit_used = Bool[1, 1, 1],
     implicit_used = Bool[1, 1, 1], scratch = 7),
    (make = IMEXSSP3332, name = "SSP3(3,3,2)", R = BigFloat, order = 2,
     stiffly_accurate = (false, false), ssp = 1,
     solves = Bool[1, 1, 1], explicit_used = Bool[1, 1, 1],
     implicit_used = Bool[1, 1, 1], scratch = 7),
    (make = IMEXSSP3433, name = "SSP3(4,3,3)", R = BigFloat, order = 3,
     stiffly_accurate = (false, false), ssp = 1,
     solves = Bool[1, 1, 1, 1], explicit_used = Bool[0, 1, 1, 1],
     implicit_used = Bool[1, 1, 1, 1], scratch = 8),
    (make = ARS222, name = "ARS(2,2,2)", R = BigFloat, order = 2,
     stiffly_accurate = (true, true), ssp = 0,
     solves = Bool[0, 1, 1], explicit_used = Bool[1, 1, 0],
     implicit_used = Bool[0, 1, 1], scratch = 5),
    (make = ARS443, name = "ARS(4,4,3)", R = Rational{BigInt}, order = 3,
     stiffly_accurate = (true, true), ssp = 0,
     solves = Bool[0, 1, 1, 1, 1], explicit_used = Bool[1, 1, 1, 1, 0],
     implicit_used = Bool[0, 1, 1, 1, 1], scratch = 9),
]

# The residual allowed for an irrational tableau held at 256 bits, whose
# round-off is about 1e-77. A rational tableau must meet each condition
# exactly.
tolerance(::IMEXTableau{Rational{BigInt}}) = 0
tolerance(::IMEXTableau{BigFloat}) = 1e-70

# The labels of the order-`p` conditions that miss by more than `tol`, so
# that a failure names them.
failing_conditions(tab, p, tol) =
    at256(() -> [label for (label, r) in order_residuals(tab, p) if abs(r) > tol])

# The error a call throws, or `nothing`.
function thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

# A transcription error in a coefficient, or a coupling condition that the
# two abscissae break, shows up here as a failed condition: this is the
# independent check of every tableau in `src/tableaus.jl`.
@testset "Each named tableau meets every order condition up to its order" begin
    for spec in TABLEAUS
        tab = spec.make()
        @test tab isa IMEXTableau{spec.R}
        @test tab.name == spec.name
        for p in 1:(spec.order)
            @test failing_conditions(tab, p, tolerance(tab)) == String[]
        end
    end
end

# A tableau that is accidentally of higher order than stated, or order
# conditions too weak to tell orders apart, would make the order test
# above vacuous.
@testset "Each named tableau misses some condition of the next order by > 1e-3" begin
    for spec in TABLEAUS
        tab = spec.make()
        @test max_residual(tab, spec.order + 1) > 1e-3
    end
end

# A transcription error in the implicit diagonal, or a helper that
# mistakes the degree of R at a trivial first stage, would change the
# stability claims in `CODE.md`, which are computed, not quoted.
@testset "Each implicit part is A-stable with R(∞) = 0, so L-stable" begin
    for spec in TABLEAUS
        tab = spec.make()
        @test abs(R_infinity(tab)) ≤ tolerance(tab)
        @test is_A_stable(tab)
    end
end

# A wrong stability polynomial would make the A-stability claim above
# meaningless; it is checked against the direct formula.
@testset "P/Q is R(z) = 1 + z bᵀ(I − zA)⁻¹𝟙 off the real axis" begin
    for spec in TABLEAUS
        tab = spec.make()
        P, Q = stability_polynomials(tab)
        for z in at256(() -> (Complex(BigFloat(-3), BigFloat(2)),
                              Complex(BigFloat(1 // 2), BigFloat(-7)), BigFloat(-40)))
            @test at256(() -> abs(polyval(P, z) / polyval(Q, z) -
                                  stability_function(tab, z))) < 1e-70
        end
    end
end

# The SSP coefficient and stiff accuracy recorded in `CODE.md` would
# silently go stale if a tableau changed.
@testset "The SSP coefficient and stiff accuracy are as recorded" begin
    for spec in TABLEAUS
        tab = spec.make()
        @test abs(ssp_coefficient(tab) - spec.ssp) ≤ 1e-10
        @test stiffly_accurate(tab) == spec.stiffly_accurate
    end
end

# A wrong pattern would make step 2's plan call `f_exp!` where nothing
# reads it, or skip a tendency that is read, and a wrong scratch count
# would allocate the wrong number of arrays ("One step", "The stage plan
# and storage" in `CODE.md`).
@testset "The solving, explicit-used and implicit-used patterns are as recorded" begin
    for spec in TABLEAUS
        tab = spec.make()
        @test IMEXRungeKutta.nstages(tab) == length(spec.solves)
        @test IMEXRungeKutta.solves(tab) == spec.solves
        @test IMEXRungeKutta.explicit_used(tab) == spec.explicit_used
        @test IMEXRungeKutta.implicit_used(tab) == spec.implicit_used
        @test IMEXRungeKutta.scratch_count(tab) == spec.scratch
    end
    # "Cost" in `CODE.md`: three explicit evaluations and four stage
    # solves per SSP3(4,3,3) step, and 1 + 4 + 3 = 8 scratch arrays.
    tab = IMEXSSP3433()
    @test count(IMEXRungeKutta.explicit_used(tab)) == 3
    @test count(IMEXRungeKutta.solves(tab)) == 4
    @test IMEXRungeKutta.scratch_count(tab) == 8
end

# Mixing up the two abscissae is invisible on any problem autonomous in
# `t` (SciML/OrdinaryDiffEq.jl#4620). Summed from converted entries, an
# abscissa would also not be the value the tableau defines.
@testset "The abscissae are the exact row sums, and c̃ differs from c" begin
    for spec in TABLEAUS
        tab = spec.make()
        at256() do
            @test tab.c̃ == [sum(tab.Ã[k, :]) for k in 1:length(tab.b)]
            @test tab.c == [sum(tab.A[k, :]) for k in 1:length(tab.b)]
        end
    end
    tab = IMEXSSP3433()
    α = tab.A[1, 1]
    @test tab.c̃ == [0, 0, 1, 1 // 2]
    @test tab.c == [α, 0, 1, 1 // 2]
    @test tab.c̃[1] != tab.c[1]
    @test IMEXSSP2322().c̃ == [0, 0, 1]
    @test IMEXSSP2322().c == [1 // 2, 0, 1]
    @test IMEXSSP2332().c̃ == [0, 1 // 2, 1]
    @test IMEXSSP2332().c == [1 // 4, 1 // 4, 1]
    @test ARS443().c == ARS443().c̃ == [0, 1 // 2, 2 // 3, 1 // 2, 1]
end

# A slip in a closed form that the order conditions happen not to see
# would still contradict the derivation in "SSP3(4,3,3) in closed form"
# and the stated diagonals.
@testset "The closed forms are the values CODE.md derives" begin
    at256() do
        tab = IMEXSSP3433()
        α = tab.A[1, 1]
        @test abs(3α^2 - 9α + 2) < 1e-70
        @test 0 < α < 1 // 2
        @test tab.A[4, 1] == α / 4
        @test tab.A[4, 2] == (1 - 2α) / 4
        for tab in (IMEXSSP222(), IMEXSSP3332())
            @test abs(tab.A[1, 1] - (1 - 1 / sqrt(BigFloat(2)))) < 1e-70
        end
        tab = ARS222()
        @test abs(tab.A[2, 2] - (1 - 1 / sqrt(BigFloat(2)))) < 1e-70
        @test abs(tab.Ã[3, 1] + 1 / sqrt(BigFloat(2))) < 1e-70
    end
    # SSP2(3,3,2) as the reviewer recalled it from the paper (not yet
    # checked against it), with the order-3 miss the review computed
    # symbolically: bᵀAc − 1/6 = 1/24.
    tab = IMEXSSP2332()
    @test Dict(order_residuals(tab, 3))["bᵀAc"] == 1 // 24
    @test [tab.A[k, k] for k in 1:3] == [1 // 4, 1 // 4, 1 // 3]
end

# A closed form evaluated at the global precision would change with a
# caller's `setprecision`, and lose digits below 256 bits.
@testset "The named tableaus are 256-bit whatever the global precision" begin
    for spec in TABLEAUS
        spec.R === BigFloat || continue
        ref = spec.make()
        for prec in (64, 1024)
            tab = setprecision(spec.make, BigFloat, prec)
            for f in (:Ã, :b̃, :A, :b, :c̃, :c)
                @test getfield(tab, f) == getfield(ref, f)
                @test all(x -> precision(x) == 256, getfield(tab, f))
            end
        end
    end
end

# A tableau that is not triangular, or that needs `g(uⁿ)` at a stage that
# makes no solve, cannot run under the stage contract; accepted silently,
# it would give wrong results rather than an error ("Admissibility" in
# `CODE.md`).
@testset "The constructor refuses a malformed or inadmissible tableau, saying why" begin
    z2 = [0 0; 0 0]
    ex = [0 0; 1 0]
    h = [1 // 2, 1 // 2]
    function refused(args...)
        err = thrown(() -> IMEXTableau("T", args...))
        return err isa ArgumentError ? err.msg : err
    end
    msg = refused([0 0 0; 1 0 0], h, z2, h)
    @test occursin("Ã must be square", msg)
    msg = refused(ex, h, [1 0 0; 0 1 0; 0 0 1], h)
    @test occursin("same size as Ã", msg)
    msg = refused(ex, [1 // 3, 1 // 3, 1 // 3], [1 0; 0 1], h)
    @test occursin("b̃ must have one weight per stage (2)", msg)
    msg = refused(ex, h, [1 0; 0 1], [1])
    @test occursin("b must have one weight per stage (2)", msg)
    msg = refused(zeros(0, 0), Float64[], zeros(0, 0), Float64[])
    @test occursin("at least one", msg)
    # Not triangular: an explicit diagonal, an explicit upper entry, and an
    # implicit upper entry.
    msg = refused([1 0; 1 0], h, [1 0; 0 1], h)
    @test occursin("strictly lower triangular", msg) && occursin("ã[1,1]", msg)
    msg = refused([0 1; 1 0], h, [1 0; 0 1], h)
    @test occursin("strictly lower triangular", msg) && occursin("ã[1,2]", msg)
    msg = refused(ex, h, [1 1; 0 1], h)
    @test occursin("A must be lower triangular", msg) && occursin("a[1,2]", msg)
    # ESDIRK-type: the trapezoidal rule's first column needs g(uⁿ).
    msg = refused(ex, h, [0 0; 1//2 1//2], h)
    @test occursin("not admissible", msg) && occursin("a[1,1] = 0", msg)
    @test occursin("a[2,1] = 1//2", msg) && occursin("ESDIRK", msg)
    # A zero diagonal with a nonzero weight.
    msg = refused(ex, h, [0 0; 0 1], h)
    @test occursin("not admissible", msg) && occursin("b[1] = 1//2", msg)
    msg = refused([0.0 0.0; NaN 0.0], h, [1 0; 0 1], h)
    @test occursin("non-finite", msg)
end

# A caller's own tableau goes through the same constructor, and must be
# held exactly: rationals as rationals, and binary floats without loss.
@testset "A caller's tableau is held exactly, as rationals or as BigFloat" begin
    tab = IMEXTableau("heun-trap", [0 0; 1 0], [1 // 2, 1 // 2], [1 0; -1//2 1], [1 // 2, 1 // 2])
    @test tab isa IMEXTableau{Rational{BigInt}}
    @test tab.c == [1, 1 // 2]
    tab = IMEXTableau("floats", [0.0 0.0; 0.1 0.0], [0.5, 0.5], [0.3 0.0; 0.0 0.3], [0.5, 0.5])
    @test tab isa IMEXTableau{BigFloat}
    @test tab.Ã[2, 1] == big(0.1)
    @test tab.c̃ == [0, big(0.1)]
end

# Rounding twice, summing converted entries, forming a quotient in `T`, or
# converting at the global precision would each move a coefficient off
# the correctly rounded 256-bit value, and so off the value the tableau
# defines.
function correctly_rounded(x::BigFloat, q::BigFloat)
    return x == q && precision(x) == 256
end
function correctly_rounded(x::AbstractFloat, q::BigFloat)
    return at256() do
        e = abs(BigFloat(x) - q)
        return e ≤ abs(BigFloat(prevfloat(x)) - q) && e ≤ abs(BigFloat(nextfloat(x)) - q)
    end
end
function reference_coefficients(tab)
    return at256() do
        s = length(tab.b)
        big256(x) = BigFloat(x)
        a(k) = tab.A[k, k]
        Ā = [j < k && !iszero(a(j)) ? big256(tab.A[k, j] / a(j)) : big256(0)
             for k in 1:s, j in 1:s]
        b̄ = [!iszero(a(j)) ? big256(tab.b[j] / a(j)) : big256(0) for j in 1:s]
        return (; Ã = big256.(tab.Ã), b̃ = big256.(tab.b̃), γ = big256.(a.(1:s)), Ā, b̄,
                c̃ = big256.(tab.c̃), c = big256.(tab.c))
    end
end
@testset "The converted coefficients are the correctly rounded 256-bit values" begin
    for spec in TABLEAUS
        tab = spec.make()
        ref = reference_coefficients(tab)
        for T in (Float32, Float64, BigFloat), Tt in (Float64, T)
            co = IMEXRungeKutta.coefficients(T, Tt, tab)
            for f in (:Ã, :b̃, :γ, :Ā, :b̄)
                @test eltype(co[f]) === T
                @test all(correctly_rounded.(co[f], ref[f]))
            end
            for f in (:c̃, :c)
                @test eltype(co[f]) === Tt
                @test all(correctly_rounded.(co[f], ref[f]))
            end
            # Under another global precision, the same values.
            for prec in (64, 1024)
                co′ = setprecision(() -> IMEXRungeKutta.coefficients(T, Tt, tab),
                                   BigFloat, prec)
                @test all(f -> co′[f] == co[f], keys(co))
                T === BigFloat && @test all(x -> precision(x) == 256, co′.Ā)
            end
        end
    end
    # An anchor independent of the helpers: α of SSP3(4,3,3) in Float64,
    # and SSP2(2,2,2)'s 1/γ = 2 + √2 in Float32.
    @test IMEXRungeKutta.coefficients(Float64, Float64, IMEXSSP3433()).γ[1] ===
          0.24169426078820838
    @test IMEXRungeKutta.coefficients(Float32, Float32, IMEXSSP222()).b̄ ==
          fill(Float32(1 + 1 / sqrt(big(2))), 2)
end

# The literature's SSP3(4,3,3), and both upstream implementations, use
# α, β, η to 14 digits. If those were not the closed form rounded, the
# closed form here would be a different method; and `CODE.md` states the
# order residuals the truncation leaves.
@testset "The 14 printed digits of SSP3(4,3,3) are the closed form rounded" begin
    tab = IMEXSSP3433()
    at256() do
        α, β, η = tab.A[1, 1], tab.A[4, 1], tab.A[4, 2]
        @test round(BigInt, α * big(10)^14) == 24169426078821
        @test round(BigInt, β * big(10)^14) == 6042356519705
        @test round(BigInt, η * big(10)^14) == 12915286960590
    end
    α = 24169426078821 // 10^14
    β = 6042356519705 // 10^14
    η = 12915286960590 // 10^14
    # Held exactly: the explicit part and the weights are rational too.
    w = [0, 1 // 6, 1 // 6, 2 // 3]
    printed = IMEXTableau("SSP3(4,3,3), 14 digits",
                          [0 0 0 0; 0 0 0 0; 0 1 0 0; 0 1//4 1//4 0], w,
                          [α 0 0 0; -α α 0 0; 0 1-α α 0; β η 1//2-β-η-α α], w)
    at256() do
        @test BigFloat.(printed.Ã) == tab.Ã
        @test BigFloat.(printed.b̃) == tab.b̃
        @test BigFloat.(printed.b) == tab.b
    end
    @test printed isa IMEXTableau{Rational{BigInt}}
    @test failing_conditions(printed, 1, 0) == String[]
    @test failing_conditions(printed, 2, 0) == String[]
    @test 1e-15 < max_residual(printed, 3) < 5e-15
    # Truncation also leaves R(∞) off zero (measured 3.1e-13).
    @test 1e-13 < abs(R_infinity(printed)) < 1e-12
end

# OrdinaryDiffEqSDIRK's ARS443 has b̃ = b, where the paper (§2.8, p. 160)
# and ClimaTimeSteppers have b̃ = the last row of Ã. Step 3's oracle comparison
# of ARS(4,4,3) depends on knowing that both are third order and that
# upstream's evaluates the explicit part at stage 5 as well.
@testset "Upstream's ARS(4,4,3) variant, b̃ = b, is a different third-order method" begin
    tab = ARS443()
    variant = IMEXTableau("ARS(4,4,3), b̃ = b", tab.Ã, tab.b, tab.A, tab.b)
    @test variant.b̃ != tab.b̃
    for p in 1:3
        @test failing_conditions(variant, p, 0) == String[]
    end
    @test max_residual(variant, 4) > 1e-3
    @test IMEXRungeKutta.explicit_used(variant) == Bool[1, 1, 1, 1, 1]
end

# The order conditions are the independent check of every transcription,
# so a slip in any one coefficient must break one of them, or `R(∞) = 0`,
# or be refused by the constructor. The sweep finds the coefficients for
# which that is not so; there is exactly one (measured in step 1).
function undetected_perturbations(spec)
    tab = spec.make()
    R = eltype(tab.b)
    δ = at256(() -> R(1 // 1000))
    found = String[]
    for (f, label) in ((:Ã, "ã"), (:b̃, "b̃"), (:A, "a"), (:b, "b"))
        x = getfield(tab, f)
        for i in eachindex(x)
            if x isa Matrix
                k, j = Tuple(CartesianIndices(x)[i])
                (f === :Ã ? j < k : j ≤ k) || continue
            end
            parts = Dict(f => copy(getfield(tab, f)) for f in (:Ã, :b̃, :A, :b))
            at256(() -> (parts[f][i] += δ))
            m = thrown(() -> IMEXTableau("m", parts[:Ã], parts[:b̃], parts[:A], parts[:b]))
            m isa ArgumentError && continue
            m = IMEXTableau("m", parts[:Ã], parts[:b̃], parts[:A], parts[:b])
            missed = all(p -> isempty(failing_conditions(m, p, tolerance(m))), 1:spec.order)
            missed &= abs(R_infinity(m)) ≤ tolerance(m)
            missed && push!(found, "$label$(Tuple(CartesianIndices(x)[i]))")
        end
    end
    return found
end
@testset "A slip in any coefficient but SSP2(3,2,2)'s a₁₁ fails a check" begin
    for spec in TABLEAUS
        @test undetected_perturbations(spec) ==
              (spec.name == "SSP2(3,2,2)" ? ["a(1, 1)"] : String[])
    end
    # That one is free at order 2: with b₁ = b̃₁ = 0 it enters only `A c`,
    # an order-3 term, and stiff accuracy makes `R(∞) = 0` whatever it is.
    # So it is pinned here: 1/2, the diagonal, as both upstreams have it.
    tab = IMEXSSP2322()
    @test [tab.A[k, k] for k in 1:3] == fill(1 // 2, 3)
    @test tab.A[2, 1] == -tab.A[1, 1]
end

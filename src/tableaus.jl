# The ten named tableaus, in closed form ("Tableaus" in `CODE.md`): seven
# IMEX and, at the end, three purely explicit.
#
# They are functions, not constants, because a BigFloat does not survive
# precompilation reliably. The irrational ones are computed inside
# `with_coefficient_precision`, at 256 bits whatever the global precision.
# The rational ones are held exactly. No Float64 literal appears.
#
# Each is cross-checked against OrdinaryDiffEqSDIRK's `imex_tableaus.jl`
# (2.9.6) and ClimaTimeSteppers' `src/solvers/imex_tableaus.jl` (main,
# 2026-09-24), by reading them, except SSP2(3,3,2), which neither has.
# `CODE.md` ("Tableaus", "Cross-checks") records the one disagreement. The
# ARS schemes are cited by the paper's own section and page, checked
# against it. The Pareschi–Russo schemes are cited by their table in the
# preprint arXiv:1009.2757 (May 2004), checked against it (Tables 2–6 are
# identical in the October 2003 preprint); the published version may
# number them differently. The independent check of each transcription is
# the order conditions in `test/tableau_tests.jl`.

# `1 − 1/√2`, the diagonal of SSP2(2,2,2), SSP3(3,3,2) and ARS(2,2,2),
# which makes each implicit part L-stable. Call inside
# `with_coefficient_precision`.
gamma_sqrt2() = 1 - 1 / sqrt(BigFloat(2))

"""
    IMEXSSP222()

SSP2(2,2,2) of Pareschi & Russo (2005): two stages, both explicit-used and
both solving, second order, with the diagonal `γ = 1 − 1/√2`. The explicit
part is Heun's method (SSP coefficient 1). It is the debugging scheme.
Returns an [`IMEXTableau`](@ref) of 256-bit `BigFloat`; the measured
properties are in `CODE.md`, "Tableaus".
"""
function IMEXSSP222()
    # Pareschi & Russo (2005), IMEX-SSP2(2,2,2), the L-stable scheme
    # (Table 2 of arXiv:1009.2757).
    return with_coefficient_precision() do
        γ = gamma_sqrt2()
        z = zero(BigFloat)
        o = one(BigFloat)
        h = o / 2
        Ã = [z z
             o z]
        b̃ = [h, h]
        A = [γ z
             1-2γ γ]
        b = [h, h]
        return IMEXTableau{BigFloat}("SSP2(2,2,2)", Ã, b̃, A, b)
    end
end

"""
    IMEXSSP2322()

SSP2(3,2,2) of Pareschi & Russo (2005): three stages, all solving, of
which the first is not explicit-used; second order, with the diagonal
`1/2` and a stiffly accurate implicit part. It is rational, and so held
exactly as `Rational{BigInt}`. `IMEXSSPksσp` is SSPk(s,σ,p), so this is
SSP2(3,2,2), as OrdinaryDiffEqSDIRK and ClimaTimeSteppers (`SSP322`) have
it; SSP2(3,3,2) is [`IMEXSSP2332`](@ref). The measured properties are in
`CODE.md`, "Tableaus".
"""
function IMEXSSP2322()
    # Pareschi & Russo (2005), IMEX-SSP2(3,2,2), the stiffly accurate
    # scheme (Table 3 of arXiv:1009.2757).
    q(x) = Rational{BigInt}(x)
    Ã = q.([0 0 0
            0 0 0
            0 1 0])
    b̃ = q.([0, 1 // 2, 1 // 2])
    A = q.([1//2 0 0
            -1//2 1//2 0
            0 1//2 1//2])
    b = q.([0, 1 // 2, 1 // 2])
    return IMEXTableau{Rational{BigInt}}("SSP2(3,2,2)", Ã, b̃, A, b)
end

"""
    IMEXSSP2332()

SSP2(3,3,2) of Pareschi & Russo (2005): three stages, all explicit-used
and all solving, second order, with the diagonal `(1/4, 1/4, 1/3)` and a
stiffly accurate implicit part. The explicit part is the three-stage
second-order SSP method (SSP coefficient 2). It is rational, and so held
exactly as `Rational{BigInt}`. Neither OrdinaryDiffEqSDIRK nor
ClimaTimeSteppers has it, so it has no oracle; its coefficients are
checked against the paper (Table 4 of arXiv:1009.2757). Not to be
confused with
[`IMEXSSP2322`](@ref), SSP2(3,2,2). The measured properties are in
`CODE.md`, "Tableaus".
"""
function IMEXSSP2332()
    # Pareschi & Russo (2005), IMEX-SSP2(3,3,2), the stiffly accurate
    # scheme with three explicit stages (Table 4 of arXiv:1009.2757); in
    # neither upstream.
    q(x) = Rational{BigInt}(x)
    Ã = q.([0 0 0
            1//2 0 0
            1//2 1//2 0])
    b̃ = q.([1 // 3, 1 // 3, 1 // 3])
    A = q.([1//4 0 0
            0 1//4 0
            1//3 1//3 1//3])
    b = q.([1 // 3, 1 // 3, 1 // 3])
    return IMEXTableau{Rational{BigInt}}("SSP2(3,3,2)", Ã, b̃, A, b)
end

"""
    IMEXSSP3332()

SSP3(3,3,2) of Pareschi & Russo (2005): three stages, all explicit-used
and all solving, second order, with the diagonal `γ = 1 − 1/√2`. The
explicit part is the three-stage third-order SSP method of Shu & Osher
(SSP coefficient 1). Returns an [`IMEXTableau`](@ref) of 256-bit
`BigFloat`; the measured properties are in `CODE.md`, "Tableaus".
"""
function IMEXSSP3332()
    # Pareschi & Russo (2005), IMEX-SSP3(3,3,2), the L-stable scheme (Table 5
    # of arXiv:1009.2757).
    return with_coefficient_precision() do
        γ = gamma_sqrt2()
        z = zero(BigFloat)
        o = one(BigFloat)
        Ã = [z z z
             o z z
             o/4 o/4 z]
        b̃ = [o / 6, o / 6, 2o / 3]
        A = [γ z z
             1-2γ γ z
             o/2-γ z γ]
        b = [o / 6, o / 6, 2o / 3]
        return IMEXTableau{BigFloat}("SSP3(3,3,2)", Ã, b̃, A, b)
    end
end

"""
    IMEXSSP3433()

SSP3(4,3,3) of Pareschi & Russo (2005): four stages, all solving, of
which the first is not explicit-used; third order. The explicit part is
the three-stage third-order SSP method of Shu & Osher on stages 2–4 (SSP
coefficient 1). It is the intended production scheme.

The paper prints `α, β, η` to 14 digits. Here they are in closed form:
`α = (9 − √57)/6`, `β = α/4` and `η = (1 − 2α)/4`, the values that satisfy
the order-3 conditions and make `R(∞) = 0` ("SSP3(4,3,3) in closed form"
in `CODE.md`). The implicit part is not stiffly accurate. Returns an
[`IMEXTableau`](@ref) of 256-bit `BigFloat`; the measured properties are
in `CODE.md`, "Tableaus".
"""
function IMEXSSP3433()
    # Pareschi & Russo (2005), IMEX-SSP3(4,3,3), the L-stable scheme (Table 6
    # of arXiv:1009.2757), with the printed
    # α = 0.24169426078821, β = 0.06042356519705, η = 0.12915286960590
    # replaced by their closed forms.
    return with_coefficient_precision() do
        α = (9 - sqrt(BigFloat(57))) / 6
        β = α / 4
        η = (1 - 2α) / 4
        z = zero(BigFloat)
        o = one(BigFloat)
        Ã = [z z z z
             z z z z
             z o z z
             z o/4 o/4 z]
        b̃ = [z, o / 6, o / 6, 2o / 3]
        A = [α z z z
             -α α z z
             z 1-α α z
             β η o/2-β-η-α α]
        b = [z, o / 6, o / 6, 2o / 3]
        return IMEXTableau{BigFloat}("SSP3(4,3,3)", Ã, b̃, A, b)
    end
end

"""
    ARS222()

ARS(2,2,2) of Ascher, Ruuth & Spiteri (1997), §2.6: a trivial first stage
(`a_11 = 0`, so `U₁ = uⁿ`), then two solving stages, second order, with
the diagonal `γ = 1 − 1/√2` and `δ = 1 − 1/(2γ) = −1/√2`. Both parts are
stiffly accurate: `b` and `b̃` are the last rows of `A` and `Ã`, so the
explicit part uses stages 1 and 2 only. Returns an [`IMEXTableau`](@ref)
of 256-bit `BigFloat`; the measured properties are in `CODE.md`,
"Tableaus".
"""
function ARS222()
    # Ascher, Ruuth & Spiteri (1997), §2.6, p. 158: the two-stage L-stable
    # DIRK of §2.5 with γ = (2 − √2)/2, and the explicit weights
    # (δ, 1 − δ, 0), δ = 1 − 1/(2γ), that make it stiffly accurate.
    return with_coefficient_precision() do
        γ = gamma_sqrt2()
        δ = 1 - 1 / (2γ)
        z = zero(BigFloat)
        Ã = [z z z
             γ z z
             δ 1-δ z]
        b̃ = [δ, 1 - δ, z]
        A = [z z z
             z γ z
             z 1-γ γ]
        b = [z, 1 - γ, γ]
        return IMEXTableau{BigFloat}("ARS(2,2,2)", Ã, b̃, A, b)
    end
end

"""
    ARS443()

ARS(4,4,3) of Ascher, Ruuth & Spiteri (1997), §2.8: a trivial first stage, then
four solving stages with the diagonal `1/2`, third order. Both parts are
stiffly accurate: `b` and `b̃` are the last rows of `A` and `Ã`, so the
explicit part uses stages 1–4 only, the four explicit stages of the name.
It is rational, and so held exactly as `Rational{BigInt}`.

These are the paper's weights (§2.8, p. 160), as in ClimaTimeSteppers.
OrdinaryDiffEqSDIRK 2.9.6's `ARS443` has `b̃ = b` instead, which is not
the paper's and evaluates the explicit part at stage 5 too (`CODE.md`,
"Tableaus", "Cross-checks"). The measured properties are in `CODE.md`,
"Tableaus".
"""
function ARS443()
    # Ascher, Ruuth & Spiteri (1997), §2.8, p. 160. The paper prints the
    # explicit weights b̃ = (1/4, 7/4, 3/4, −7/4, 0), identical to the last
    # row of Ã, and b = (0, 3/2, −3/2, 1/2, 1/2), the last row of A.
    q(x) = Rational{BigInt}(x)
    Ã = q.([0 0 0 0 0
            1//2 0 0 0 0
            11//18 1//18 0 0 0
            5//6 -5//6 1//2 0 0
            1//4 7//4 3//4 -7//4 0])
    b̃ = q.([1 // 4, 7 // 4, 3 // 4, -7 // 4, 0])
    A = q.([0 0 0 0 0
            0 1//2 0 0 0
            0 1//6 1//2 0 0
            0 -1//2 1//2 1//2 0
            0 3//2 -3//2 1//2 1//2])
    b = q.([0, 3 // 2, -3 // 2, 1 // 2, 1 // 2])
    return IMEXTableau{Rational{BigInt}}("ARS(4,4,3)", Ã, b̃, A, b)
end

# The purely explicit tableaus ("Explicit tableaus" in `CODE.md`): an
# additive method whose implicit part is zero, `A = 0` and `b = 0`. Every
# stage is explicit-used and none solves, so `solve_imp!` is never called
# and may be `nothing`. Stage 1 has an empty row and is the trivial stage:
# `f_exp!` reads `uⁿ` itself, which only the step limiter has limited.

# A purely explicit tableau from its explicit part, held exactly.
function explicit_tableau(name, Ã, b̃)
    q(x) = Rational{BigInt}(x)
    s = length(b̃)
    return IMEXTableau{Rational{BigInt}}(name, q.(Ã), q.(b̃), zeros(Rational{BigInt}, s, s),
                                         zeros(Rational{BigInt}, s))
end

"""
    Euler()

The explicit Euler method, `uⁿ⁺¹ = uⁿ + Δt f(uⁿ, tⁿ)`: one stage, first
order, SSP coefficient 1. It is for debugging. The implicit part is zero,
so `solve_imp!` is never called and may be `nothing`. Its one stage reads
`uⁿ` itself, so only the step limiter limits what `f_exp!` sees (`CODE.md`,
"Explicit tableaus"). OrdinaryDiffEq's name, as the other named tableaus
are; it clashes with OrdinaryDiffEqLowOrderRK's `Euler`.
"""
function Euler()
    return explicit_tableau("Euler", fill(0, 1, 1), [1])
end

"""
    RK4()

The classical fourth-order Runge–Kutta method of Kutta (1901): four
stages, `c̃ = (0, 1/2, 1/2, 1)`, `b̃ = (1/6, 1/3, 1/3, 1/6)`, fourth order,
SSP coefficient 0. The implicit part is zero, so `solve_imp!` is never
called and may be `nothing`. Its first stage reads `uⁿ` itself, so a
caller who limits every right-hand-side input passes the same function as
the stage and the step limiter (`CODE.md`, "Explicit tableaus").
OrdinaryDiffEq's name, as the other named tableaus are; it clashes with
OrdinaryDiffEqLowOrderRK's `RK4`.
"""
function RK4()
    # Kutta (1901); Hairer, Nørsett & Wanner, Solving ODEs I, Table II.1.2.
    return explicit_tableau("RK4",
                            [0 0 0 0
                             1//2 0 0 0
                             0 1//2 0 0
                             0 0 1 0],
                            [1 // 6, 1 // 3, 1 // 3, 1 // 6])
end

"""
    SSPRK33()

SSPRK(3,3), the three-stage third-order SSP method of Shu & Osher (1988),
in Butcher form: `c̃ = (0, 1, 1/2)`, `b̃ = (1/6, 1/6, 2/3)`, SSP coefficient
1. It is exactly the explicit part of [`IMEXSSP3332`](@ref). The implicit
part is zero, so `solve_imp!` is never called and may be `nothing`. Its
first stage reads `uⁿ` itself, so a caller who limits every
right-hand-side input passes the same function as the stage and the step
limiter (`CODE.md`, "Explicit tableaus"). OrdinaryDiffEq's name, as the
other named tableaus are; it clashes with OrdinaryDiffEqSSPRK's `SSPRK33`.
"""
function SSPRK33()
    # Shu & Osher (1988), eq. (2.18), in Butcher form: the Shu–Osher stages
    # u⁽¹⁾ = uⁿ + Δt L(uⁿ), u⁽²⁾ = ¾uⁿ + ¼(u⁽¹⁾ + Δt L(u⁽¹⁾)) and
    # uⁿ⁺¹ = ⅓uⁿ + ⅔(u⁽²⁾ + Δt L(u⁽²⁾)).
    return explicit_tableau("SSPRK(3,3)",
                            [0 0 0
                             1 0 0
                             1//4 1//4 0],
                            [1 // 6, 1 // 6, 2 // 3])
end

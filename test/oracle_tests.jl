import OrdinaryDiffEqSDIRK as ODE
import OrdinaryDiffEqLowOrderRK as LowRK
import OrdinaryDiffEqSSPRK as SSPRK
using IMEXRungeKutta: IMEXProblem, IMEXTableau
using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP3332, IMEXSSP3433, ARS222, ARS443
using IMEXRungeKutta: Euler, RK4, SSPRK33, ImplicitEuler
using LinearAlgebra: I, mul!

# OrdinaryDiffEqSDIRK as an oracle ("Why not an existing package" and
# "Testing", Oracle, in `CODE.md`). Its names clash with ours, so it is
# imported and qualified.
#
# The problem is linear and real, `u′ = Lu + Mu + a cos(3t) v` on three
# components, with `Lu` implicit and the rest explicit. Upstream's
# `SplitODEProblem(f1, f2, …)` treats **`f1` implicitly** and `f2`
# explicitly, the opposite order from `IMEXProblem(f_exp!, solve_imp!, …)`.
# Its default nonlinear solver, Newton with a ForwardDiff Jacobian, is
# exact to round-off on a linear `g`. Our stage solve is
# `U = (I − γΔt L) \ u★`.
#
# The restrictions (`CODE.md`):
# - the state is real (upstream's AD Jacobian rejects a complex one);
# - SSP2(3,3,2) is in neither upstream, and has no oracle;
# - OrdinaryDiffEqSDIRK 2.9.7 or later: earlier releases evaluate the last
#   explicit stage at `t + Δt` (SciML/OrdinaryDiffEq.jl#4620) and have
#   another `b̃` for `ARS443`.

const ORACLE_L = [-2.0 0.5 0.0; 0.3 -1.5 0.2; 0.0 0.4 -3.0]
const ORACLE_M = [0.0 1.0 -0.5; -1.0 0.0 0.3; 0.5 -0.3 0.0]
const ORACLE_v = [1.0, -0.5, 0.25]
const ORACLE_u0 = [1.0, 0.5, -0.25]

# The amplitude `a` of the `t`-dependent term is `p`: 0 for the autonomous
# comparison, 1 for the `t`-dependent one. One function for both keeps
# upstream's solver compiled once per algorithm.
const AUTONOMOUS = 0.0
const T_DEPENDENT = 1.0

oracle_g!(du, u, a, t) = (mul!(du, ORACLE_L, u); nothing)
function oracle_f!(du, u, a, t)
    mul!(du, ORACLE_M, u)
    du .+= (a * cos(3t)) .* ORACLE_v
    return nothing
end
oracle_imp!(U, u★, γΔt, a, t) = (U .= (I - γΔt * ORACLE_L) \ u★; nothing)

function upstream_run(alg, a; dt = 0.1, n = 10)
    prob = ODE.SplitODEProblem(oracle_g!, oracle_f!, copy(ORACLE_u0), (0.0, n * dt), a)
    sol = ODE.solve(prob, alg; dt, adaptive = false, save_everystep = false)
    return sol.u[end]
end
function our_run(tab, a; dt = 0.1, n = 10)
    prob = IMEXProblem(oracle_f!, oracle_imp!, copy(ORACLE_u0), (0.0, n * dt), a)
    return solve(prob, tab; dt).u
end
oracle_difference(alg, tab, a; kw...) =
    maximum(abs, upstream_run(alg, a; kw...) - our_run(tab, a; kw...))

# SSP3(4,3,3) from the 14 printed digits, as upstream holds them, held
# exactly as decimals here, to measure how much of the difference is
# theirs.
const SSP3433_PRINTED = let t = IMEXSSP3433()
    α = 24169426078821 // 10^14
    β = 6042356519705 // 10^14
    η = 12915286960590 // 10^14
    A = [α 0 0 0; -α α 0 0; 0 1-α α 0; β η 1//2-β-η-α α]
    IMEXTableau("SSP3(4,3,3), 14 digits", t.Ã, t.b̃, A, t.b)
end

# The pairs, each with its last explicit abscissa `c̃_s`.
const ORACLE_TABLE = [
    (alg = ODE.IMEXSSP222(), make = IMEXSSP222, c̃s = 1),
    (alg = ODE.IMEXSSP2322(), make = IMEXSSP2322, c̃s = 1),
    (alg = ODE.IMEXSSP3332(), make = IMEXSSP3332, c̃s = 1 // 2),
    (alg = ODE.IMEXSSP3433(), make = IMEXSSP3433, c̃s = 1 // 2),
    (alg = ODE.ARS222(), make = ARS222, c̃s = 1),
    (alg = ODE.ARS443(), make = ARS443, c̃s = 1),
]

# A transcription error in a tableau, or a stage the integrator weights
# differently from a reference implementation, shows as a difference far
# above round-off. Measured over ten steps of Δt = 0.1 against 2.9.7:
# 5.6e−17, 2.5e−16, 1.4e−16, 5.3e−16, 2.8e−17 and 2.8e−16, in the order of
# the table; the largest is SSP3(4,3,3), whose 14-digit coefficients
# upstream uses.
@testset "Each tableau upstream has agrees with it to 1e−12 over ten steps" begin
    @test length(ORACLE_TABLE) == 6
    for spec in ORACLE_TABLE
        @test oracle_difference(spec.alg, spec.make(), AUTONOMOUS) < 1e-12
    end
end

# A mistimed explicit stage shows only where the explicit part depends on
# `t`, and the last explicit abscissa only where `c̃_s ≠ 1`: SSP3(3,3,2)
# and SSP3(4,3,3). Upstream got that one wrong until 2.9.7
# (https://github.com/SciML/OrdinaryDiffEq.jl/issues/4620), 0.0225 off for
# both; the table must keep covering it. Measured against 2.9.7: 1.4e−16,
# 2.6e−16, 6.9e−17, 4.5e−16, 1.4e−16 and 6.6e−17, in the order of the
# table.
@testset "Each tableau upstream has agrees with it with a t-dependent explicit part, c̃_s ≠ 1 included" begin
    @test any(spec -> spec.c̃s != 1, ORACLE_TABLE)
    for spec in ORACLE_TABLE
        tab = spec.make()
        @test tab.c̃[end] == spec.c̃s
        @test oracle_difference(spec.alg, tab, T_DEPENDENT) < 1e-12
    end
end

# `CODE.md` says upstream's 14-digit SSP3(4,3,3) differs from the closed
# form by about 1e−15. If it differed by more, the 1e−12 above would be
# hiding a real disagreement. Measured: 4.9e−16 between ours and the
# printed digits, and 1.9e−16 between upstream and the printed digits.
@testset "SSP3(4,3,3)'s 14 printed digits change ten steps by about 1e−15" begin
    d = maximum(abs, our_run(IMEXSSP3433(), AUTONOMOUS) - our_run(SSP3433_PRINTED, AUTONOMOUS))
    @test d < 1e-14
    @test oracle_difference(ODE.IMEXSSP3433(), SSP3433_PRINTED, AUTONOMOUS) < 1e-12
end

# The purely explicit tableaus against OrdinaryDiffEqLowOrderRK's `Euler`
# and `RK4` and OrdinaryDiffEqSSPRK's `SSPRK33`, whose names clash with
# ours as the IMEX ones do ("Explicit tableaus" in `CODE.md`). The same
# linear problem, all of it explicit, `u′ = (L + M)u + a cos(3t) v`, with no
# stage solver; the `t`-dependent case sees every explicit abscissa.
function oracle_explicit_f!(du, u, a, t)
    mul!(du, ORACLE_L + ORACLE_M, u)
    du .+= (a * cos(3t)) .* ORACLE_v
    return nothing
end
function upstream_explicit_run(alg, a; dt = 0.1, n = 10)
    prob = LowRK.ODEProblem(oracle_explicit_f!, copy(ORACLE_u0), (0.0, n * dt), a)
    sol = LowRK.solve(prob, alg; dt, adaptive = false, save_everystep = false)
    return sol.u[end]
end
function our_explicit_run(tab, a; dt = 0.1, n = 10)
    prob = IMEXProblem(oracle_explicit_f!, nothing, copy(ORACLE_u0), (0.0, n * dt), a)
    return solve(prob, tab; dt).u
end

const EXPLICIT_ORACLE_TABLE = [
    (alg = LowRK.Euler(), make = Euler),
    (alg = LowRK.RK4(), make = RK4),
    (alg = SSPRK.SSPRK33(), make = SSPRK33),
]

# A transcription error or a mistimed stage in an explicit tableau shows
# as a difference far above round-off. Measured over ten steps of
# Δt = 0.1, autonomous and `t`-dependent: 3.5e−17 and 9.4e−17 for Euler,
# 2.8e−17 and 3.8e−17 for RK4, 2.8e−17 and 1.2e−16 for SSPRK(3,3).
@testset "Each explicit tableau agrees with OrdinaryDiffEq to 1e−12 over ten steps, with t or not" begin
    for spec in EXPLICIT_ORACLE_TABLE, a in (AUTONOMOUS, T_DEPENDENT)
        d = maximum(abs, upstream_explicit_run(spec.alg, a) - our_explicit_run(spec.make(), a))
        @test d < 1e-12
    end
end

# Backward Euler against OrdinaryDiffEqSDIRK's `ImplicitEuler`, whose name
# clashes with ours ("Implicit Euler" in `CODE.md`). The same linear
# problem, all of it implicit, `u′ = (L + M)u + a cos(3t) v`, with
# `f_exp! = nothing`; its stage solve is
# `U = (I − Δt(L + M)) \ (u★ + Δt a cos(3t) v)` at `t = tⁿ + Δt`, which the
# `t`-dependent case checks. Upstream's Newton iteration is exact to
# round-off on a linear `g`.
function oracle_implicit_imp!(U, u★, γΔt, a, t)
    U .= (I - γΔt * (ORACLE_L + ORACLE_M)) \ (u★ .+ (γΔt * a * cos(3t)) .* ORACLE_v)
    return nothing
end
function upstream_implicit_run(a; dt = 0.1, n = 10)
    prob = ODE.ODEProblem(oracle_explicit_f!, copy(ORACLE_u0), (0.0, n * dt), a)
    sol = ODE.solve(prob, ODE.ImplicitEuler(); dt, adaptive = false, save_everystep = false)
    return sol.u[end]
end
function our_implicit_run(a; dt = 0.1, n = 10)
    prob = IMEXProblem(nothing, oracle_implicit_imp!, copy(ORACLE_u0), (0.0, n * dt), a)
    return solve(prob, ImplicitEuler(); dt).u
end

# A stage solve at the wrong time, or an increment taken from the wrong
# arrays, shows as a difference far above round-off. Measured over ten
# steps of Δt = 0.1, autonomous and `t`-dependent: 8.3e−17 and 9.7e−17.
@testset "ImplicitEuler agrees with OrdinaryDiffEqSDIRK's to 1e−12 over ten steps, with t or not" begin
    for a in (AUTONOMOUS, T_DEPENDENT)
        d = maximum(abs, upstream_implicit_run(a) - our_implicit_run(a))
        @test d < 1e-12
    end
end

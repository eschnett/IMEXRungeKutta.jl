import OrdinaryDiffEqSDIRK as ODE
import OrdinaryDiffEqLowOrderRK as LowRK
import OrdinaryDiffEqSSPRK as SSPRK
using IMEXRungeKutta: IMEXProblem, IMEXTableau
using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP3332, IMEXSSP3433, ARS222, ARS443
using IMEXRungeKutta: Euler, RK4, SSPRK33
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
# - upstream evaluates the last explicit stage at `t + Δt`
#   (SciML/OrdinaryDiffEq.jl#4620), which matters only when `c̃_s ≠ 1` and
#   `f` depends on `t`;
# - upstream's `ARS443` has `b̃ = b`, not the paper's, so it is compared
#   with a tableau built with that `b̃`;
# - SSP2(3,3,2) is in neither upstream, and has no oracle.

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

const ARS443_UPSTREAM_B̃ = let t = ARS443()
    IMEXTableau("ARS(4,4,3), upstream's b̃ = b", t.Ã, t.b, t.A, t.b)
end

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
    (alg = ODE.ARS443(), make = () -> ARS443_UPSTREAM_B̃, c̃s = 1),
]

# A transcription error in a tableau, or a stage the integrator weights
# differently from a reference implementation, shows as a difference far
# above round-off. Measured over ten steps of Δt = 0.1: 5.6e−17, 2.5e−16,
# 1.4e−16, 5.3e−16, 2.8e−17 and 8.3e−17, in the order of the table; the
# largest is SSP3(4,3,3), whose 14-digit coefficients upstream uses.
@testset "Each tableau upstream has agrees with it to 1e−12 over ten steps" begin
    @test length(ORACLE_TABLE) == 6
    for spec in ORACLE_TABLE
        @test oracle_difference(spec.alg, spec.make(), AUTONOMOUS) < 1e-12
    end
end

# The last explicit abscissa is what #4620 gets wrong. Where `c̃_s = 1`,
# upstream's `t + Δt` is right, and the comparison with a `t`-dependent
# explicit part must pass; a failure there would be ours. Where
# `c̃_s ≠ 1` it is `@test_broken` until #4620 is fixed
# (https://github.com/SciML/OrdinaryDiffEq.jl/issues/4620). A fix upstream
# turns it into an unexpected pass, which fails the suite and is noticed.
# Measured: at most 3.1e−16 where `c̃_s = 1`, and 0.0225 for SSP3(3,3,2)
# and SSP3(4,3,3).
@testset "With a t-dependent explicit part: agreement where c̃_s = 1, #4620 where not" begin
    for spec in ORACLE_TABLE
        tab = spec.make()
        @test tab.c̃[end] == spec.c̃s
        d = oracle_difference(spec.alg, tab, T_DEPENDENT)
        if spec.c̃s == 1
            @test d < 1e-12
        else
            @test d > 1e-3    # the mistiming, not round-off
            @test_broken d < 1e-12
        end
    end
end

# `CODE.md` says upstream's 14-digit SSP3(4,3,3) differs from the closed
# form by about 1e−15. If it differed by more, the 1e−12 above would be
# hiding a real disagreement.
@testset "SSP3(4,3,3)'s 14 printed digits change ten steps by about 1e−15" begin
    d = maximum(abs, our_run(IMEXSSP3433(), AUTONOMOUS) - our_run(SSP3433_PRINTED, AUTONOMOUS))
    @test d < 1e-14
    @test oracle_difference(ODE.IMEXSSP3433(), SSP3433_PRINTED, AUTONOMOUS) < 1e-12
end

# Our ARS(4,4,3) is the paper's, and upstream's is not ("Cross-checks" in
# `CODE.md`). Both are third order, so they differ by O(Δt⁴) per step: a
# change that made the difference O(Δt³) would be a wrong tableau on one
# side. Measured: 1.95e−5 in one step of Δt = 0.1, and 5.2e−10 at
# Δt = 0.00625, with local slopes 3.59, 3.78, 3.89 and 3.94 between;
# 2.5e−5 over ten steps of Δt = 0.1.
@testset "Our ARS(4,4,3) differs from upstream's by O(Δt⁴) per step" begin
    one_step(tab, dt) = our_run(tab, AUTONOMOUS; dt, n = 1)
    dts = (0.025, 0.0125, 0.00625)
    diffs = [maximum(abs, one_step(ARS443(), dt) - one_step(ARS443_UPSTREAM_B̃, dt))
             for dt in dts]
    @test abs(fitted_order(dts, diffs) - 4) < 0.15
    # Upstream's own ARS443 is the b̃ = b variant, not ours.
    @test oracle_difference(ODE.ARS443(), ARS443(), AUTONOMOUS) > 1e-6
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

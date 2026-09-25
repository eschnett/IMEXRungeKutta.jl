using IMEXRungeKutta: IMEXProblem
using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP2332, IMEXSSP3332, IMEXSSP3433
using IMEXRungeKutta: ARS222, ARS443
using IMEXRungeKutta: Euler, RK4, SSPRK33

# Total variation under upwind advection with relaxation ("Testing", SSP,
# in `CODE.md`):
#
#     u_t + u_x = −(u − φ)/ε   on the periodic [0, 1),
#
# with first-order upwind differences on 100 cells, explicit, and the
# relaxation implicit, `U = (ε u★ + γΔt φ)/(ε + γΔt)`. The profile `φ` is
# fixed in time and is the initial square wave itself, `u⁰ = φ = 1` on
# [0.25, 0.5) and 0 elsewhere, so `TV(u⁰) = TV(φ) = 2`. Forward Euler on the
# advection is TVD for `Δt ≤ Δt_FE = Δx`.
#
# The exact solution is a positive average of translates of `u⁰` and of
# `φ`, so `TV(u(t + τ)) ≤ max(TV(u(t)), TV(φ))`, and with `u⁰ = φ` the
# total variation never exceeds its initial value. That is the property
# asked of a step: `TV(uⁿ⁺¹) ≤ max(TV(uⁿ), TV(φ))`, to a relative 1e−12.
# Without relaxation (`ε = ∞`, `solve_imp!` leaves `U = u★`) it is
# `TV(uⁿ⁺¹) ≤ TV(uⁿ)`, the TVD property itself.
#
# `C = Δt/Δt_FE` is the largest value at which 50 steps keep it, found by
# bisection on [0, 4] to 1e−4. The measured values are in `CODE.md`, "SSP
# and total variation" (measured in step 3).

const SSP_N = 100
const SSP_Δx = 1.0 / SSP_N
const SSP_φ = [0.25 ≤ (i - 0.5) / SSP_N < 0.5 ? 1.0 : 0.0 for i in 1:SSP_N]

# The upwind difference, `a = 1`. The one loop over the state here is the
# test problem's, not the integrator's.
function ssp_upwind!(du, u, p, t)
    n = length(u)
    du[1] = -(u[1] - u[n]) / SSP_Δx
    for i in 2:n
        du[i] = -(u[i] - u[i - 1]) / SSP_Δx
    end
    return nothing
end
function ssp_relax_imp!(U, u★, γΔt, ε, t)
    isinf(ε) && return nothing
    U .= (ε .* u★ .+ γΔt .* SSP_φ) ./ (ε + γΔt)
    return nothing
end

ssp_tv(u) = abs(u[1] - u[end]) + sum(abs(u[i] - u[i - 1]) for i in 2:length(u))

# Whether 50 steps at `Δt = C Δx` keep the total variation as above, and
# the largest excess over the bound in the first step.
function ssp_run(tab, C, ε; nsteps = 50)
    dt = C * SSP_Δx
    integ = init(IMEXProblem(ssp_upwind!, ssp_relax_imp!, copy(SSP_φ), (0.0, nsteps * dt), ε),
                 tab; dt)
    tvφ = ssp_tv(SSP_φ)
    tv = ssp_tv(integ.u)
    ok = true
    first_excess = 0.0
    while integ.nstep < integ.nsteps
        step!(integ)
        bound = isinf(ε) ? tv : max(tv, tvφ)
        tv = ssp_tv(integ.u)
        integ.nstep == 1 && (first_excess = tv - bound)
        ok &= tv ≤ bound * (1 + 1e-12)
    end
    return (; ok, first_excess)
end

function largest_C(tab, ε; hi = 4.0, tol = 1e-4)
    ssp_run(tab, hi, ε).ok && return hi
    lo = 0.0
    while hi - lo > tol
        mid = (lo + hi) / 2
        ssp_run(tab, mid, ε).ok ? (lo = mid) : (hi = mid)
    end
    return lo
end

# Per tableau: the SSP coefficient of the explicit part (step 1), and `C`
# measured in step 3 without relaxation and with `ε = 1e−2` (`Δt/ε = C`).
# `κ = −wᵀ𝟙 = bᵀA⁻¹c̃ − 1` is the stiff-limit coefficient of
# `ap_tests.jl`, nonzero for the three schemes whose implicit part is not
# stiffly accurate.
const SSP_TABLE = [
    (make = IMEXSSP222, ssp = 1, C∞ = 1.0, C₁₀₋₂ = 0.7158, κ = 0.70710678),
    (make = IMEXSSP2322, ssp = 1, C∞ = 1.0, C₁₀₋₂ = 0.7088, κ = 0.0),
    (make = IMEXSSP2332, ssp = 2, C∞ = 2.0, C₁₀₋₂ = 1.4709, κ = 0.0),
    (make = IMEXSSP3332, ssp = 1, C∞ = 1.0, C₁₀₋₂ = 0.8613, κ = 0.70710678),
    (make = IMEXSSP3433, ssp = 1, C∞ = 1.0, C₁₀₋₂ = 0.6840, κ = 0.28436465),
    (make = ARS222, ssp = 0, C∞ = 1.0, C₁₀₋₂ = 0.7286, κ = 0.0),
    (make = ARS443, ssp = 0, C∞ = 0.0021, C₁₀₋₂ = 0.0021, κ = 0.0),
]

# A stage-arithmetic or tableau change that broke the explicit part's
# monotonicity would show here first. On this linear problem only the
# explicit stability polynomial matters: `1 + z + z²/2` for SSP2(2,2,2),
# SSP2(3,2,2) and ARS(2,2,2) alike, whose threshold is 1 although
# ARS(2,2,2)'s SSP coefficient is 0; `1 + z + z²/2 + z³/12` for
# SSP2(3,3,2), threshold 2; and for ARS(4,4,3) a `z⁴` coefficient of
# −7/288, so no positive threshold. Its 0.0021 is where the TV growth,
# `O(C⁴)`, falls below the 1e−12 tolerance.
@testset "Without relaxation, C is the explicit part's linear threshold" begin
    @test length(SSP_TABLE) == 7
    for spec in SSP_TABLE
        C = largest_C(spec.make(), Inf)
        @test abs(C - spec.C∞) < 1e-3
        spec.ssp > 0 && @test abs(C - spec.ssp) < 1e-3
    end
    @test !ssp_run(ARS443(), 0.01, Inf).ok
end

# The purely explicit tableaus have no relaxation to run with; without it
# their threshold is that of their stability polynomial, and equals the SSP
# coefficient where it is positive ("Explicit tableaus" in `CODE.md`).
# RK4's SSP coefficient is 0, but on this linear problem only its
# stability polynomial matters, `1 + z + z²/2 + z³/6 + z⁴/24`, whose
# threshold is 1.
const EXPLICIT_SSP_TABLE = [
    (make = Euler, ssp = 1, C∞ = 1.0),
    (make = RK4, ssp = 0, C∞ = 1.0),
    (make = SSPRK33, ssp = 1, C∞ = 1.0),
]
@testset "An explicit tableau's C is its linear threshold, the SSP coefficient where > 0" begin
    for spec in EXPLICIT_SSP_TABLE
        C = largest_C(spec.make(), Inf)
        @test abs(C - spec.C∞) < 1e-3
        spec.ssp > 0 && @test abs(C - spec.ssp) < 1e-3
    end
end

# Relaxation as fast as the advection: a regression value per tableau.
# It is below the explicit threshold for every tableau.
@testset "With relaxation at ε = 1e−2, C is the recorded one" begin
    for spec in SSP_TABLE
        C = largest_C(spec.make(), 1e-2)
        @test abs(C - spec.C₁₀₋₂) < 1e-3
        @test C ≤ spec.C∞ + 1e-3
    end
end

# The total variation's largest excess over `TV(u⁰)` in 50 steps.
function ssp_excess(tab, C, ε; nsteps = 50)
    dt = C * SSP_Δx
    integ = init(IMEXProblem(ssp_upwind!, ssp_relax_imp!, copy(SSP_φ), (0.0, nsteps * dt), ε),
                 tab; dt)
    tv0 = ssp_tv(SSP_φ)
    excess = 0.0
    while integ.nstep < integ.nsteps
        step!(integ)
        excess = max(excess, ssp_tv(integ.u) - tv0)
    end
    return excess
end

# In the stiff limit each stage solve puts `U` on `φ`, and a step ends
# `Δt wᵀF` off it ("Where a step ends in the stiff limit" in `CODE.md`),
# here `F = −(φᵢ − φᵢ₋₁)/Δx` at each jump. For the three schemes with
# `κ ≠ 0` that is an overshoot of `κC` on each side of both jumps, so the
# first step raises the total variation by exactly `4κC`, at every `C`:
# their `C` is 0. The four whose implicit part is stiffly accurate end on
# `φ` to O(ε), and so does their total variation: SSP2(3,2,2)'s rises by
# `8(C − 1)ε/Δx` for `C > 1` (800ε at `C = 2`), which puts its bisected
# `C` at 1.0024 for ε = 1e−12 (3.4999 for ε = 1e−15); the other three
# stay at `TV(u⁰)` to round-off up to `C = 4`. A regression in the stage
# contract, or a tableau change, would move either.
const SSP_STIFF_C = Dict("SSP2(2,2,2)" => 0.0, "SSP2(3,2,2)" => 1.0024,
                         "SSP2(3,3,2)" => 4.0, "SSP3(3,3,2)" => 0.0, "SSP3(4,3,3)" => 0.0,
                         "ARS(2,2,2)" => 4.0, "ARS(4,4,3)" => 4.0)

@testset "Stiff relaxation (ε = 1e−12): TV rises by 4κC where κ ≠ 0, by O(ε) elsewhere" begin
    ε = 1e-12
    for spec in SSP_TABLE
        tab = spec.make()
        @test abs(largest_C(tab, ε) - SSP_STIFF_C[tab.name]) < 1e-3
        for C in (0.5, 1.0, 2.0, 4.0)
            if spec.κ > 0
                @test abs(ssp_run(tab, C, ε).first_excess / (4 * spec.κ * C) - 1) < 1e-6
            else
                excess = ssp_excess(tab, C, ε)
                @test excess ≤ 1e4 * ε
                C ≤ 1 && @test excess ≤ 1e-12
            end
        end
        spec.κ > 0 && @test !ssp_run(tab, 1e-3, ε).ok
    end
end

using IMEXRungeKutta: IMEXProblem

# A smoke test of the order on a real problem, so that step 2 cannot ship
# an integrator that passes only mocks. Step 3 of `PLAN.md` does the full
# validation.
#
# `u′ = −u + cos t`, with the implicit part `−u`, whose stage solve is
# `U = u★/(1 + γΔt)`, and the explicit part `cos t`, which depends on `t`:
# the mistimed last explicit stage of SciML/OrdinaryDiffEq.jl#4620, at
# `tⁿ + Δt`, drops SSP3(3,3,2) and SSP3(4,3,3) to order 1.02 here
# (measured in step 2, by that mutation), and is invisible on a problem
# autonomous in `t`. A swap of `c̃` and `c` at every stage is not caught
# here: the coupling conditions `b̃ᵀc = 1/2` and `b̃ᵀc² = 1/3` make an `f`
# of `t` alone integrate the same with either. The call-time test in
# `mechanics_tests.jl` catches that. The exact solution is
# `u(t) = (cos t + sin t)/2 + (u(0) − 1/2) e^{−t}`.

f_forcing!(du, u, p, t) = (du .= cos(t); nothing)
solve_relax!(U, u★, γΔt, p, t) = (U .= u★ ./ (1 + γΔt); nothing)
exact_forced(t, u0) = (cos(t) + sin(t)) / 2 + (u0 - 1 / 2) * exp(-t)

# The error at t = 2 for u(0) = 1, per Δt.
function forced_errors(tab, dts)
    return map(dts) do dt
        prob = IMEXProblem(f_forcing!, solve_relax!, [1.0], (0.0, 2.0))
        integ = solve(prob, tab; dt)
        return abs(integ.u[1] - exact_forced(2.0, 1.0))
    end
end

# The observed order from the two finest steps.
observed_order(errs, dts) = log(errs[end - 1] / errs[end]) / log(dts[end - 1] / dts[end])

const SMOKE_ORDERS = [
    (make = IMEXSSP222, order = 2), (make = IMEXSSP2322, order = 2),
    (make = IMEXSSP2332, order = 2), (make = IMEXSSP3332, order = 2),
    (make = IMEXSSP3433, order = 3), (make = ARS222, order = 2), (make = ARS443, order = 3),
]

# An integrator that mis-weighted a stage, mistimed the last explicit
# stage, or recovered the increment wrongly would lose at least one order
# here.
@testset "The observed order on u′ = −u + cos t is the tableau's, ±0.2" begin
    @test length(SMOKE_ORDERS) == 7
    dts = (0.1, 0.05, 0.025)
    for spec in SMOKE_ORDERS
        errs = forced_errors(spec.make(), dts)
        p = observed_order(errs, dts)
        @test abs(p - spec.order) < 0.2
    end
    # PLAN.md's two named cases, asserted on their own.
    @test abs(observed_order(forced_errors(IMEXSSP222(), dts), dts) - 2) < 0.2
    @test abs(observed_order(forced_errors(IMEXSSP3433(), dts), dts) - 3) < 0.2
end

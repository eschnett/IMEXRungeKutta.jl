using IMEXRungeKutta: IMEXProblem
using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP2332, IMEXSSP3332, IMEXSSP3433
using IMEXRungeKutta: ARS222, ARS443

# The observed order of every tableau on two problems, each with an
# implicit part `−u` whose stage solve is `U = u★/(1 + γΔt)` ("Testing",
# Order, in `CODE.md`):
# - `u′ = iu − u`, with a complex state and the explicit part `iu`, which
#   depends on `u` and not on `t`: it sees how the two parts couple;
# - `u′ = −u + cos t`, whose explicit part depends on `t` alone: it sees
#   the explicit abscissae, which a problem autonomous in `t` does not
#   (the #4620 failure mode, SciML/OrdinaryDiffEq.jl#4620).
#
# Both run to `t = 1` with `Δt = 1/10, …, 1/160`, and the order is the
# least-squares slope of `log error` against `log Δt` over the finest
# three (`fitted_order`, in `problems.jl`). The measured values are in
# `CODE.md`, "Observed orders" (measured in step 3).

order_rotate!(du, u, p, t) = (du .= im .* u; nothing)
order_forcing!(du, u, p, t) = (du .= cos(t); nothing)
order_decay_imp!(U, u★, γΔt, p, t) = (U .= u★ ./ (1 + γΔt); nothing)

order_exact_rotate(t) = exp((im - 1) * t)
order_exact_forced(t) = (cos(t) + sin(t)) / 2 + exp(-t) / 2   # u(0) = 1

const ORDER_DTS = (1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160)

function order_errors(tab, f!, u0, exact)
    return map(ORDER_DTS) do dt
        integ = solve(IMEXProblem(f!, order_decay_imp!, [u0], (0.0, 1.0)), tab; dt)
        return abs(integ.u[1] - exact(1.0))
    end
end

finest_order(errs) = fitted_order(ORDER_DTS[(end - 2):end], errs[(end - 2):end])

# The tableaus, with the stated order and the orders measured in step 3
# (recorded, not asserted beyond the ±0.1 of the stated order).
const ORDER_TABLE = [
    (make = IMEXSSP222, order = 2, rotate = 2.004, forced = 2.002),
    (make = IMEXSSP2322, order = 2, rotate = 2.008, forced = 2.029),
    (make = IMEXSSP2332, order = 2, rotate = 2.003, forced = 2.001),
    (make = IMEXSSP3332, order = 2, rotate = 1.994, forced = 1.993),
    (make = IMEXSSP3433, order = 3, rotate = 3.007, forced = 3.006),
    (make = ARS222, order = 2, rotate = 2.004, forced = 2.009),
    (make = ARS443, order = 3, rotate = 3.005, forced = 3.001),
]

# A mis-weighted stage, a wrongly recovered increment or a coupling term
# summed with the wrong coefficient drops at least one order on the split
# linear problem. SSP2(3,3,2)'s coefficients are the step-1 reviewer's
# recollection, not a transcription (open in `CODE.md`), so for it this
# is part of the evidence, beside its order conditions.
@testset "On u′ = iu − u (complex, iu explicit) each tableau has its order, ±0.1" begin
    @test length(ORDER_TABLE) == 7
    for spec in ORDER_TABLE
        errs = order_errors(spec.make(), order_rotate!, 1.0 + 0.0im, order_exact_rotate)
        p = finest_order(errs)
        @test abs(p - spec.order) < 0.1
        # The recorded number, so that CODE.md's table stays true.
        @test abs(p - spec.rotate) < 0.005
    end
end

# A mistimed explicit stage, such as #4620's last stage at `tⁿ + Δt`,
# drops SSP3(3,3,2) and SSP3(4,3,3) to order 1 here (measured in step 2
# by that mutation), and is invisible to a problem autonomous in `t`.
@testset "On u′ = −u + cos t (explicit part of t alone) each tableau has its order, ±0.1" begin
    for spec in ORDER_TABLE
        errs = order_errors(spec.make(), order_forcing!, 1.0, order_exact_forced)
        p = finest_order(errs)
        @test abs(p - spec.order) < 0.1
        @test abs(p - spec.forced) < 0.005
    end
end

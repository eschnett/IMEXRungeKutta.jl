using IMEXRungeKutta: IMEXProblem
using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP2332, IMEXSSP3332, IMEXSSP3433
using IMEXRungeKutta: ARS222, ARS443

# The stiff limit on the Kaps problem ("Testing", Stiff limit, in
# `CODE.md`), `kaps_exp!` and `kaps_imp!` in `problems.jl`:
#
#     y₁′ = −(2 + 1/ε) y₁ + y₂²/ε,    y₂′ = y₁ − y₂ − y₂²,
#
# with `y(0) = (1, 1)` and the exact solution `y₁ = e^{−2t}`, `y₂ = e^{−t}`
# for every `ε`. The split is that of `CODE.md`: the implicit part is
# `g = ((y₂² − y₁)/ε, 0)`, whose stage solve is `U₂ = u★₂` and then `U₁` in
# closed form.
#
# The error is the largest over the steps to `t = 1` and over both
# components, and the order is the least-squares slope over
# `Δt = 1/40, 1/80, 1/160` (`fitted_order`). The table is in `CODE.md`,
# "The stiff limit" (measured in step 3).

const KAPS_DTS = (1 / 40, 1 / 80, 1 / 160)
const KAPS_EPS = (1.0, 1e-3, 1e-6, 1e-9)

# The largest error over the steps, per component.
function kaps_errors(tab, ε, dt)
    integ = init(IMEXProblem(kaps_exp!, kaps_imp!, [1.0, 1.0], (0.0, 1.0), ε), tab; dt)
    e₁ = e₂ = 0.0
    while integ.nstep < integ.nsteps
        step!(integ)
        e₁ = max(e₁, abs(integ.u[1] - exp(-2 * integ.t)))
        e₂ = max(e₂, abs(integ.u[2] - exp(-integ.t)))
    end
    return e₁, e₂
end

function kaps_orders(tab, ε)
    errs = [kaps_errors(tab, ε, dt) for dt in KAPS_DTS]
    return (both = fitted_order(KAPS_DTS, [max(e...) for e in errs]),
            y₂ = fitted_order(KAPS_DTS, last.(errs)))
end

# The observed orders measured in step 3, per `ε` in `KAPS_EPS`.
const KAPS_TABLE = [
    (make = IMEXSSP222, order = 2, orders = (2.025, 2.225, 2.003, 2.002)),
    (make = IMEXSSP2322, order = 2, orders = (2.023, 2.149, 2.004, 2.004)),
    (make = IMEXSSP2332, order = 2, orders = (2.016, 2.117, 1.998, 1.998)),
    (make = IMEXSSP3332, order = 2, orders = (3.025, 3.386, 2.997, 2.996)),
    (make = IMEXSSP3433, order = 3, orders = (3.026, 2.511, 1.989, 1.988)),
    (make = ARS222, order = 2, orders = (2.020, 2.003, 2.010, 2.010)),
    (make = ARS443, order = 3, orders = (3.019, 1.493, 3.017, 3.012)),
]

# An order reduction that changes, from a change to the stage arithmetic,
# the stage solve's contract or a tableau, would otherwise go unnoticed:
# the order tests are not stiff. The numbers are regression values, not
# theory. At ε = 1e−3, Δt/ε runs from 25 to 6, between the regimes, and
# the slope there is not an asymptotic order.
@testset "On the Kaps problem, the observed order per ε is the recorded one, ±0.15" begin
    @test length(KAPS_TABLE) == 7
    for spec in KAPS_TABLE, (i, ε) in enumerate(KAPS_EPS)
        @test abs(kaps_orders(spec.make(), ε).both - spec.orders[i]) < 0.15
    end
end

# SSP3(4,3,3), the production scheme, is not stiffly accurate. The claim
# made to TreeGRRMHD is that its order drops from 3 to 2 in the stiff
# limit, in the stiff component, while the non-stiff component keeps
# order 3. A tableau change that restored or worsened either would change
# what TreeGRRMHD relies on.
@testset "SSP3(4,3,3) drops to order 2 in the stiff limit, in y₁ only" begin
    for ε in (1e-6, 1e-9)
        o = kaps_orders(IMEXSSP3433(), ε)
        @test abs(o.both - 2) < 0.05
        @test abs(o.y₂ - 3) < 0.05
    end
    # Not stiff, it is third order.
    @test abs(kaps_orders(IMEXSSP3433(), 1.0).both - 3) < 0.05
    # ARS(4,4,3), stiffly accurate in both parts, keeps order 3.
    @test abs(kaps_orders(ARS443(), 1e-9).both - 3) < 0.05
end

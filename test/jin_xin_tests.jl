using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP2332, IMEXSSP3332, IMEXSSP3433
using IMEXRungeKutta: ARS222, ARS443

# The Jin–Xin relaxation of 2D Burgers' equation, `examples/jin_xin_2d.jl`,
# included as it is, so the example cannot drift from what is tested
# ("Testing", A PDE, in `CODE.md`). A 3 × 20 × 20 `Array` state, the linear
# hyperbolic part explicit and upwind, the local relaxation implicit with a
# closed-form stage solve; 25 steps of Δt = 0.01 to t = 1/4, a = 2.
include(joinpath(pkgdir(IMEXRungeKutta), "examples", "jin_xin_2d.jl"))

# Per tableau: the stage-limiter calls per step (explicit-used stages that
# are not trivial), and `max |v − u²/2|` at ε = 1e−8 after 25 steps, as
# measured when the example was added (2026-09-24). The stiff-limit
# residual is `O(ε)` where both parts are stiffly accurate (ARS),
# `O(Δt²)` where only the implicit part is (SSP2(3,2,2), SSP2(3,3,2)), and
# `O(Δt)` otherwise ("Where a step ends in the stiff limit" in `CODE.md`).
const JINXIN_TABLE = [
    (make = IMEXSSP222, limits = 2, residual = 0.03254),
    (make = IMEXSSP2322, limits = 2, residual = 0.001679),
    (make = IMEXSSP2332, limits = 3, residual = 0.0008404),
    (make = IMEXSSP3332, limits = 3, residual = 0.03153),
    (make = IMEXSSP3433, limits = 3, residual = 0.01240),
    (make = ARS222, limits = 1, residual = 4.147e-8),
    (make = ARS443, limits = 3, residual = 4.146e-8),
]

# The integrator must accept a state that is not a vector, and the stage
# solve leaves `u` as it entered: so `sum(u)` changes by round-off only. A
# stage solve that saw an uninitialized `U`, an update that weighted an
# increment of `u`, or a plan that assumed a vector would break it.
@testset "Jin–Xin on a 3×20×20 Array conserves u to round-off: $(spec.make().name)" for
    spec in JINXIN_TABLE
    for ε in (1e-2, 1e-8)
        r = jinxin_run(spec.make(), ε)
        @test size(r.U) == (3, 20, 20)
        @test r.steps == 25
        @test r.t == 1 / 4
        @test abs(r.mass) < 1e-12
        @test 0.5 < r.umin < r.umax < 1.5      # inside the initial range
    end
end

# The stage limiter runs where `f_exp!` reads and nowhere else ("One
# step" in `CODE.md`): a call at a stage the explicit part does not read,
# or on the trivial first stage of an ARS scheme, would change the count.
@testset "Jin–Xin makes the stage-limiter calls of the tableau: $(spec.make().name)" for
    spec in JINXIN_TABLE
    r = jinxin_run(spec.make(), 1e-2)
    @test r.nlimit == spec.limits * r.steps
end

# Where the stiff limit leaves `v` and `w` depends on the tableau alone:
# `O(ε)` for ARS, `O(Δt²)` or `O(Δt)` for the others. A plan that took the
# increment after the limiter, or weighted the update differently, would
# move these numbers.
@testset "Jin–Xin at ε = 1e−8 ends where the tableau says: $(spec.make().name)" for
    spec in JINXIN_TABLE
    r = jinxin_run(spec.make(), 1e-8)
    @test isapprox(r.residual, spec.residual; rtol = 0.02)
end

# In the stiff limit every tableau approximates the same relaxed Burgers
# solution: the third-order SSP3(4,3,3) is within 4e−5 of ARS(4,4,3), and
# the second-order ones within 1.3e−3 (measured 3.7e−5 and 1.27e−3).
@testset "Jin–Xin at ε = 1e−8: every tableau approximates the same Burgers solution" begin
    ref = jinxin_run(ARS443(), 1e-8).U[1, :, :]
    distance(tab) = maximum(abs.(jinxin_run(tab, 1e-8).U[1, :, :] .- ref))
    @test distance(IMEXSSP3433()) < 1e-4
    for spec in JINXIN_TABLE
        @test distance(spec.make()) < 2e-3
    end
end

# The owner path indexes the state linearly: a 3-dimensional `Array` must
# give the same bits as the broadcast, with `:even` and with a partition of
# several ranges per thread that cut across components and cells.
@testset "Jin–Xin by owner gives the broadcast's bits: $(spec.make().name)" for
    spec in JINXIN_TABLE
    n = 3 * 20 * 20
    nt = Threads.nthreads()
    pieces = [((k - 1) * n ÷ (2nt) + 1):(k * n ÷ (2nt)) for k in 1:(2nt)]
    split = [[pieces[c], pieces[c + nt]] for c in 1:nt]
    b = jinxin_run(spec.make(), 1e-8).U
    @test jinxin_run(spec.make(), 1e-8; partition = :even).U == b
    @test jinxin_run(spec.make(), 1e-8; partition = split).U == b
end

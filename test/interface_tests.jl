using CommonSolve: CommonSolve
using IMEXRungeKutta: IMEXRungeKutta, IMEXProblem, IMEXTableau
using IMEXRungeKutta: IMEXSSP222, IMEXSSP3433, ARS222, ARS443

# "The interface" and "Time and the step count" in `CODE.md`.

f_decay!(du, u, p, t) = (du .= cos(t); nothing)
solve_decay_imp!(U, u★, γΔt, p, t) = (U .= u★ ./ (1 + γΔt); nothing)
decay_problem(u0, tspan, p = nothing) = IMEXProblem(f_decay!, solve_decay_imp!, u0, tspan, p)

# An extra step from round-off would put the end of a chunk one step past
# where the caller's CFL bookkeeping expects it; a step count not rounded
# up would make `Δt` exceed the CFL limit `dt`.
@testset "nsteps = ⌈(t1 − t0)/dt⌉ with no extra step from round-off, Δt ≤ dt" begin
    nsteps(t0, t1, dt) = init(decay_problem([1.0], (t0, t1)), IMEXSSP222(); dt).nsteps
    Δt(t0, t1, dt) = init(decay_problem([1.0], (t0, t1)), IMEXSSP222(); dt).dt
    @test nsteps(0.0, 1.0, 1 / 10) == 10
    @test Δt(0.0, 1.0, 1 / 10) == 0.1
    # Quotients just above an integer, where a bare ceiling adds a step:
    # 0.07/0.01 = 7.000000000000001, 2.1/0.3 = 7.000000000000001, and
    # (3 · 0.1)/0.1 = 3.0000000000000004.
    @test 0.07 / 0.01 > 7 && 2.1 / 0.3 > 7 && (3 * 0.1) / 0.1 > 3
    @test nsteps(0.0, 0.07, 0.01) == 7
    @test nsteps(0.0, 2.1, 0.3) == 7
    @test nsteps(0.0, 3 * 0.1, 0.1) == 3
    @test nsteps(0.0, 1.1, 0.1) == 11
    # Not a whole number of steps: rounded up, and then Δt < dt.
    @test nsteps(0.0, 1.0, 0.3) == 4
    @test Δt(0.0, 1.0, 0.3) == 0.25
    @test nsteps(0.0, 1.0, 2.0) == 1
    @test Δt(0.0, 1.0, 2.0) == 1.0
    @test nsteps(0.0, 1.0, 0.35) == 3
    @test nsteps(0.0, 1.0, 1 / 3 + 1e-9) == 3
    @test nsteps(0.0, 1.0, 1 / 3 - 1e-9) == 4
    # Chunks of a long run, `(kT, (k + 1)T)` with `dt = T/m`: `t1 − t0`
    # inherits the rounding of both ends, which a tolerance in the ulps of
    # the quotient alone does not cover. Each chunk is exactly `m` steps,
    # and `Δt` exceeds `dt` by at most that tolerance.
    worst = 0.0
    for T in (0.1, 0.3, 1 / 7, 2.5e-3), k in (0, 1, 7, 99, 1000, 123456), m in 1:12
        t0, t1, dt = k * T, (k + 1) * T, T / m
        @test nsteps(t0, t1, dt) == m
        worst = max(worst, Δt(t0, t1, dt) / dt - 1)
        @test Δt(t0, t1, dt) ≤ dt + 4 * (eps(dt) + eps(t0) + eps(t1))
    end
    # Measured in step 2: at most 1.09e-11, at k = 123456.
    @test worst < 1e-10
end

# An accumulated `t` drifts by an ulp per step, and a last `t` that misses
# `t1` breaks a chunked driver that compares times; a step past `t1` would
# integrate beyond the interval ("No accumulated time" in `CODE.md`).
@testset "tⁿ = t0 + nΔt afresh, the last t is t1 exactly, and one more step! throws" begin
    # On (0.2, 0.9), `t0 + 7Δt` is 0.8999999999999999, not 0.9.
    integ = init(decay_problem([1.0], (0.2, 0.9)), IMEXSSP3433(); dt = 0.1)
    @test integ.nsteps == 7
    @test 0.2 + 7 * integ.dt != 0.9
    @test integ.t === 0.2 && integ.nstep == 0
    for n in 1:7
        step!(integ)
        @test integ.nstep == n
        @test integ.t === (n == 7 ? 0.9 : 0.2 + n * integ.dt)
    end
    @test_throws ArgumentError step!(integ)
    @test integ.nstep == 7 && integ.t === 0.9
    err = try
        step!(integ)
    catch e
        e
    end
    @test occursin("taken all its 7 steps", err.msg)
    # `solve!` on a finished integrator does nothing and returns it.
    @test solve!(integ) === integ
    @test integ.nstep == 7
end

# The time type and `T` tied together would force a Float32 state onto
# Float32 time, whose ulp at t = 1000 is 6e-5 ("Two types" in `CODE.md`).
@testset "The time type and T are separate: Float32 state, Float64 time, and back" begin
    integ = mock_integrator(IMEXSSP3433(), Float32[1, 2]; tspan = (0.0, 1.0), dt = 0.1)
    @test typeof(integ.t) === Float64 && typeof(integ.dt) === Float64
    @test eltype(integ.u) === Float32
    solve!(integ)
    log = integ.p
    @test eltype(log.γΔt) === Float32 && eltype(log.t) === Float64
    @test integ.t === 1.0
    # A Float32 run agrees with the Float64 one to Float32 accuracy.
    ref = solve(decay_problem([1.0, 2.0], (0.0, 1.0)), IMEXSSP3433(); dt = 0.1)
    @test maximum(abs.(integ.u .- ref.u)) < 1e-5
    # Float64 state, Float32 time.
    integ = init(decay_problem([1.0], (0.0f0, 1.0f0)), IMEXSSP222(); dt = 0.1f0)
    @test typeof(integ.t) === Float32 && eltype(integ.u) === Float64
    solve!(integ)
    @test integ.t === 1.0f0
    # A BigFloat state, at the global precision, with Float64 time; the
    # coefficients are 256-bit ("What coefficients holds" in `CODE.md`).
    big = solve(decay_problem(BigFloat[1, 2], (0.0, 1.0)), IMEXSSP3433(); dt = 0.1)
    @test eltype(big.u) === BigFloat && typeof(big.t) === Float64
    @test maximum(abs.(big.u .- ref.u)) < 1e-14
    # An integer tspan and a rational dt give float time.
    integ = init(decay_problem([1.0], (0, 1)), IMEXSSP222(); dt = 1 // 10)
    @test typeof(integ.t) === Float64 && integ.nsteps == 10
end

# `T = real(eltype(u0))`; a complex state must not make the coefficients
# complex or the time complex ("Two types" in `CODE.md`). The split
# linear ODE `u′ = iu − u`, with `iu` explicit, is step 3's order problem.
f_rotate!(du, u, p, t) = (du .= im .* u; nothing)
@testset "A complex state works: u′ = iu − u to the tableau's accuracy" begin
    for (tab, p) in ((IMEXSSP222(), 2), (IMEXSSP3433(), 3), (ARS443(), 3))
        errs = map((0.02, 0.01)) do dt
            prob = IMEXProblem(f_rotate!, solve_decay_imp!, ComplexF64[1, 1im], (0.0, 1.0))
            integ = solve(prob, tab; dt)
            @test typeof(integ.dt) === Float64
            return maximum(abs.(integ.u .- ComplexF64[1, 1im] .* exp((im - 1) * 1.0)))
        end
        @test abs(log2(errs[1] / errs[2]) - p) < 0.2
    end
end

# SciML's default: a problem without parameters passes `nothing`.
@testset "p defaults to nothing, and every callback receives it" begin
    seen = Any[]
    f!(du, u, p, t) = (push!(seen, p); du .= 0; nothing)
    g!(U, u★, γΔt, p, t) = (push!(seen, p); nothing)
    lim!(u, integ, p, t) = (push!(seen, p); nothing)
    prob = IMEXProblem(f!, g!, [1.0], (0.0, 1.0))
    @test prob.p === nothing
    integ = init(prob, IMEXSSP3433(); dt = 0.5, stage_limiter = lim!, step_limiter = lim!)
    @test integ.p === nothing
    step!(integ)
    @test length(seen) == 3 + 4 + 3 + 1
    @test all(x -> x === nothing, seen)
    prob = IMEXProblem(f!, g!, [1.0], (0.0, 1.0), :params)
    empty!(seen)
    solve(prob, IMEXSSP3433(); dt = 0.5)
    @test all(x -> x === :params, seen) && length(seen) == 2 * 7
end

# Aliasing by default would overwrite the caller's `u0`, which is often a
# field set of a larger structure; refusing to alias would cost a
# state-sized array that `alias_u0` exists to save ("The interface").
@testset "init copies u0 unless alias_u0 = true" begin
    u0 = [1.0, 2.0]
    integ = solve(decay_problem(u0, (0.0, 1.0)), ARS222(); dt = 0.1)
    @test integ.u !== u0 && u0 == [1.0, 2.0]
    integ2 = solve(decay_problem(u0, (0.0, 1.0)), ARS222(); dt = 0.1, alias_u0 = true)
    @test integ2.u === u0 && u0 == integ.u
end

# CommonSolve's contract: `solve` is `solve!(init(…))`, and returns what
# `solve!` returns, the integrator, here by a method of our own rather
# than CommonSolve's generic fallback (proposed in step 2).
@testset "solve(prob, tab; dt) returns the integrator, at t1" begin
    prob = decay_problem([1.0], (0.0, 1.0))
    integ = solve(prob, IMEXSSP3433(); dt = 0.1)
    @test integ isa IMEXRungeKutta.IMEXIntegrator
    @test integ.t === 1.0 && integ.nstep == integ.nsteps == 10
    @test integ.u == solve!(init(prob, IMEXSSP3433(); dt = 0.1)).u
    @test which(CommonSolve.solve, Tuple{IMEXProblem,IMEXTableau}).module === IMEXRungeKutta
    @test integ.tableau.name == "SSP3(4,3,3)"
    @test occursin("SSP3(4,3,3)", sprint(show, integ))
end

# A `step!` whose return type the compiler cannot infer would box and
# allocate, and hide the plan's type from the stage code ("The stage plan
# and storage" in `CODE.md`).
@testset "step! passes @inferred for every tableau" begin
    for make in NAMED_TABLEAUS
        integ = mock_integrator(make(), [1.0, 2.0])
        @test (@inferred step!(integ)) === nothing
        integ = init(decay_problem(Float32[1], (0.0, 1.0)), make(); dt = 0.1)
        @test (@inferred step!(integ)) === nothing
    end
end

# The caller changes `integ.u` in place between steps, and nothing may
# carry over ("The caller may change integ.u in place" in `CODE.md`); a
# rebinding would leave the plan on the old array, so it is refused.
f_auto!(du, u, p, t) = (du .= .-u .^ 2; nothing)
@testset "integ.u may change in place between steps, but not be rebound" begin
    prob = IMEXProblem(f_auto!, solve_decay_imp!, [1.0, 2.0], (0.0, 1.0))
    a = init(prob, IMEXSSP3433(); dt = 0.1)
    step!(a)
    a.u .= [0.5, -0.25]
    step!(a)
    b = init(IMEXProblem(f_auto!, solve_decay_imp!, [0.5, -0.25], (0.0, 1.0)),
             IMEXSSP3433(); dt = 0.1)
    step!(b)
    @test a.u == b.u
    @test_throws ErrorException (a.u = [1.0, 2.0])
end

# A malformed request accepted silently would integrate something else;
# each refusal says why.
@testset "init refuses what it cannot integrate, saying why" begin
    msg(f) = try
        f()
        ""
    catch e
        e isa ArgumentError ? e.msg : "not an ArgumentError: $e"
    end
    prob = decay_problem([1.0], (0.0, 1.0))
    # A partition for a state that is not a CPU `Array` (step 5; the other
    # partition refusals are in `owner_tests.jl`).
    @test occursin("only for a CPU Array",
                   msg(() -> init(decay_problem(view([1.0, 2.0], 1:2), (0.0, 1.0)), ARS222();
                                  dt = 0.1, partition = :even)))
    @test occursin("floating-point",
                   msg(() -> init(decay_problem([1], (0.0, 1.0)), ARS222(); dt = 0.1)))
    @test occursin("t1 > t0",
                   msg(() -> init(decay_problem([1.0], (1.0, 0.0)), ARS222(); dt = 0.1)))
    @test occursin("t1 > t0",
                   msg(() -> init(decay_problem([1.0], (1.0, 1.0)), ARS222(); dt = 0.1)))
    @test occursin("positive", msg(() -> init(prob, ARS222(); dt = 0.0)))
    @test occursin("positive", msg(() -> init(prob, ARS222(); dt = -0.1)))
    @test occursin("positive", msg(() -> init(prob, ARS222(); dt = Inf)))
    @test occursin("finite",
                   msg(() -> init(decay_problem([1.0], (0.0, Inf)), ARS222(); dt = 0.1)))
    @test occursin("(t0, t1)", msg(() -> IMEXProblem(f_decay!, solve_decay_imp!, [1.0],
                                                     (0.0, 1.0, 2.0))))
end

# An export without a docstring, or one that does not point at the design
# document, leaves a caller guessing at a contract that `CODE.md` states
# ("Documentation" in `CODE.md`). A method docstring on CommonSolve's
# functions is stored under IMEXRungeKutta's own docs table.
@testset "Every export has a docstring that points at CODE.md" begin
    meta = Base.Docs.meta(IMEXRungeKutta)
    for name in names(IMEXRungeKutta)
        binding = Base.Docs.Binding(IMEXRungeKutta, name)
        @test haskey(meta, binding)
        haskey(meta, binding) || continue
        texts = [join(doc.text) for doc in values(meta[binding].docs)]
        @test any(t -> occursin("CODE.md", t), texts)
    end
end

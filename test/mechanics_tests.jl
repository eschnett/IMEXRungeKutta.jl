using IMEXRungeKutta: IMEXRungeKutta, IMEXTableau, IMEXProblem
using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP2332, IMEXSSP3332, IMEXSSP3433
using IMEXRungeKutta: ARS222, ARS443
using IMEXRungeKutta: coefficients, scratch_count, plan_calls

# The mechanics of "One step" in `CODE.md`, against the mocks of
# `test/mocks.jl`, for all seven named tableaus, and for three tableaus of
# a caller's own that exercise the plan's corner cases.

# The calls one step at `tⁿ = tn` makes, in order, with their times,
# rederived here from "One step" and the exact tableau, independently of
# the plan: per stage, the solve at `c`, then, if the stage is
# explicit-used, the stage limiter (unless the stage is trivial) and
# `f_exp!` at `c̃`; then the step limiter at `tⁿ⁺¹`.
function expected_calls(tab, tn, Δt, tnext)
    co = coefficients(Float64, Float64, tab)
    s = length(tab.b)
    seq = Tuple{Symbol,Float64}[]
    for k in 1:s
        solves = !iszero(tab.A[k, k])
        explicit_used = any(!iszero, tab.Ã[:, k]) || !iszero(tab.b̃[k])
        empty_row = all(iszero, tab.Ã[k, 1:(k - 1)]) && all(iszero, tab.A[k, 1:(k - 1)])
        solves && push!(seq, (:solve_imp, tn + co.c[k] * Δt))
        if explicit_used
            (!solves && empty_row) || push!(seq, (:stage_limiter, tn + co.c̃[k] * Δt))
            push!(seq, (:f_exp, tn + co.c̃[k] * Δt))
        end
    end
    push!(seq, (:step_limiter, tnext))
    return seq
end

logged_calls(log) = [(log.kind[i], log.t[i]) for i in 1:(log.n)]

# Every array the plan reads or writes.
function plan_arrays(plan)
    arrays = Any[]
    for st in plan.stages
        append!(arrays, last.(st.terms))
        for a in (st.u★, st.U, st.d, st.k̃)
            a === nothing || push!(arrays, a)
        end
    end
    append!(arrays, last.(plan.update))
    return arrays
end

# Tableaus of a caller's own that reach the plan's corner cases.
# - "extra": stage 2 solves, is not implicit-used and has a nonempty row,
#   so its `u★` needs the one extra array.
# - "empty row": stage 1 solves and is not implicit-used, but its row is
#   empty, so its `u★` is `integ.u` and no extra array is needed.
# - "dead": stage 2 makes no solve and nothing reads it, so it does
#   nothing; stage 3 solves but neither part reads it, so it still makes
#   its one call ("One step": "If a_kk ≠ 0: solve_imp!") into the extra
#   array, and nothing else.
const CORNER_TABLEAUS = [
    (tab = IMEXTableau("extra", [0 0; 1 0], [1 // 2, 1 // 2], [1 0; 1//2 1//2], [1, 0]),
     f_exp = 2, solve_imp = 2, stage_limiter = 2, scratch = 5),
    (tab = IMEXTableau("empty row", [0 0; 1 0], [0, 1], [1 0; 0 1], [0, 1]),
     f_exp = 2, solve_imp = 2, stage_limiter = 2, scratch = 4),
    (tab = IMEXTableau("dead", [0 0 0; 1 0 0; 1 0 0], [1, 0, 0], [1 0 0; 0 0 0; 0 0 1],
                       [1, 0, 0]),
     f_exp = 1, solve_imp = 2, stage_limiter = 1, scratch = 4),
]

# A per-tableau test that silently skipped a tableau would leave its plan
# untested; the list must be the ten exported names.
@testset "The per-tableau tests cover all ten named tableaus" begin
    exported = [getfield(IMEXRungeKutta, n) for n in names(IMEXRungeKutta)
                if occursin(r"^(IMEXSSP|ARS|Euler$|RK4$|SSPRK)", String(n))]
    @test all(f -> f() isa IMEXTableau, exported)
    @test length(NAMED_TABLEAUS) == 10
    @test Set(NAMED_TABLEAUS) == Set(exported)
end

# An explicit evaluation where no column of `Ã` and no `b̃_k` reads it is
# the expensive dead work this package exists to skip; a missing one, or a
# second stage solve per stage, breaks the stage contract ("One step",
# "Cost" in `CODE.md`).
@testset "Each step makes the hand-counted calls, and the plan counts the same" begin
    for spec in [ALL_CALL_COUNTS; CORNER_TABLEAUS]
        tab = haskey(spec, :make) ? spec.make() : spec.tab
        integ = mock_integrator(tab, [1.0, 2.0])
        log = integ.p
        for n in 1:3
            reset!(log)
            step!(integ)
            @test length(calls(log, :f_exp)) == spec.f_exp
            @test length(calls(log, :solve_imp)) == spec.solve_imp
            @test length(calls(log, :stage_limiter)) == spec.stage_limiter
            @test length(calls(log, :step_limiter)) == 1
        end
        @test plan_calls(integ.plan) ==
              (f_exp = spec.f_exp, solve_imp = spec.solve_imp,
               stage_limiter = spec.stage_limiter)
    end
    # "Cost" in `CODE.md`: three explicit evaluations per SSP3(4,3,3) step,
    # where OrdinaryDiffEqSDIRK makes five.
    integ = mock_integrator(IMEXSSP3433(), [1.0])
    step!(integ)
    @test length(calls(integ.p, :f_exp)) == 3
    @test length(calls(integ.p, :solve_imp)) == 4
end

# A plan that allocated an array for a structural zero would waste a
# state-sized array; one that read an array it had not allocated, or two
# stages sharing one they both need, would give wrong results ("The stage
# plan and storage" in `CODE.md`).
@testset "The scratch is scratch_count distinct arrays, and all the plan reads" begin
    for spec in [ALL_CALL_COUNTS; CORNER_TABLEAUS]
        tab = haskey(spec, :make) ? spec.make() : spec.tab
        integ = mock_integrator(tab, [1.0, 2.0])
        scratch = integ.plan.scratch
        @test length(scratch) == scratch_count(tab) == spec.scratch
        @test allunique(objectid.(scratch))
        @test all(a -> a !== integ.u, scratch)
        arrays = plan_arrays(integ.plan)
        @test all(a -> a === integ.u || any(b -> b === a, scratch), arrays)
        @test all(b -> any(a -> a === b, arrays), scratch)
    end
    # The extra array is the `u★` of "extra"'s stage 2, and "empty row"'s
    # stage 1 forms no `u★` at all: it is `integ.u`.
    integ = mock_integrator(CORNER_TABLEAUS[1].tab, [1.0])
    st = integ.plan.stages[2]
    @test st.u★ !== integ.u && st.d === nothing && st.u★ !== st.U
    integ = mock_integrator(CORNER_TABLEAUS[2].tab, [1.0])
    @test integ.plan.stages[1].u★ === integ.u
end

# Mixing up the two abscissae is invisible on a problem autonomous in `t`
# (SciML/OrdinaryDiffEq.jl#4620); a limiter or a solve out of order would
# break "One step". The log is compared with the sequence rederived from
# the exact tableau, call by call, time by time.
@testset "The calls come in the order of One step, each at its own abscissa" begin
    for spec in [ALL_CALL_COUNTS; CORNER_TABLEAUS]
        tab = haskey(spec, :make) ? spec.make() : spec.tab
        integ = mock_integrator(tab, [1.0, 2.0]; tspan = (0.25, 1.25), dt = 0.1)
        log = integ.p
        for n in 0:(integ.nsteps - 1)
            reset!(log)
            tn = integ.t
            @test tn == 0.25 + n * integ.dt
            step!(integ)
            @test logged_calls(log) == expected_calls(tab, tn, integ.dt, integ.t)
        end
        @test integ.t === 1.25
    end
    # An anchor independent of the helper: SSP3(4,3,3)'s first stage solves
    # at tⁿ + αΔt, where its explicit abscissa is 0, and SSP2(2,2,2)'s stage
    # 1 solves at tⁿ + γΔt but evaluates `f` at tⁿ.
    integ = mock_integrator(IMEXSSP3433(), [1.0]; tspan = (0.0, 1.0), dt = 0.5)
    step!(integ)
    log = integ.p
    @test log.kind[1] === :solve_imp && log.t[1] == 0.24169426078820838 * 0.5
    integ = mock_integrator(IMEXSSP222(), [1.0]; tspan = (0.0, 1.0), dt = 0.5)
    step!(integ)
    log = integ.p
    γ = 1 - 1 / sqrt(2)
    @test log.kind[1:3] == [:solve_imp, :stage_limiter, :f_exp]
    @test log.t[1] ≈ γ * 0.5 && log.t[3] == 0
end

# A `U` that did not start as `u★`, or that aliased it, would make every
# caller's solver copy the untouched components itself, or clobber `u★`
# as it wrote `U`; a wrong `γΔt` would solve another stage equation ("The
# callback contracts" in `CODE.md`).
@testset "solve_imp! gets U = u★, distinct, u★ kept, and γΔt = a_kk Δt in T" begin
    for spec in [ALL_CALL_COUNTS; CORNER_TABLEAUS]
        tab = haskey(spec, :make) ? spec.make() : spec.tab
        integ = mock_integrator(tab, [1.0, -2.0, 3.0])
        log = integ.p
        step!(integ)
        reset!(log)
        step!(integ)
        γ = coefficients(Float64, Float64, tab).γ
        solving = findall(!iszero, [tab.A[k, k] for k in 1:length(tab.b)])
        for (i, k) in zip(calls(log, :solve_imp), solving)
            @test log.U_was_u★[i]
            @test log.distinct[i]
            @test log.u★_kept[i]
            @test log.γΔt[i] == γ[k] * integ.dt
            @test log.arr[i] !== integ.u
            # `u★ = uⁿ` exactly where the row is empty, and it is then
            # `integ.u` itself; otherwise it is formed in scratch.
            @test (log.u★[i] === integ.u) == IMEXRungeKutta.row_empty(tab, k)
        end
    end
end

# A limiter on `integ.u` would change `uⁿ` mid-step; one called on a
# different array from the one `f_exp!` reads, or not just before it,
# would limit nothing that is read ("The stage limiter acts only where
# f_exp! reads" in `CODE.md`).
@testset "The stage limiter runs just before each f_exp!, on its array, never on u" begin
    for spec in [ALL_CALL_COUNTS; CORNER_TABLEAUS]
        tab = haskey(spec, :make) ? spec.make() : spec.tab
        integ = mock_integrator(tab, [1.0, 2.0])
        log = integ.p
        step!(integ)
        reset!(log)
        tn = integ.t
        step!(integ)
        for i in calls(log, :f_exp)
            if log.arr[i] === integ.u
                # Only at a trivial stage, and then with no limiter call.
                @test i == 1 || log.kind[i - 1] !== :stage_limiter
            else
                @test log.kind[i - 1] === :stage_limiter
                @test log.arr[i - 1] === log.arr[i]
                @test log.t[i - 1] == log.t[i]
            end
        end
        for i in calls(log, :stage_limiter)
            @test log.kind[i + 1] === :f_exp
            @test log.arr[i] !== integ.u
            @test log.integ_t[i] == tn
            @test log.integ_nstep[i] == 1
        end
    end
end

# A copy of `uⁿ` at a trivial stage costs a state pass for nothing, and a
# limiter call there would limit a state that has already been through the
# step limiter ("A trivial first stage is uⁿ" in `CODE.md`).
@testset "At a trivial first stage, f_exp! receives integ.u itself" begin
    for spec in ALL_CALL_COUNTS
        tab = spec.make()
        integ = mock_integrator(tab, [1.0, 2.0])
        step!(integ)
        log = integ.p
        first_f = first(calls(log, :f_exp))
        trivial = tab.name in ("ARS(2,2,2)", "ARS(4,4,3)", "Euler", "RK4", "SSPRK(3,3)")
        @test (log.arr[first_f] === integ.u) == trivial
        @test count(i -> log.arr[i] === integ.u, calls(log, :f_exp)) == (trivial ? 1 : 0)
    end
end

# At a trivial first stage `f_exp!` reads `uⁿ` as the previous step's step
# limiter left it, and no stage limiter runs there; so a purely explicit
# tableau limits every right-hand-side input only with both limiters
# ("Explicit tableaus" in `CODE.md`). The step limiter writes a marker into
# `uⁿ⁺¹`, the stage limiter another into each stage value, and `f` is zero,
# so the first evaluation of step 2 sees the step limiter's marker exactly
# where the first stage is trivial, and the stage limiter's elsewhere.
function marker_f!(du, u, seen, t)
    push!(seen, u[1])
    du .= 0
    return nothing
end
marker_step_limiter!(u, integ, p, t) = (u[1] = 1000; nothing)
marker_stage_limiter!(u, integ, p, t) = (u[1] = -1; nothing)
@testset "A trivial first stage reads uⁿ as the step limiter left it, unlimited by the stage limiter" begin
    for spec in ALL_CALL_COUNTS
        tab = spec.make()
        seen = Float64[]
        prob = IMEXProblem(marker_f!, (U, u★, γΔt, p, t) -> nothing, [1.0, 2.0], (0.0, 1.0),
                           seen)
        integ = init(prob, tab; dt = 0.1, stage_limiter = marker_stage_limiter!,
                     step_limiter = marker_step_limiter!)
        step!(integ)
        empty!(seen)
        step!(integ)
        trivial = tab.name in ("ARS(2,2,2)", "ARS(4,4,3)", "Euler", "RK4", "SSPRK(3,3)")
        @test first(seen) == (trivial ? 1000 : -1)
        @test count(==(1000), seen) == (trivial ? 1 : 0)
    end
end

# A step limiter on a scratch array, at the wrong time, or after
# `integ.t` has advanced would break OrdinaryDiffEq's contract, which
# TreeHydro's limiters are written against ("The callback contracts" in
# `CODE.md`).
@testset "The step limiter runs once per step, on integ.u, at tⁿ⁺¹, before t advances" begin
    for spec in ALL_CALL_COUNTS
        integ = mock_integrator(spec.make(), [1.0, 2.0]; tspan = (0.0, 0.3), dt = 0.1)
        log = integ.p
        for n in 0:2
            reset!(log)
            tn = integ.t
            step!(integ)
            (i,) = calls(log, :step_limiter)
            @test i == log.n
            @test log.arr[i] === integ.u
            @test log.t[i] == (n == 2 ? 0.3 : (n + 1) * integ.dt)
            @test log.t[i] == integ.t
            @test log.integ_t[i] == tn
            @test log.integ_nstep[i] == n
        end
    end
end

# A structural zero that is read multiplies whatever the scratch holds,
# and `0·NaN = NaN`; scratch that carried a value from one step to the
# next would break "the caller may change integ.u between steps".
@testset "Scratch filled with NaN before every step changes no bit of the result" begin
    for spec in [ALL_CALL_COUNTS; CORNER_TABLEAUS]
        tab = haskey(spec, :make) ? spec.make() : spec.tab
        a = mock_integrator(tab, [1.0, -0.5, 2.0])
        b = mock_integrator(tab, [1.0, -0.5, 2.0])
        for n in 1:4
            foreach(x -> fill!(x, NaN), b.plan.scratch)
            step!(a)
            step!(b)
        end
        @test !any(isnan, b.u)
        @test b.u == a.u
    end
end

# A coefficient that underflows to zero in `T` must neither drop its term
# from the plan (the pattern is the exact tableau's) nor, worse, leave an
# unallocated or unwritten array to be read (`coefficients`' docstring).
@testset "A coefficient that underflows in T keeps its term and its array" begin
    tab = IMEXTableau("underflow", [0 0; 1e-60 0], [0, 1], [1//2 0; 0 1//2],
                      [1 // 2, 1 // 2])
    @test IMEXRungeKutta.explicit_used(tab) == [true, true]
    integ = mock_integrator(tab, Float32[1, 2])
    (term,) = filter(t -> t[2] === integ.plan.stages[1].k̃, integ.plan.stages[2].terms)
    @test term[1] === 0.0f0
    @test length(integ.plan.scratch) == scratch_count(tab) == 5
    @test plan_calls(integ.plan).f_exp == 2
    foreach(x -> fill!(x, NaN32), integ.plan.scratch)
    step!(integ)
    @test !any(isnan, integ.u)
end

# A step that wrote `integ.u` before its last stage could not be retried
# from `uⁿ` after the stage solver fails ("Failures and exceptions" in
# `CODE.md`).
@testset "An exception from a callback leaves integ.u and integ.t unchanged" begin
    for spec in ALL_CALL_COUNTS
        tab = spec.make()
        integ = mock_integrator(tab, [1.0, 2.0])
        log = integ.p
        step!(integ)
        reset!(log)
        step!(integ)
        # The calls of one step, less the step limiter, after which
        # `integ.u` is undefined.
        ncalls = log.n - 1
        for at in 1:ncalls
            kind = log.kind[at]
            integ = mock_integrator(tab, [1.0, 2.0])
            step!(integ)
            u = copy(integ.u)
            reset!(integ.p)
            integ.p.throw_at = at
            @test_throws MockFailure step!(integ)
            @test integ.p.kind[at] === kind
            @test integ.u == u
            @test integ.t == integ.dt
            @test integ.nstep == 1
        end
        # After the step limiter throws, `integ.t` and `nstep` have not
        # advanced either.
        integ = mock_integrator(tab, [1.0, 2.0])
        integ.p.throw_at = ncalls + 1
        @test_throws MockFailure step!(integ)
        @test integ.p.kind[ncalls + 1] === :step_limiter
        @test integ.t == 0 && integ.nstep == 0
    end
end

# The explicit part run by the plan: stage values, abscissae and weights of
# `(Ã, b̃, c̃)`. With `g ≡ 0` every increment is exactly zero, and a plan
# that mis-weighted a tendency, dropped one, or evaluated `f` at the wrong
# time would differ from the explicit method by more than round-off.
rhs_nonlinear(u, t) = -u .^ 2 ./ 2 .+ cos(t)
function f_nonlinear!(du, u, p, t)
    du .= .-u .^ 2 ./ 2 .+ cos(t)
    return nothing
end
solve_nothing!(U, u★, γΔt, p, t) = nothing
function reference_erk(tab, u0, t0, Δt, nsteps)
    return setprecision(BigFloat, 256) do
        s = length(tab.b)
        u = BigFloat.(u0)
        Δ = BigFloat(Δt)
        for n in 0:(nsteps - 1)
            tn = BigFloat(t0) + n * Δ
            k = Vector{Vector{BigFloat}}(undef, s)
            for i in 1:s
                U = u + Δ * sum((BigFloat(tab.Ã[i, j]) * k[j] for j in 1:(i - 1));
                                init = zero(u))
                k[i] = rhs_nonlinear(U, tn + BigFloat(tab.c̃[i]) * Δ)
            end
            u = u + Δ * sum(BigFloat(tab.b̃[j]) * k[j] for j in 1:s)
        end
        return u
    end
end
@testset "With g ≡ 0 the result is the explicit RK method, to round-off" begin
    for spec in ALL_CALL_COUNTS
        tab = spec.make()
        u0 = [1.0, -0.5, 2.0]
        prob = IMEXProblem(f_nonlinear!, solve_nothing!, u0, (0.5, 1.5))
        integ = solve(prob, tab; dt = 0.1)
        ref = reference_erk(tab, u0, 0.5, integ.dt, 10)
        # Measured in step 2: at most 5.0e-16.
        @test maximum(abs.(integ.u .- ref)) < 5e-15
    end
end

# A component the stage solver leaves as it entered must evolve by the
# explicit part alone, to the last bit, or a conservative explicit scheme
# stops being conservative ("Untouched components stay untouched" in
# `CODE.md`). The solver here relaxes component 1 only; the explicit part
# is componentwise, so components 2–4 must match the run with `g ≡ 0`.
function solve_first!(U, u★, γΔt, p, t)
    λ = γΔt / p
    U[1] = (u★[1] + λ * 3) / (1 + λ)
    return nothing
end
@testset "Untouched components match the explicit-only run bitwise" begin
    for spec in CALL_COUNTS
        tab = spec.make()
        u0 = [1.0, -0.5, 2.0, 0.25]
        a = solve(IMEXProblem(f_nonlinear!, solve_first!, u0, (0.0, 1.0), 1e-3), tab;
                  dt = 0.1)
        b = solve(IMEXProblem(f_nonlinear!, solve_nothing!, u0, (0.0, 1.0)), tab; dt = 0.1)
        @test a.u[2:4] == b.u[2:4]
        @test abs(a.u[1] - b.u[1]) > 0.1
    end
end

# A limiter correction folded into the increment would be re-weighted by
# `a_jk/a_kk` and `b_k/a_kk` ("The tendency is taken before the limiter"
# in `CODE.md`). With an `f` that does not read `u`, even a limiter that
# overwrites the stage value must change no bit of the result; with one
# that does, it must change it.
function limiter_overwrite!(u, integ, p, t)
    u .= 1000
    return nothing
end
f_time!(du, u, p, t) = (du .= cos(t); nothing)
solve_decay!(U, u★, γΔt, p, t) = (U .= u★ ./ (1 + γΔt); nothing)
@testset "The stage limiter's change reaches f_exp! and nothing else" begin
    for spec in ALL_CALL_COUNTS
        tab = spec.make()
        u0 = [1.0, 2.0]
        for (f!, same) in ((f_time!, true), (f_nonlinear!, false))
            prob = IMEXProblem(f!, solve_decay!, u0, (0.0, 1.0))
            a = solve(prob, tab; dt = 0.1)
            b = solve(prob, tab; dt = 0.1, stage_limiter = limiter_overwrite!)
            # Explicit Euler calls no stage limiter, so nothing changes.
            @test (a.u == b.u) == (same || spec.stage_limiter == 0)
        end
    end
end

# `step!` allocating would cost a garbage collection every few steps of a
# long run; a closure inside a `@testset` allocates itself, so this helper
# is top-level and takes concrete arguments ("Commands" in `CLAUDE.md`).
function step_allocations(integ)
    step!(integ)
    return @allocated step!(integ)
end
@testset "step! is allocation-free after warm-up" begin
    if CHECK_BOUNDS_FORCED
        @info "Skipping the allocation tests under --check-bounds=yes"
    else
        for spec in [ALL_CALL_COUNTS; CORNER_TABLEAUS]
            tab = haskey(spec, :make) ? spec.make() : spec.tab
            # The mocks, both limiters, and a plain run.
            @test step_allocations(mock_integrator(tab, rand(100))) == 0
            prob = IMEXProblem(f_nonlinear!, solve_decay!, rand(100), (0.0, 1.0))
            @test step_allocations(init(prob, tab; dt = 0.1)) == 0
            # A Float32 state with Float64 time, and a complex state.
            @test step_allocations(mock_integrator(tab, rand(Float32, 100))) == 0
            @test step_allocations(mock_integrator(tab, rand(ComplexF64, 100))) == 0
        end
    end
end

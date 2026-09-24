# `IMEXProblem`, the integrator, and CommonSolve's `init`, `step!`,
# `solve!` and `solve` ("The interface", "The callback contracts",
# "Failures and exceptions" and "Time and the step count" in `CODE.md`).

"""
    IMEXProblem(f_exp!, solve_imp!, u0, (t0, t1), p = nothing)

The problem `u′ = f(u, t) + g(u, t)` on `t0 ≤ t ≤ t1`, `u(t0) = u0`, with
`f` non-stiff and treated explicitly and `g` stiff and treated implicitly.
`g` itself is never given: the caller solves its stage equation.

- `f_exp!(du, u, p, t)` writes all of `du = f(u, t)` and does not change
  `u`.
- `solve_imp!(U, u★, γΔt, p, t)` writes `U` so that
  `U = u★ + γΔt g(U, t)`, by any means. `U` and `u★` are distinct arrays;
  on entry `U` holds a copy of `u★`, so the solver need only write the
  components it solves for, and `u★` must not be changed. `γΔt` has the
  state's real type `T`, and `t` the time type. The return value is
  ignored.

It is called once per implicit stage; there is no nonlinear-solver loop,
tolerance or retry here, and convergence, fallbacks and counters belong to
the solver, which reaches them through `p`. `p` is passed to every
callback untouched and defaults to `nothing`. `u0` may be any array that
broadcasts, with a floating-point or complex floating-point element type.

Integrate it with [`init`](@ref) and [`step!`](@ref) or
[`solve!`](@ref), or with [`solve`](@ref). See `CODE.md`, "The interface"
and "The callback contracts".
"""
struct IMEXProblem{F,G,U,Tt,P}
    f_exp!::F
    solve_imp!::G
    u0::U
    tspan::Tuple{Tt,Tt}
    p::P
end

function IMEXProblem(f_exp!, solve_imp!, u0, tspan, p = nothing)
    length(tspan) == 2 || throw(ArgumentError("IMEXProblem: tspan must be (t0, t1), but \
                                               it has $(length(tspan)) elements"))
    t0, t1 = promote(tspan[1], tspan[2])
    return IMEXProblem(f_exp!, solve_imp!, u0, (t0, t1), p)
end

"""
    IMEXIntegrator

The integrator [`init`](@ref) returns. Its public fields are:

- `u`: the state `uⁿ`, which the caller may change in place between steps
  (nothing carries over from one step to the next), but not rebind;
- `t`: the time `tⁿ = t0 + n Δt`, computed afresh at each step, and
  exactly `t1` after the last;
- `dt`: the step `Δt`, of the time type;
- `p`: the parameters;
- `nstep`: the steps taken, `n`;
- `nsteps`: the steps to `t1`;
- `tableau`: the [`IMEXTableau`](@ref).

Every other field is internal. The limiters receive the integrator as
their second argument, as OrdinaryDiffEq's do; during a step, `u` is `uⁿ`
and `t` is `tⁿ`. See `CODE.md`, "The interface".
"""
mutable struct IMEXIntegrator{uType,Tt,P,Tab<:IMEXTableau,F,G,SL,StL,Plan<:StagePlan}
    const u::uType
    t::Tt
    const dt::Tt
    const p::P
    nstep::Int
    const nsteps::Int
    const tableau::Tab
    const t0::Tt
    const t1::Tt
    const f_exp!::F
    const solve_imp!::G
    const stage_limiter::SL
    const step_limiter::StL
    const plan::Plan
end

function Base.show(io::IO, integ::IMEXIntegrator)
    print(io, "IMEXIntegrator(\"", integ.tableau.name, "\", t = ", integ.t, ", step ",
          integ.nstep, " of ", integ.nsteps, ")")
    return nothing
end

"""
    step_count(t0, t1, dt)

`nsteps = ⌈(t1 − t0)/dt⌉`, except that a quotient within a few ulps above
an integer `m` gives `m`: the ulps of `(t1 − t0)/dt`, and those of `t0`
and `t1` relative to `dt`, since `t1 − t0` inherits the rounding of both.
So a chunk meant to be a whole number of steps is not given one more by
round-off ("Time and the step count" in `CODE.md`).
"""
function step_count(t0::Tt, t1::Tt, dt::Tt) where {Tt}
    r = (t1 - t0) / dt
    m = round(r)
    tol = 4 * (eps(r) + (eps(abs(t0)) + eps(abs(t1))) / dt)
    n = m ≥ 1 && abs(r - m) ≤ tol ? m : ceil(r)
    return Int(n)
end

"""
    resolve_partition(u0, partition)

The `partition` keyword of `init`, checked: `nothing` for the broadcast
path, and otherwise an [`OwnerPartition`](@ref) of `eachindex(u0)`, for a
CPU `Array` state only. A partition for any other array type is an
`ArgumentError`, since the by-owner path indexes the state. See
`CODE.md`, "Stage arithmetic".
"""
resolve_partition(u0, ::Nothing) = nothing
function resolve_partition(u0, partition)
    u0 isa Array ||
        throw(ArgumentError("init: partition = $(repr(partition; context = :limit => true)) \
                             is only for a CPU Array state, but the state is a \
                             $(typeof(u0)): the by-owner stage arithmetic indexes the \
                             state, and any other array type takes partition = nothing, \
                             the broadcast"))
    partition isa OwnerPartition || return owner_partition(length(u0), partition)
    (partition.n == length(u0) && length(partition.ranges) == Threads.nthreads()) ||
        throw(ArgumentError("init: the OwnerPartition covers $(partition.n) entries on \
                             $(length(partition.ranges)) threads, but the state has \
                             $(length(u0)) and there are $(Threads.nthreads()) threads"))
    return partition
end

"""
    init(prob::IMEXProblem, tab::IMEXTableau; dt,
         stage_limiter = nothing, step_limiter = nothing,
         partition = nothing, alias_u0 = false)

Build an integrator for `prob` with the tableau `tab` (such as
[`IMEXSSP3433`](@ref)`()`), for fixed steps of at most `dt`.

- **The step.** `init` takes `nsteps = ⌈(t1 − t0)/dt⌉`, with a tolerance
  of a few ulps so that round-off does not add a step, and then
  `Δt = (t1 − t0)/nsteps`. So `Δt ≤ dt`, up to that tolerance. The
  integration runs forward: `t1 > t0` and `dt > 0`.
- **The types.** The time type is `float` of the promoted type of `t0`,
  `t1` and `dt`. The arithmetic type is `T = real(eltype(u0))`, so a
  complex state works, and a `Float32` state with `Float64` time too.
  The coefficients are converted to `T` and the abscissae to the time
  type, once, here.
- **The limiters**, `stage_limiter!(u, integ, p, t)` and
  `step_limiter!(u, integ, p, t)`, change `u` in place; `nothing` means
  no call. The stage limiter is called on the stage value just before
  each `f_exp!` call, on a scratch array, never on `integ.u`, and not at
  a trivial stage (where `f_exp!` reads `integ.u` itself). The step
  limiter is called once per step on `integ.u`, which then holds `uⁿ⁺¹`,
  at `tⁿ⁺¹`; `integ.t` and `integ.nstep` advance after it returns.
- **`alias_u0 = true`** makes `integ.u` be `u0` itself, which saves one
  state-sized array; by default `u0` is copied.
- **`partition`** chooses the stage arithmetic. `nothing`, the default,
  is one fused broadcast per combination, for any array type, device
  arrays included; on the host it is serial. For a CPU `Array` state,
  the by-owner path runs each combination on every thread at once, each
  element on the thread that owns it, so that the stage arrays stay with
  the threads that the caller's own kernels use:
  - an explicit partition has one element per thread,
    `Threads.nthreads()` of them; element `c` is a unit range or a
    collection of unit ranges of indices into `u0`, owned by default-pool
    thread `c`. Together they must cover `eachindex(u0)` exactly once,
    and a gap, an overlap or an index out of bounds is an
    `ArgumentError` naming the index;
  - `:even` splits `eachindex(u0)` into `nthreads()` equal contiguous
    ranges.

  The result is bitwise the same on every path and at every thread
  count. By owner, `init` also writes `integ.u` (unless aliased) and the
  scratch through the partition, for first touch, and a step allocates a
  few small objects per thread per combination, independent of the state
  size (none at one thread).

This builds the stage plan: the scratch arrays, from `similar(u0)` and
written once, and the tableau's nonzero pattern compiled into types, so
that [`step!`](@ref) is type-stable and, on the broadcast path,
allocation-free. See `CODE.md`, "The interface", "Time and the step
count", "The stage plan and storage" and "Stage arithmetic".
"""
function CommonSolve.init(prob::IMEXProblem, tab::IMEXTableau; dt,
                          stage_limiter = nothing, step_limiter = nothing,
                          partition = nothing, alias_u0::Bool = false)
    u0 = prob.u0
    T = real(eltype(u0))
    T <: AbstractFloat ||
        throw(ArgumentError("init: the state's element type is $(eltype(u0)), but it \
                             must be floating-point or complex floating-point, since the \
                             tableau's coefficients are converted to its real type"))
    t0, t1 = prob.tspan
    dt isa Real || throw(ArgumentError("init: dt must be a real number, but it is $dt"))
    Tt = float(promote_type(typeof(t0), typeof(t1), typeof(dt)))
    t0, t1, dt = Tt(t0), Tt(t1), Tt(dt)
    (isfinite(t0) && isfinite(t1)) ||
        throw(ArgumentError("init: tspan = $((t0, t1)) must be finite"))
    t1 > t0 || throw(ArgumentError("init: tspan = $((t0, t1)) must have t1 > t0; the \
                                    integration runs forward over a nonempty interval"))
    (isfinite(dt) && dt > 0) ||
        throw(ArgumentError("init: dt = $dt must be positive and finite"))
    nsteps = step_count(t0, t1, dt)
    Δt = (t1 - t0) / nsteps
    part = resolve_partition(u0, partition)
    u = alias_u0 ? u0 : copy_initial(u0, part)
    plan = build_plan(tab, u, u0, T, Δt, part)
    return IMEXIntegrator(u, t0, Δt, prob.p, 0, nsteps, tab, t0, t1, prob.f_exp!,
                          prob.solve_imp!, stage_limiter, step_limiter, plan)
end

call_limiter!(::Nothing, u, integ, p, t) = nothing
function call_limiter!(limiter!, u, integ, p, t)
    limiter!(u, integ, p, t)
    return nothing
end

# Stage `k` of "One step" in `CODE.md`, at `tⁿ = tn`. The branches are on
# type parameters, so each stage compiles to straight-line code.
@inline function run_stage!(integ::IMEXIntegrator, st::Stage{S,E,I}, tn) where {S,E,I}
    part = integ.plan.partition
    u = integ.u
    if S
        # 1. u★, in `d_k` (or the extra array), and `U = u★` with it; `uⁿ`
        # itself if the row is empty. 2. The stage solve, from `U = u★`,
        # then `d_k = U − u★`.
        if isempty(st.terms)
            copy_state!(st.U, st.ustar, part)
        else
            lincomb_copy!(st.ustar, st.U, u, st.terms, part)
        end
        integ.solve_imp!(st.U, st.ustar, st.γΔt, integ.p, tn + st.c * integ.dt)
        I && increment!(st.d, st.U, st.ustar, part)
    elseif E && !isempty(st.terms)
        # No solve: `U = u★`, formed in `U`.
        lincomb!(st.U, u, st.terms, part)
    end
    if E
        # 3. The limiter acts only where `f_exp!` reads, and not on `uⁿ`
        # at a trivial stage.
        t = tn + st.c̃ * integ.dt
        trivial(st) || call_limiter!(integ.stage_limiter, st.U, integ, integ.p, t)
        integ.f_exp!(st.k̃, st.U, integ.p, t)
    end
    return nothing
end

@inline run_stages!(integ::IMEXIntegrator, ::Tuple{}, tn) = nothing
@inline function run_stages!(integ::IMEXIntegrator, stages::Tuple, tn)
    run_stage!(integ, first(stages), tn)
    return run_stages!(integ, Base.tail(stages), tn)
end

"""
    step!(integ::IMEXIntegrator)

Take one step, from `tⁿ` to `tⁿ⁺¹`, and return `nothing`.

The stages run as in "One step" in `CODE.md`: per stage, `u★`, then one
`solve_imp!` call if `a_kk ≠ 0`, then the stage limiter and `f_exp!` if
the stage's explicit tendency is read. Then the update
`uⁿ⁺¹ = uⁿ + Δt Σ b̃_j k̃_j + Σ (b_j/a_jj) d_j` is written into `integ.u`,
the step limiter is called, and `integ.t` and `integ.nstep` advance. The
last step sets `integ.t = t1` exactly.

`integ.u` is written only by the update and the step limiter. So an
exception from `f_exp!`, `solve_imp!` or the stage limiter leaves
`integ.u = uⁿ` and `integ.t = tⁿ`; after one from the step limiter,
`integ.u` is undefined. A `step!` after the last step throws an
`ArgumentError`. `step!` is type-stable. For callbacks that do not
allocate, it is allocation-free on the broadcast path and, by owner, at
one thread; by owner at more threads it allocates a few hundred bytes per
thread per combination, whatever the state size (`CODE.md`, "By owner,
as built").
"""
function CommonSolve.step!(integ::IMEXIntegrator)
    n = integ.nstep
    n < integ.nsteps ||
        throw(ArgumentError("step!: the integrator has taken all its $(integ.nsteps) \
                             steps and is at t1 = $(integ.t1); build a fresh one with \
                             init for the next interval"))
    tn = integ.t0 + n * integ.dt
    plan = integ.plan
    run_stages!(integ, plan.stages, tn)
    lincomb!(integ.u, integ.u, plan.update, plan.partition)
    tnext = n + 1 == integ.nsteps ? integ.t1 : integ.t0 + (n + 1) * integ.dt
    call_limiter!(integ.step_limiter, integ.u, integ, integ.p, tnext)
    integ.t = tnext
    integ.nstep = n + 1
    return nothing
end

"""
    solve!(integ::IMEXIntegrator)

Step to `t1`: [`step!`](@ref) until `integ.nstep == integ.nsteps`. Returns
`integ`, whose `u` is then the state at `t1` and whose `t` is exactly
`t1`. On an integrator already at `t1` it does nothing. See `CODE.md`,
"The interface".
"""
function CommonSolve.solve!(integ::IMEXIntegrator)
    while integ.nstep < integ.nsteps
        step!(integ)
    end
    return integ
end

"""
    solve(prob::IMEXProblem, tab::IMEXTableau; dt, kwargs...)

`solve!(init(prob, tab; dt, kwargs...))`: integrate to `t1` and return the
integrator, whose `u` is the state at `t1`. There is no solution type;
the keywords are those of [`init`](@ref). See `CODE.md`, "The
interface".
"""
function CommonSolve.solve(prob::IMEXProblem, tab::IMEXTableau; kwargs...)
    return solve!(init(prob, tab; kwargs...))
end

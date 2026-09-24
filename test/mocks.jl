# Test-only mock callbacks. They record every call into buffers
# preallocated in `p`, so that the mechanics tests can also run under the
# allocation helper ("Mocks" in `PLAN.md`'s sharp edges).

using IMEXRungeKutta: IMEXRungeKutta

# One entry per callback call: which callback, its time, its array
# argument (`u` for `f_exp!` and the limiters, `U` for `solve_imp!`), and
# what the callback saw of the integrator and of its other arguments.
mutable struct CallLog{A,T,Tt}
    n::Int
    kind::Vector{Symbol}
    t::Vector{Tt}
    arr::Vector{A}
    # `solve_imp!`: `γΔt`, the `u★` array, whether `U` entered as a copy of
    # `u★` and a distinct array, and whether `u★` was unchanged after `U`
    # was written.
    γΔt::Vector{T}
    ustar::Vector{A}
    U_was_ustar::Vector{Bool}
    distinct::Vector{Bool}
    ustar_kept::Vector{Bool}
    # The limiters: `integ.t` and `integ.nstep` as they saw them.
    integ_t::Vector{Tt}
    integ_nstep::Vector{Int}
    # A snapshot of `u★`, and the call (of any kind) at which to throw, or 0.
    snap::A
    throw_at::Int
end

function CallLog(u0::A, ::Type{Tt}; capacity = 1000) where {A,Tt}
    T = real(eltype(u0))
    return CallLog{A,T,Tt}(0, fill(:none, capacity), zeros(Tt, capacity),
                           Vector{A}(undef, capacity), zeros(T, capacity),
                           Vector{A}(undef, capacity), falses(capacity), falses(capacity),
                           falses(capacity), zeros(Tt, capacity), zeros(Int, capacity),
                           similar(u0), 0)
end

reset!(log::CallLog) = (log.n = 0; log.throw_at = 0; log)

struct MockFailure <: Exception
    call::Int
end

function record!(log::CallLog, kind::Symbol, t, arr)
    i = log.n += 1
    log.kind[i] = kind
    log.t[i] = t
    log.arr[i] = arr
    i == log.throw_at && throw(MockFailure(i))
    return i
end

# The calls of kind `kind` in the log, as indices.
calls(log::CallLog, kind::Symbol) = [i for i in 1:(log.n) if log.kind[i] === kind]
calls(log::CallLog) = log.kind[1:(log.n)]

# `u′ = −u + cos t`, componentwise: the explicit part `cos t`, and the
# implicit part `−u`, whose stage solve is `U = u★/(1 + γΔt)`.
function mock_f!(du, u, log::CallLog, t)
    record!(log, :f_exp, t, u)
    du .= cos(t)
    return nothing
end

function mock_solve!(U, ustar, γΔt, log::CallLog, t)
    i = record!(log, :solve_imp, t, U)
    log.γΔt[i] = γΔt
    log.ustar[i] = ustar
    log.U_was_ustar[i] = U == ustar
    log.distinct[i] = U !== ustar
    copyto!(log.snap, ustar)
    U .= ustar ./ (1 + γΔt)
    log.ustar_kept[i] = ustar == log.snap
    return nothing
end

function mock_stage_limiter!(u, integ, log::CallLog, t)
    i = record!(log, :stage_limiter, t, u)
    log.integ_t[i] = integ.t
    log.integ_nstep[i] = integ.nstep
    return nothing
end

function mock_step_limiter!(u, integ, log::CallLog, t)
    i = record!(log, :step_limiter, t, u)
    log.integ_t[i] = integ.t
    log.integ_nstep[i] = integ.nstep
    return nothing
end

# An integrator over the mocks, with both limiters, on `(t0, t1)` with
# step `dt`.
function mock_integrator(tab, u0; tspan = (0.0, 1.0), dt = 0.1, kwargs...)
    log = CallLog(u0, float(promote_type(typeof.(tspan)..., typeof(dt))))
    prob = IMEXProblem(mock_f!, mock_solve!, u0, tspan, log)
    return init(prob, tab; dt, stage_limiter = mock_stage_limiter!,
                step_limiter = mock_step_limiter!, kwargs...)
end

# The seven named tableaus, with the calls one step makes, counted by hand
# from "One step" in `CODE.md`, and the scratch count of "Measured
# properties" there.
const CALL_COUNTS = [
    (make = IMEXSSP222, f_exp = 2, solve_imp = 2, stage_limiter = 2, scratch = 5),
    (make = IMEXSSP2322, f_exp = 2, solve_imp = 3, stage_limiter = 2, scratch = 6),
    (make = IMEXSSP2332, f_exp = 3, solve_imp = 3, stage_limiter = 3, scratch = 7),
    (make = IMEXSSP3332, f_exp = 3, solve_imp = 3, stage_limiter = 3, scratch = 7),
    (make = IMEXSSP3433, f_exp = 3, solve_imp = 4, stage_limiter = 3, scratch = 8),
    # ARS: a trivial first stage, which calls `f_exp!` on `integ.u` itself
    # with no stage limiter.
    (make = ARS222, f_exp = 2, solve_imp = 2, stage_limiter = 1, scratch = 5),
    (make = ARS443, f_exp = 4, solve_imp = 4, stage_limiter = 3, scratch = 9),
]
const NAMED_TABLEAUS = [spec.make for spec in CALL_COUNTS]

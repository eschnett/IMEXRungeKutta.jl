# The stage arithmetic, broadcast against by owner, over a thread sweep
# ("Stage arithmetic" in CODE.md; step 5 of PLAN.md).
#
#     julia --project=. bench/stage_arithmetic.jl
#
# runs itself once per thread count in IMEXRK_BENCH_THREADS (default
# 1,2,4,6,8,12), each a fresh `julia -t n`, and prints one tab-separated
# line per measurement:
#
#     threads  path  what  entries  min_ms  median_ms  GB/s  bytes_allocated
#
# on a `Vector{Float64}` state of IMEXRK_BENCH_BYTES bytes (default 10⁸,
# 12.5 million entries). GB/s counts the state-sized reads and writes the
# arithmetic makes (no write-allocate), over the least time. `what` is
#
#     update    one combination of x₀ and 7 terms, 8 reads and 1 write
#               (SSP3(4,3,3)'s update has 6 terms, ARS(4,4,3)'s 8), into
#               an array of its own
#     step      one SSP3(4,3,3) step with trivial callbacks (`f_exp!` and
#               `solve_imp!` return at once), so the step is its stage
#               arithmetic alone
#     launch    `update` on 1000 entries per thread: the cost of a
#               launch, not of the data
#
# and `path` is
#
#     broadcast   partition = nothing, one fused broadcast
#     owner       partition = :even, the package's fresh sticky tasks
#     persistent  the update only: a prototype of persistent sticky
#                 workers (below), one per default-pool thread, each
#                 waiting on its own Event, with the job built once; it is
#                 here to be measured against `owner`, not used
#
# IMEXRK_BENCH_REPS (default 20) repetitions per measurement, after one
# warm-up; IMEXRK_BENCH_INTERACTIVE (below) the interactive pool's size.
# The Symmetry job is bench/symmetry_stage_arithmetic.sh.

using IMEXRungeKutta
using IMEXRungeKutta: IMEXProblem, IMEXSSP3433, lincomb!, first_touch!, owner_partition,
                      run_owned, LinComb, getindices
using Printf

const THREADS = get(ENV, "IMEXRK_BENCH_THREADS", "1,2,4,6,8,12")
const BYTES = parse(Float64, get(ENV, "IMEXRK_BENCH_BYTES", "1e8"))
const REPS = parse(Int, get(ENV, "IMEXRK_BENCH_REPS", "20"))

if get(ENV, "IMEXRK_BENCH_CHILD", "") != "1"
    println("# ", Sys.cpu_info()[1].model, ", ", Sys.CPU_THREADS, " CPU threads, Julia ",
            VERSION)
    println("threads\tpath\twhat\tentries\tmin_ms\tmedian_ms\tGB/s\tbytes_allocated")
    flush(stdout)
    for nt in parse.(Int, split(THREADS, ","))
        # IMEXRK_BENCH_INTERACTIVE, if set, is the interactive pool's size:
        # 0 keeps a pinned run (JULIA_EXCLUSIVE=1) at one thread per core,
        # where Julia 1.12 and later would otherwise add an interactive one.
        k = get(ENV, "IMEXRK_BENCH_INTERACTIVE", "")
        threads = isempty(k) ? "$nt" : "$nt,$k"
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$threads $(@__FILE__)`
        run(addenv(cmd, "IMEXRK_BENCH_CHILD" => "1"))
    end
    exit(0)
end

# A prototype of persistent sticky workers: one task per default-pool
# thread, placed as the package places its fresh ones, each waiting on its
# own autoreset Event. The job is a callable built once, called with the
# thread number (a small `Int`, which a dynamic call does not box), so a
# launch allocates nothing. A caller would need one such job per
# combination, built in `init`, and a process-wide pool behind a lock.
mutable struct Pool
    const go::Vector{Threads.Event}
    const done::Threads.Event
    const remaining::Threads.Atomic{Int}
    job::Any
    const lock::ReentrantLock
end

function worker(pool::Pool, c::Int)
    while true
        wait(pool.go[c])
        pool.job(c)
        Threads.atomic_sub!(pool.remaining, 1) == 1 && notify(pool.done)
    end
end

function Pool()
    nt = Threads.nthreads()
    pool = Pool([Threads.Event(true) for _ in 1:nt], Threads.Event(true),
                Threads.Atomic{Int}(0), nothing, ReentrantLock())
    offset = Threads.threadpoolsize(:interactive)
    for c in 1:nt
        task = Task(() -> worker(pool, c))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, offset + c - 1)
        schedule(task)
    end
    return pool
end

function launch!(pool::Pool, job)
    lock(pool.lock)
    try
        pool.job = job
        pool.remaining[] = length(pool.go)
        foreach(notify, pool.go)
        wait(pool.done)
    finally
        unlock(pool.lock)
    end
    return nothing
end

struct Kernel{D,X,L,XS}
    dst::D
    x₀::X
    l::L
    xs::XS
end
function (k::Kernel)(r)
    dst, x₀, l, xs = k.dst, k.x₀, k.l, k.xs
    @inbounds @simd ivdep for i in r
        dst[i] = l(x₀[i], getindices(xs, i)...)
    end
    return nothing
end
mutable struct Job{K,P}
    kernel::K
    part::P
end
(job::Job)(c::Int) = run_owned(job.kernel, job.part, c)

# Timing: the least and the median of REPS, after one warm-up, and the
# bytes allocated by one more call.
function measure(f)
    f()
    ts = Float64[]
    for _ in 1:REPS
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e6)
    end
    sort!(ts)
    alloc = @allocated f()
    return ts[1], ts[(length(ts) + 1) ÷ 2], alloc
end

function report(path, what, n, passes, (tmin, tmed, alloc))
    gbs = passes * n * sizeof(Float64) / (tmin * 1e-3) / 1e9
    @printf("%d\t%s\t%s\t%d\t%.4f\t%.4f\t%.1f\t%d\n", Threads.nthreads(), path, what, n,
            tmin, tmed, gbs, alloc)
    flush(stdout)
    return nothing
end

function bench_update(n, what)
    part = owner_partition(n, :even)
    # First touch by owner, as `init` does, so that on a NUMA node each
    # thread's pages are on its own domain; then the values, also by owner.
    src = rand(n)
    function owned()
        a = first_touch!(Vector{Float64}(undef, n), part)
        return lincomb!(a, src, (), part)
    end
    x₀ = owned()
    dst = owned()
    terms = ntuple(j -> (0.1 * j, owned()), 7)
    passes = length(terms) + 2
    report("broadcast", what, n, passes, measure(() -> lincomb!(dst, x₀, terms, nothing)))
    report("owner", what, n, passes, measure(() -> lincomb!(dst, x₀, terms, part)))
    if Threads.nthreads() > 1
        pool = Pool()
        job = Job(Kernel(dst, x₀, LinComb(map(first, terms)), map(last, terms)), part)
        report("persistent", what, n, passes, measure(() -> launch!(pool, job)))
    end
    return nothing
end

# One SSP3(4,3,3) step with callbacks that return at once: the stage
# arithmetic alone. The state-sized passes are counted from the plan.
f_trivial!(du, u, p, t) = nothing
solve_trivial!(U, u★, γΔt, p, t) = nothing

function step_passes(plan, owner::Bool)
    n = length(plan.update) + 2
    for st in plan.stages
        m = length(st.terms)
        if IMEXRungeKutta.solves(st)
            # u★ and the copy into U: two passes on the broadcast path, one
            # by owner; then the increment, 2 reads and 1 write.
            n += m == 0 ? 2 : owner ? m + 3 : (m + 2) + 2
            IMEXRungeKutta.implicit_used(st) && (n += 3)
        elseif IMEXRungeKutta.explicit_used(st) && m > 0
            n += m + 2
        end
    end
    return n
end

function bench_step(n)
    prob = IMEXProblem(f_trivial!, solve_trivial!, rand(n), (0.0, 1.0))
    for (path, partition) in (("broadcast", nothing), ("owner", :even))
        integ = init(prob, IMEXSSP3433(); dt = 1e-6, partition)
        passes = step_passes(integ.plan, partition !== nothing)
        report(path, "step", n, passes, measure(() -> step!(integ)))
    end
    return nothing
end

const N = round(Int, BYTES / sizeof(Float64))
bench_update(N, "update")
bench_step(N)
bench_update(1000 * Threads.nthreads(), "launch")

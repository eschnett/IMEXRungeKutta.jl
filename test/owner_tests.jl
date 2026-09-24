using IMEXRungeKutta: IMEXRungeKutta, IMEXProblem, IMEXTableau
using IMEXRungeKutta: IMEXSSP222, IMEXSSP3433, ARS443
using IMEXRungeKutta: OwnerPartition, owner_partition, even_partition, block_partition
using IMEXRungeKutta: by_owner, lincomb!, scratch_count

# The stage arithmetic by owner ("Stage arithmetic" and "The stage plan and
# storage" in `CODE.md`; step 5 of `PLAN.md`). The suite runs at one thread
# and at four ("Commands" in `CLAUDE.md`), and every claim here is checked
# at whichever it runs at.

const NT = Threads.nthreads()
# The default pool's thread ids follow the interactive pool's: thread `c`
# of the partition is `Threads.threadid() == OFFSET + c`.
const OFFSET = Threads.threadpoolsize(:interactive)

bits(a) = reinterpret(UInt8, vec(a))

# A multi-set state vector, as TreeAMR lays one out: three field sets, each
# a segment of `NB` blocks of its own length, and thread `c` owns the same
# blocks in every set, so it owns one range per set.
const NB = 7
const SET_LENGTHS = (11, 13, 5)
const SET_OFFSETS = cumsum((0, (NB .* SET_LENGTHS)...))
const NSTATE = SET_OFFSETS[end]       # 203: no multiple of a SIMD width
block_ranges(nb) = [((c - 1) * nb ÷ NT + 1):(c * nb ÷ NT) for c in 1:NT]
multiset_partition() = block_partition(block_ranges(NB),
                                       [(SET_OFFSETS[s], SET_LENGTHS[s]) for s in 1:3])

# Irregular by hand: ranges of odd lengths, out of order within a thread,
# interleaved between threads, and, at more than one thread, a last thread
# that owns nothing.
function irregular_partition(n)
    pieces = UnitRange{Int}[]
    lo = 1
    for len in Iterators.cycle((1, 17, 3, 32, 9, 2))
        lo > n && break
        push!(pieces, lo:min(n, lo + len - 1))
        lo += len
    end
    owners = NT == 1 ? 1 : NT - 1
    part = [UnitRange{Int}[] for _ in 1:NT]
    for (i, r) in enumerate(pieces)
        pushfirst!(part[mod1(i, owners)], r)
    end
    return part
end

# `u′ = cos t − u²·(1 + u) − (u − ū)/ε`: an explicit part that depends on `u`
# and `t` nonlinearly, and a relaxation whose stage solve has a closed form
# and uses `u★` componentwise; limiters that change the state.
function f_owner!(du, u, p, t)
    @. du = cos(t) - u^2 * (1 + u)
    return nothing
end
function solve_owner!(U, u★, γΔt, p, t)
    @. U = (u★ + (γΔt / p.ε) * p.ū) / (1 + γΔt / p.ε)
    return nothing
end
# Scale back any entry above `bound` in modulus; for a complex state too.
function clamp_limiter!(u, integ, p, t)
    b = p.bound
    @. u = ifelse(abs(u) > b, u * (b / abs(u)), u)
    return nothing
end

function owner_problem(u0)
    T = real(eltype(u0))
    p = (ε = T(1 // 100), ū = map(x -> x / 2, u0), bound = T(9 // 10))
    return IMEXProblem(f_owner!, solve_owner!, u0, (0.0, 1.0), p)
end

function owner_run(tab, u0, partition; nsteps = 4)
    integ = init(owner_problem(u0), tab; dt = 0.1, partition,
                 stage_limiter = clamp_limiter!, step_limiter = clamp_limiter!)
    us = [begin
              step!(integ)
              copy(integ.u)
          end
          for _ in 1:nsteps]
    return us
end

function random_state(::Type{T}, n) where {T}
    u = rand(T, n)
    # Magnitudes over 2⁻²⁰ to 1, so that a reordered or contracted sum
    # rounds differently somewhere.
    return u .* map(x -> real(T)(2)^(-20 * x), rand(real(T), n))
end

all_tableaus() = [[spec.make() for spec in CALL_COUNTS];
                  [spec.tab for spec in CORNER_TABLEAUS]]

# A per-element kernel that differed from the broadcast's, in its term
# order, a contraction to FMA or a reassociation under `@simd`, would make
# the result depend on the path and on the thread count ("The result does
# not depend on the path" in `CODE.md`).
@testset "Broadcast, :even and multi-range partitions give the same bits" begin
    for T in (Float64, Float32, ComplexF64), tab in all_tableaus()
        u0 = random_state(T, NSTATE)
        ref = owner_run(tab, u0, nothing)
        for partition in (:even, multiset_partition(), irregular_partition(NSTATE))
            got = owner_run(tab, u0, partition)
            @test all(n -> bits(got[n]) == bits(ref[n]), eachindex(ref))
        end
    end
end

# Forming `u★` and `U` in one pass is where the owner path differs from the
# broadcast in what it does, not only in where: a `U` left unwritten, or
# holding another stage's value, reaches a solver that writes only some
# components ("The callback contracts" in `CODE.md`). The mocks record what
# `solve_imp!` saw; scratch full of NaN must still leave no NaN.
function solve_first_half!(U, u★, γΔt, p, t)
    h = length(U) ÷ 2
    @views U[1:h] .= u★[1:h] ./ (1 + γΔt)
    return nothing
end
@testset "By owner, solve_imp! gets U = u★, distinct, and u★ kept" begin
    for tab in all_tableaus(), partition in (:even, irregular_partition(NSTATE))
        u0 = random_state(Float64, NSTATE)
        a = mock_integrator(tab, u0; partition)
        b = mock_integrator(tab, u0)
        for n in 1:3
            reset!(a.p)
            reset!(b.p)
            foreach(x -> fill!(x, NaN), a.plan.scratch)
            step!(a)
            step!(b)
            solves = calls(a.p, :solve_imp)
            @test all(i -> a.p.U_was_ustar[i] && a.p.distinct[i] && a.p.ustar_kept[i], solves)
            @test calls(a.p) == calls(b.p)
            @test bits(a.u) == bits(b.u)
        end
        # A solver that writes half of `U`: the other half is `u★`'s.
        prob(u) = IMEXProblem(f_owner!, solve_first_half!, u, (0.0, 1.0), nothing)
        c = init(prob(u0), tab; dt = 0.1, partition)
        d = init(prob(u0), tab; dt = 0.1)
        for n in 1:3
            foreach(x -> fill!(x, NaN), c.plan.scratch)
            step!(c)
            step!(d)
        end
        @test bits(c.u) == bits(d.u)
    end
end

# The two traps a left-to-right fold without contraction must not fall
# into, over a range long enough to be vectorized: `(1 + 2⁵⁴) − 2⁵⁴` is 0
# left to right and 1 reassociated, and `−1 + (1 + 2⁻³⁰)(1 − 2⁻³⁰)` is 0
# with the product rounded and `−2⁻⁶⁰` as an FMA (in `Float32`, 2²⁵ and
# 2⁻¹³).
@testset "The owner loop folds left to right, with no FMA and no reassociation" begin
    n = 1027
    for T in (Float64, Float32), partition in (:even, irregular_partition(n))
        part = owner_partition(n, partition)
        big = T(2)^(T === Float64 ? 54 : 25)
        dst = fill(T(NaN), n)
        lincomb!(dst, ones(T, n), ((one(T), fill(big, n)), (one(T), fill(-big, n))), part)
        @test all(iszero, dst)
        e = T(2)^(T === Float64 ? -30 : -13)
        lincomb!(dst, fill(-one(T), n), ((1 + e, fill(1 - e, n)),), part)
        @test all(iszero, dst)
        bc = fill(T(NaN), n)
        lincomb!(bc, fill(-one(T), n), ((1 + e, fill(1 - e, n)),), nothing)
        @test bits(bc) == bits(dst)
    end
end

# A task placed on the wrong thread, or `jl_set_task_tid` given a 1-based
# id or one that ignores the interactive pool, would silently move every
# block to another core in every combination, which is what the owner path
# exists to prevent ("Stage arithmetic" in `CODE.md`).
struct RangeLog
    tid::Matrix{Int}
    count::Matrix{Int}
end
RangeLog(nranges) = RangeLog(zeros(Int, NT, nranges), zeros(Int, NT, nranges))
function (log::RangeLog)(c, j, r)
    log.tid[c, j] = Threads.threadid()
    log.count[c, j] += 1
    return nothing
end
reset!(log::RangeLog) = (fill!(log.tid, 0); fill!(log.count, 0); log)

# Combinations per step by owner, counted from "One step": a solving stage
# forms `u★` and `U` in one pass (or copies `uⁿ` into `U`) and takes its
# increment if it is implicit-used; a stage that does not solve forms `U`
# if it is explicit-used and its row is not empty; then the update.
function owner_launches(tab)
    R = IMEXRungeKutta
    sol, ex, imp = R.solves(tab), R.explicit_used(tab), R.implicit_used(tab)
    n = 1
    for k in eachindex(sol)
        if sol[k]
            n += 1 + imp[k]
        elseif ex[k] && !R.row_empty(tab, k)
            n += 1
        end
    end
    return n
end

@testset "Each range runs on its own thread, once per combination" begin
    # Hand-counted: SSP3(4,3,3) makes 2 passes at stage 1 (U from uⁿ, and
    # d₁), 2 at each of stages 2–4 (u★ with U, and d_k) and the update;
    # ARS(4,4,3) nothing at its trivial stage 1, 2 at each of stages 2–5,
    # and the update; SSP2(2,2,2) 2 + 2 + 1.
    @test owner_launches(IMEXSSP3433()) == 9
    @test owner_launches(ARS443()) == 9
    @test owner_launches(IMEXSSP222()) == 5
    for tab in (IMEXSSP3433(), ARS443(), IMEXSSP222())
        partition = multiset_partition()
        log = RangeLog(3)
        part = owner_partition(NSTATE, partition; hook = log)
        integ = init(owner_problem(rand(NSTATE)), tab; dt = 0.1, partition = part)
        # `init`'s first touch: `integ.u`, then each scratch array.
        @test all(==(1 + scratch_count(tab)), log.count)
        for n in 1:3
            reset!(log)
            step!(integ)
            @test all(==(owner_launches(tab)), log.count)
            if NT == 1
                @test all(==(Threads.threadid()), log.tid)
            else
                @test all(c -> all(==(OFFSET + c), log.tid[c, :]), 1:NT)
            end
        end
    end
end

# The caller's own kernels may run on threads too, and a `step!` inside a
# spawned task, a sticky task or a threaded region must neither deadlock
# nor lose its placement ("Stage arithmetic" in `CODE.md`: the tasks are
# sticky without entering a threaded region, as TreeAMR's are).
function nested_step!(integ, how)
    if how === :spawn
        fetch(Threads.@spawn step!(integ))
    elseif how === :sticky
        task = Task(() -> step!(integ))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, OFFSET + NT - 1)
        schedule(task)
        wait(task)
    elseif how === :threads
        Threads.@threads :static for c in 1:NT
            c == NT && step!(integ)
        end
    end
    return nothing
end
@testset "A partitioned step! works from inside a task and a threaded region" begin
    u0 = random_state(Float64, NSTATE)
    ref = owner_run(IMEXSSP3433(), u0, nothing; nsteps = 3)
    for how in (:spawn, :sticky, :threads)
        log = RangeLog(3)
        part = owner_partition(NSTATE, multiset_partition(); hook = log)
        integ = init(owner_problem(u0), IMEXSSP3433(); dt = 0.1, partition = part,
                     stage_limiter = clamp_limiter!, step_limiter = clamp_limiter!)
        for n in 1:3
            reset!(log)
            nested_step!(integ, how)
            @test bits(integ.u) == bits(ref[n])
            @test all(==(owner_launches(IMEXSSP3433())), log.count)
            NT == 1 || @test all(c -> all(==(OFFSET + c), log.tid[c, :]), 1:NT)
        end
    end
end

# A gap would leave entries of the state never written, an overlap would
# race two threads on one entry; either must be refused before the first
# step, with the index named ("Stage arithmetic" in `CODE.md`).
@testset "A gap, an overlap or a malformed partition is refused, naming the index" begin
    msg(f) = try
        f()
        ""
    catch e
        e isa ArgumentError ? e.msg : "not an ArgumentError: $e"
    end
    n = 100
    prob = owner_problem(rand(n))
    refuse(partition) = msg(() -> init(prob, IMEXSSP222(); dt = 0.1, partition))
    pad(ranges...) = [collect(ranges); [UnitRange{Int}[] for _ in 1:(NT - 1)]]
    @test refuse(pad([1:40, 42:100])) == "init: the partition misses index 41 of the state \
                                          (1:100): no thread owns it"
    @test refuse(pad([1:99])) == "init: the partition misses index 100 of the state \
                                  (1:100): no thread owns it"
    @test refuse(pad([2:100])) == "init: the partition misses index 1 of the state \
                                   (1:100): no thread owns it"
    @test refuse(pad([1:50, 50:100])) == "init: the partition doubles index 50: thread 1's \
                                          range 1:50 and thread 1's range 50:100 both own it"
    if NT > 1
        overlap = [[1:60]; [[60:100]]; [UnitRange{Int}[] for _ in 1:(NT - 2)]]
        @test refuse(overlap) == "init: the partition doubles index 60: thread 1's range \
                                  1:60 and thread 2's range 60:100 both own it"
    end
    @test occursin("thread 1's range 1:101 of the partition is out of bounds",
                   refuse(pad([1:101])))
    @test occursin("thread 1's range 0:100 of the partition is out of bounds",
                   refuse(pad([0:100])))
    @test occursin("one element per thread, Threads.nthreads() = $NT, but it has $(NT + 1)",
                   refuse([[1:n]; [UnitRange{Int}[] for _ in 1:NT]]))
    @test occursin("holds 7, which is not a unit range", refuse(pad([1:n, 7])))
    @test occursin("holds 1:2:99, which is not a unit range", refuse(pad([1:2:99])))
    @test occursin(":odd is not a partition", refuse(:odd))
    # A plain range per thread is accepted, as is `:even`.
    @test init(prob, IMEXSSP222(); dt = 0.1, partition = pad(1:n)).plan.partition isa
          OwnerPartition
    @test init(prob, IMEXSSP222(); dt = 0.1, partition = :even).plan.partition.ranges ==
          even_partition(n)
    # Not a CPU `Array`: a view of one.
    vprob = owner_problem(view(rand(n), 1:n))
    @test occursin("only for a CPU Array",
                   msg(() -> init(vprob, IMEXSSP222(); dt = 0.1, partition = :even)))
    # An `OwnerPartition` built for another length.
    @test occursin("covers 99 entries",
                   msg(() -> init(prob, IMEXSSP222(); dt = 0.1,
                                  partition = owner_partition(99, :even))))
end

# `:even` is TreeAMR's `threadchunks` rule, one entry per thread, so that
# a caller who also uses it gets the same split ("Stage arithmetic").
@testset ":even splits 1:n by the threadchunks rule, one entry per thread" begin
    for n in (0, 1, NT - 1, NT, 10NT + 3, 1001)
        ranges = even_partition(n)
        @test length(ranges) == NT
        @test all(==(1) ∘ length, ranges)
        flat = only.(ranges)
        @test reduce(vcat, collect.(flat); init = Int[]) == 1:n
        lens = length.(flat)
        @test maximum(lens) - minimum(lens) ≤ 1
        @test issorted(lens; rev = true)
    end
end

# The helper's ranges must be those of the blocks each thread owns, in
# every segment, or the stage arithmetic would move blocks between cores
# even with a partition given ("Where the partition comes from").
@testset "block_partition gives each thread its blocks in every segment" begin
    # Two segments of 4 blocks, of 3 and 2 entries; blocks 1:1, 2:4 on two
    # threads: thread 1 owns 1:3 and 13:14, thread 2 owns 4:12 and 15:20.
    @test block_partition([1:1, 2:4], [(0, 3), (12, 2)]) == [[1:3, 13:14], [4:12, 15:20]]
    # A thread with no blocks owns nothing.
    @test all(isempty, block_partition([1:0], [(0, 3), (12, 2)])[1])
    @test owner_partition(NSTATE, multiset_partition()) isa OwnerPartition
end

# `init` writes `integ.u` and the scratch through the partition for first
# touch; a copy of `u0` taken otherwise, or scratch left for the first
# step to touch, would place their pages on the wrong NUMA domain
# ("Storage" in `CODE.md`). `alias_u0` still means `u0` itself.
@testset "init copies u0 and writes the scratch through the partition" begin
    u0 = random_state(Float64, NSTATE)
    integ = init(owner_problem(u0), IMEXSSP3433(); dt = 0.1, partition = :even)
    @test integ.u !== u0 && bits(integ.u) == bits(u0)
    @test all(a -> all(iszero, a), integ.plan.scratch)
    aliased = init(owner_problem(u0), IMEXSSP3433(); dt = 0.1, partition = :even,
                   alias_u0 = true)
    @test aliased.u === u0
end

# An error on a worker must reach the caller as itself, not wrapped in a
# `TaskFailedException`, and only after every worker has finished, so that
# nothing is still writing when the caller handles it.
@testset "An error on a worker reaches the caller as itself, after all finish" begin
    part = owner_partition(NSTATE, multiset_partition())
    done = zeros(Int, NT, 3)
    function body(r)
        c = findfirst(rs -> r in rs, part.ranges)
        j = findfirst(==(r), part.ranges[c])
        done[c, j] += 1
        c == 1 && j == 2 && throw(ArgumentError("range $r"))
        return nothing
    end
    err = try
        by_owner(body, part)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test err isa ArgumentError && err.msg == "range $(part.ranges[1][2])"
    # Thread 1 stops at its second range; every other range ran.
    @test done[1, 3] == 0
    @test all(==(1), [done[c, j] for c in 1:NT, j in 1:3 if (c, j) != (1, 3)])
    # A resized state is refused rather than written out of bounds.
    integ = init(owner_problem(rand(NSTATE)), IMEXSSP222(); dt = 0.1, partition = :even)
    resize!(integ.u, NSTATE + 1)
    @test_throws DimensionMismatch step!(integ)
end

@testset "step! by owner is inferred" begin
    integ = init(owner_problem(rand(NSTATE)), IMEXSSP3433(); dt = 0.1,
                 partition = multiset_partition())
    @test (@inferred step!(integ)) === nothing
end

# The owner path's allocations: none at one thread, where it is a plain
# loop; at more, the fresh sticky tasks, a few hundred bytes per thread
# per combination, the same for any state size ("Stage arithmetic" in
# `CODE.md`). A state-sized allocation here would be a copy. The helper is
# top-level ("Allocation tests" in `PLAN.md`).
function owner_step_allocations(integ)
    step!(integ)
    step!(integ)
    return @allocated step!(integ)
end
function owner_integrator(tab, n; T = Float64)
    return init(owner_problem(random_state(T, n)), tab; dt = 0.01,
                partition = :even, stage_limiter = clamp_limiter!,
                step_limiter = clamp_limiter!)
end
@testset "step! by owner allocates nothing at one thread, and O(nthreads) at more" begin
    if CHECK_BOUNDS_FORCED
        @info "Skipping the allocation tests under --check-bounds=yes"
    else
        for tab in all_tableaus(), T in (Float64, Float32, ComplexF64)
            small = owner_step_allocations(owner_integrator(tab, 100; T))
            large = owner_step_allocations(owner_integrator(tab, 100_000; T))
            if NT == 1
                @test small == 0
                @test large == 0
            else
                @test small == large
                # Measured in step 5 (`CODE.md`, "Stage arithmetic"): per
                # combination, 64 bytes and, per thread, 403–433 on Julia
                # 1.13 and 559–589 on 1.10, growing with the terms.
                @test large ≤ owner_launches(tab) * (128 + 768 * NT)
            end
        end
    end
end

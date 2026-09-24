# The stage arithmetic: every combination the integrator forms is one fused
# linear combination `dst = x₀ + Σ c_j x_j`, one pass over the state
# ("Stage arithmetic" in `CODE.md`).
#
# Each function takes a partition as its last argument. `nothing` is the
# broadcast path: one fused broadcast, which works for any array type that
# broadcasts, device arrays included, and never indexes the state. An
# `OwnerPartition` is the by-owner path of step 5, for a CPU `Array`: a
# plain loop per range, each range on the thread that owns it (below).

# The per-element kernel of a linear combination with the coefficients
# `c`: `x₀ + c₁x₁ + c₂x₂ + …`, summed left to right, in plan order. The
# order is fixed so that every path, and every thread count, gives the same
# bits ("The result does not depend on the path" in `CODE.md`). It is an
# isbits callable when the coefficients are, so a device broadcast can
# capture it.
struct LinComb{C<:Tuple}
    c::C
end

@inline (l::LinComb)(x₀, xs...) = lincomb_sum(x₀, l.c, xs)

@inline lincomb_sum(acc, ::Tuple{}, ::Tuple{}) = acc
@inline function lincomb_sum(acc, c::Tuple, x::Tuple)
    return lincomb_sum(acc + first(c) * first(x), Base.tail(c), Base.tail(x))
end

"""
    lincomb!(dst, x₀, terms, partition)

Write `dst = x₀ + Σ c_j x_j` for the `(c_j, x_j)` pairs of the tuple
`terms`, in one pass, with the terms summed in their order. `dst` may be
`x₀` itself (the update writes `integ.u` in place, and `d_k = U − u★`
overwrites `u★`), but must not otherwise overlap an operand. With no terms
this is a copy. `partition === nothing` is one fused broadcast.
"""
function lincomb!(dst, x₀, terms::Tuple, ::Nothing)
    cs = map(first, terms)
    xs = map(last, terms)
    broadcast!(LinComb(cs), dst, x₀, xs...)
    return dst
end

"""
    copy_state!(dst, src, partition)

`dst = src`, one pass: `U` gets its copy of `u★` before the stage solve
("The callback contracts" in `CODE.md`).
"""
copy_state!(dst, src, partition) = lincomb!(dst, src, (), partition)

"""
    increment!(d, U, u★, partition)

The implicit increment `d = U − u★`, one pass. `d` may be `u★` itself,
which is where `u★` was formed ("The stage plan and storage" in
`CODE.md`).
"""
function increment!(d, U, u★, ::Nothing)
    broadcast!(-, d, U, u★)
    return d
end

"""
    first_touch!(a, partition)

Write a scratch array once, in `init`, through the partition that the
stage arithmetic will use, so that first touch places its pages where they
will be used ("Storage" in `CODE.md`). Nothing reads the value written,
which is zero.
"""
function first_touch!(a, ::Nothing)
    fill!(a, zero(eltype(a)))
    return a
end

"""
    lincomb_copy!(dst, U, x₀, terms, partition)

`dst = x₀ + Σ c_j x_j`, and `U = dst`: the stage value `u★`, formed in
its `d_k` array, and the copy of it that the stage solve starts from
("The callback contracts" in `CODE.md`). On the broadcast path that is
`lincomb!` then `copy_state!`, two broadcasts, as in step 2; by owner it
is one pass that writes both ("Stage arithmetic" in `CODE.md`).
"""
function lincomb_copy!(dst, U, x₀, terms::Tuple, ::Nothing)
    lincomb!(dst, x₀, terms, nothing)
    copy_state!(U, dst, nothing)
    return dst
end

"""
    copy_initial(u0, partition)

`integ.u`, a copy of `u0`, when `init` is not asked to alias it:
`copy(u0)` on the broadcast path; by owner, `similar(u0)` written through
the partition, so that first touch places the state's pages as it places
the scratch ("Storage" in `CODE.md`).
"""
copy_initial(u0, ::Nothing) = copy(u0)

# The by-owner path ("Stage arithmetic" in `CODE.md`), for a CPU `Array`
# state. Every element is computed by the same `LinComb` as the broadcast,
# alone and with its terms in the same order, so the two paths give the
# same bits at every thread count (a test). Thread `c` of the default pool
# runs the ranges it owns, as a sticky task placed on that thread, the way
# TreeAMR's `threaded_chunks` places chunk `c` (branch
# `claude/festive-bun-656842`): without entering a threaded region, so
# that a partitioned `step!` nests inside the caller's own parallel code.

"""
    OwnerPartition

The by-owner partition of a CPU `Array` state of length `n`, as `init`
builds it from its `partition` keyword: `ranges[c]` holds the index
ranges that thread `c` of the default pool owns, disjoint and together
covering `1:n` exactly (checked when it is built, by
[`owner_partition`](@ref)). `hook` is `nothing`, or, in the tests, a
function `hook(c, j, r)` that the worker calls before its `j`-th range
`r`, to record where it runs. See `CODE.md`, "Stage arithmetic".
"""
struct OwnerPartition{H}
    ranges::Vector{Vector{UnitRange{Int}}}
    n::Int
    hook::H
end

call_hook(::Nothing, c, j, r) = nothing
call_hook(hook, c, j, r) = (hook(c, j, r); nothing)

# Thread `c`'s ranges, in their order, on the calling task.
@inline function run_owned(f::F, part::OwnerPartition, c::Int) where {F}
    ranges = part.ranges[c]
    for j in eachindex(ranges)
        r = ranges[j]
        call_hook(part.hook, c, j, r)
        f(r)
    end
    return nothing
end

"""
    by_owner(f, part::OwnerPartition)

Call `f(r)` for every range `r` of the partition, each on the thread that
owns it, and return when all have returned. At one thread this is a plain
loop on the calling task, with no task and no allocation. Otherwise the
ranges of thread `c` run in one sticky task placed on default-pool thread
`c`: the default pool's thread ids follow the interactive pool's, and
`jl_set_task_tid` takes a 0-based id, so its argument is
`threadpoolsize(:interactive) + c − 1` (checked on Julia 1.10 and 1.13, a
test). The tasks are fresh per call. This waits for every one before
rethrowing the first error, unwrapped to what the loop body threw.
"""
function by_owner(f::F, part::OwnerPartition) where {F}
    nt = length(part.ranges)
    if nt == 1
        run_owned(f, part, 1)
        return nothing
    end
    offset = Threads.threadpoolsize(:interactive)
    tasks = Vector{Task}(undef, nt)
    for c in 1:nt
        task = Task(() -> run_owned(f, part, c))
        task.sticky = true
        placed = ccall(:jl_set_task_tid, Cint, (Any, Cint), task, offset + c - 1)
        placed == 1 || error("internal error: cannot place a task on thread $(offset + c)")
        tasks[c] = task
        schedule(task)
    end
    failure = nothing
    for task in tasks
        try
            wait(task)
        catch err
            failure === nothing && (failure = err)
        end
    end
    failure === nothing || throw(innermost_error(failure))
    return nothing
end

# `wait` reports what a task threw as a `TaskFailedException` (inside a
# `CompositeException` when several fail); the caller should see the
# error itself.
function innermost_error(err)
    err isa CompositeException && !isempty(err.exceptions) &&
        return innermost_error(first(err.exceptions))
    err isa TaskFailedException && return innermost_error(err.task.exception)
    return err
end

# The owner loops index with `@inbounds`. That is safe because every range
# lies in `1:part.n` (checked when the partition was built) and every array
# has that length (checked here, once per combination: a caller may not
# resize `integ.u`, and this makes a resize an error rather than an
# out-of-bounds write).
#
# The error is raised in a function of its own: inline, its message made
# the check allocate 32 bytes per call on Julia 1.10 (measured in step 5).
@inline function check_lengths(part::OwnerPartition, arrays...)
    n = part.n
    all(a -> length(a) == n, arrays) || length_mismatch(n, arrays)
    return nothing
end

@noinline function length_mismatch(n, arrays)
    m = length(arrays[findfirst(a -> length(a) != n, arrays)])
    throw(DimensionMismatch("the stage arithmetic's partition covers $n entries, but an \
                             array it combines has $m; integ.u must not be resized"))
end

@inline getindices(xs::Tuple, i) = map(x -> @inbounds(x[i]), xs)

# The loop bodies call the broadcast's own kernel, `LinComb`, per element,
# and `-` for the increment: no `@fastmath`, no `muladd`, and `@simd` has
# no reduction to reassociate. The bitwise tests check that.
#
# `ivdep` promises no loop-carried dependence through memory. That holds by
# the contract of `lincomb!` and `increment!`: `dst` may be `x₀` (or `d`
# may be `u★`) itself, which element `i` reads before it writes, and no
# array otherwise overlaps another; the plan's arrays are distinct (a
# test). Without it, LLVM's runtime alias check sees `dst === x₀` in the
# update and in every increment and falls back to a scalar loop: an
# in-place combination of 7 terms took 1.8 ms on 1.25 million entries at
# one thread, and 1.1 ms with it (measured in step 5).
function lincomb!(dst::Array, x₀::Array, terms::Tuple, part::OwnerPartition)
    xs = map(last, terms)
    check_lengths(part, dst, x₀, xs...)
    l = LinComb(map(first, terms))
    by_owner(part) do r
        @inbounds @simd ivdep for i in r
            dst[i] = l(x₀[i], getindices(xs, i)...)
        end
    end
    return dst
end

function lincomb_copy!(dst::Array, U::Array, x₀::Array, terms::Tuple,
                       part::OwnerPartition)
    xs = map(last, terms)
    check_lengths(part, dst, U, x₀, xs...)
    l = LinComb(map(first, terms))
    by_owner(part) do r
        @inbounds @simd ivdep for i in r
            v = l(x₀[i], getindices(xs, i)...)
            dst[i] = v
            U[i] = v
        end
    end
    return dst
end

function increment!(d::Array, U::Array, u★::Array, part::OwnerPartition)
    check_lengths(part, d, U, u★)
    by_owner(part) do r
        @inbounds @simd ivdep for i in r
            d[i] = U[i] - u★[i]
        end
    end
    return d
end

function first_touch!(a::Array, part::OwnerPartition)
    check_lengths(part, a)
    z = zero(eltype(a))
    by_owner(part) do r
        @inbounds @simd ivdep for i in r
            a[i] = z
        end
    end
    return a
end

function copy_initial(u0::Array, part::OwnerPartition)
    u = similar(u0)
    copy_state!(u, u0, part)
    return u
end

"""
    owner_partition(n, partition; hook = nothing)

Check the `partition` keyword of `init` against a state of length `n`,
and return it as an [`OwnerPartition`](@ref). `partition` has one element
per thread, `Threads.nthreads()` of them. Element `c` is a unit range, or
an iterable of unit ranges, of the indices that thread `c` owns (possibly
none). Together the ranges must cover `1:n` exactly once. Each failure is
an `ArgumentError` that names the thread and the index that is missing,
doubled or out of bounds. `partition = :even` is
[`even_partition`](@ref)`(n)`. `hook` is for the tests
([`OwnerPartition`](@ref)). See `CODE.md`, "Stage arithmetic".
"""
function owner_partition(n::Integer, partition; hook = nothing)
    n = Int(n)
    partition === :even && return OwnerPartition(even_partition(n), n, hook)
    partition isa Symbol &&
        throw(ArgumentError("init: partition = :$partition is not a partition; it must \
                             be nothing (broadcast), :even, or one collection of index \
                             ranges per thread"))
    nt = Threads.nthreads()
    len = applicable(length, partition) ? length(partition) : nothing
    len == nt ||
        throw(ArgumentError("init: the partition must have one element per thread, \
                             Threads.nthreads() = $nt, but it has \
                             $(len === nothing ? "no length" : len)"))
    ranges = Vector{Vector{UnitRange{Int}}}(undef, nt)
    for (c, element) in enumerate(partition)
        ranges[c] = thread_ranges(c, element)
    end
    check_cover(n, ranges)
    return OwnerPartition(ranges, n, hook)
end

function thread_ranges(c, element)
    element isa AbstractUnitRange{<:Integer} && return [UnitRange{Int}(element)]
    rs = UnitRange{Int}[]
    for r in element
        r isa AbstractUnitRange{<:Integer} ||
            throw(ArgumentError("init: element $c of the partition (thread $c) holds \
                                 $(repr(r)), which is not a unit range of indices"))
        push!(rs, UnitRange{Int}(r))
    end
    return rs
end

# The ranges, sorted by their first index, must follow one another from 1
# to `n` with neither a gap nor an overlap.
function check_cover(n, ranges)
    owned = [(r, c) for c in eachindex(ranges) for r in ranges[c] if !isempty(r)]
    sort!(owned; by = rc -> first(rc[1]))
    next = 1
    prev = (1:0, 0)
    for (r, c) in owned
        (first(r) ≥ 1 && last(r) ≤ n) ||
            throw(ArgumentError("init: thread $c's range $r of the partition is out of \
                                 bounds for a state of $n entries"))
        first(r) > next &&
            throw(ArgumentError("init: the partition misses index $next of the state \
                                 (1:$n): no thread owns it"))
        first(r) < next &&
            throw(ArgumentError("init: the partition doubles index $(first(r)): thread \
                                 $(prev[2])'s range $(prev[1]) and thread $c's range $r \
                                 both own it"))
        next = last(r) + 1
        prev = (r, c)
    end
    next == n + 1 ||
        throw(ArgumentError("init: the partition misses index $next of the state (1:$n): \
                             no thread owns it"))
    return nothing
end

"""
    even_partition(n)

`1:n` split into `Threads.nthreads()` contiguous ranges, one per thread,
by TreeAMR's `threadchunks` rule: equal lengths, the first `n mod nt`
one longer. Where `n < nt`, the last threads own nothing. This is
`partition = :even`.
"""
function even_partition(n::Int)
    nt = Threads.nthreads()
    len, extra = divrem(n, nt)
    ranges = Vector{Vector{UnitRange{Int}}}(undef, nt)
    lo = 1
    for c in 1:nt
        hi = lo + len - 1 + (c ≤ extra)
        ranges[c] = [lo:hi]
        lo = hi + 1
    end
    return ranges
end

"""
    block_partition(blocks, segments)

A partition for a state made of consecutive segments, each a sequence of
equal-sized blocks, such as a multi-set state vector (proposed in step 5,
decided 2026-09-24; it knows nothing of TreeAMR). `blocks[c]` is the
range of block numbers that thread `c` owns, one per thread, as TreeAMR's
`threadchunks(nblocks)` gives them. `segments` holds one
`(offset, blocklength)` pair per segment: block `b` of that segment is the
entries
`offset + (b − 1) blocklength .+ (1:blocklength)`. Thread `c` then owns
one range per segment, the entries of its blocks there. The result is a
`partition` for `init`, which checks it. See `CODE.md`, "Where the
partition comes from".
"""
function block_partition(blocks, segments)
    return [[(offset + (first(bs) - 1) * len + 1):(offset + last(bs) * len)
             for (offset, len) in segments] for bs in blocks]
end

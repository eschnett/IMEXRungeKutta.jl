# The stage arithmetic: every combination the integrator forms is one fused
# linear combination `dst = x₀ + Σ c_j x_j`, one pass over the state
# ("Stage arithmetic" in `CODE.md`).
#
# Each function takes a partition as its last argument. `nothing` is the
# broadcast path, the only one so far: one fused broadcast, which works for
# any array type that broadcasts, device arrays included, and never indexes
# the state. Step 5 of `PLAN.md` adds methods for an owner partition of a
# CPU `Array`; nothing else changes then.

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

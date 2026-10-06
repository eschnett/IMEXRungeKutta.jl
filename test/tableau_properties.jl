# Test-only properties of an `IMEXTableau`: the order conditions, the
# linear stability function and A-stability, the SSP coefficient of the
# explicit part, and stiff accuracy ("Tableaus" and "Testing" in
# `CODE.md`). They are not package API.
#
# Everything is computed in the tableau's own type: exactly for a rational
# tableau, and at 256 bits for a BigFloat one. BigFloat arithmetic rounds
# to the global precision, so every helper sets it.

using LinearAlgebra: I, LowerTriangular

at256(f) = setprecision(f, BigFloat, 256)

dotw(w, x) = sum(w .* x)

"""
    order_residuals(tab, p)

The residuals, as `label => lhs − rhs` pairs in the tableau's type, of the
order conditions of exactly order `p`.

- `p ≤ 3`: the additive conditions, with every combination of the two
  parts. At order 1, `bᵀ𝟙` and `b̃ᵀ𝟙`. At order 2, `wᵀx = 1/2` for
  `w ∈ {b, b̃}` and `x ∈ {c, c̃}`. At order 3, `wᵀ(x∘y) = 1/3` for
  `x, y ∈ {c, c̃}`, and `wᵀXy = 1/6` for `X ∈ {A, Ã}` and `y ∈ {c, c̃}`.
  These are all the conditions of order `p` of an additive method whose
  parts satisfy `c = A𝟙` and `c̃ = Ã𝟙`.
- `p = 4`: the four classical conditions of each part on its own, which
  are enough to show that the order stops at 3.
"""
function order_residuals(tab::IMEXTableau{R}, p::Integer) where {R}
    return at256() do
        ws = ("b" => tab.b, "b̃" => tab.b̃)
        xs = ("c" => tab.c, "c̃" => tab.c̃)
        Xs = ("A" => tab.A, "Ã" => tab.Ã)
        res = Pair{String,R}[]
        if p == 1
            for (wn, w) in ws
                push!(res, "$(wn)ᵀ𝟙" => sum(w) - 1)
            end
        elseif p == 2
            for (wn, w) in ws, (xn, x) in xs
                push!(res, "$(wn)ᵀ$(xn)" => dotw(w, x) - R(1 // 2))
            end
        elseif p == 3
            for (wn, w) in ws, (xn, x) in xs, (yn, y) in xs
                push!(res, "$(wn)ᵀ($(xn)∘$(yn))" => dotw(w, x .* y) - R(1 // 3))
            end
            for (wn, w) in ws, (Xn, X) in Xs, (yn, y) in xs
                push!(res, "$(wn)ᵀ$(Xn)$(yn)" => dotw(w, X * y) - R(1 // 6))
            end
        elseif p == 4
            for (wn, w, Xn, X, xn, x) in (("b", tab.b, "A", tab.A, "c", tab.c),
                                          ("b̃", tab.b̃, "Ã", tab.Ã, "c̃", tab.c̃))
                push!(res, "$(wn)ᵀ$(xn)³" => dotw(w, x .^ 3) - R(1 // 4))
                push!(res, "$(wn)ᵀ($(xn)∘$(Xn)$(xn))" => dotw(w, x .* (X * x)) - R(1 // 8))
                push!(res, "$(wn)ᵀ$(Xn)$(xn)²" => dotw(w, X * x .^ 2) - R(1 // 12))
                push!(res, "$(wn)ᵀ$(Xn)²$(xn)" => dotw(w, X * (X * x)) - R(1 // 24))
            end
        else
            throw(ArgumentError("order_residuals: order $p is not implemented"))
        end
        return res
    end
end

max_residual(tab, p) = at256(() -> maximum(abs ∘ last, order_residuals(tab, p)))

# Polynomials in `z` as ascending coefficient vectors.
function polyadd(p::Vector{R}, q::Vector{R}) where {R}
    r = zeros(R, max(length(p), length(q)))
    r[eachindex(p)] .+= p
    r[eachindex(q)] .+= q
    return r
end
function polymul(p::Vector{R}, q::Vector{R}) where {R}
    r = zeros(R, length(p) + length(q) - 1)
    for i in eachindex(p), j in eachindex(q)
        r[i + j - 1] += p[i] * q[j]
    end
    return r
end
polyval(p, z) = foldr((a, acc) -> a + z * acc, p; init = zero(z))
degree(p) = something(findlast(!iszero, p), 1) - 1

"""
    stability_polynomials(tab)

`(P, Q)` with `R(z) = P(z)/Q(z)` the linear stability function of the
implicit part, `R(z) = 1 + z bᵀ(I − zA)⁻¹𝟙`, and
`Q(z) = det(I − zA) = ∏ₖ(1 − z a_kk)`. By forward substitution on the
lower triangular `I − zA`, with `x_k = N_k/∏_{j≤k}(1 − z a_jj)`, so that
`N_k` and `P` are polynomials. Exact for a rational tableau.
"""
function stability_polynomials(tab::IMEXTableau{R}) where {R}
    return at256() do
        A, b = tab.A, tab.b
        s = length(b)
        z = [zero(R), one(R)]
        D(i, k) = foldl(polymul, ([one(R), -A[m, m]] for m in i:k); init = [one(R)])
        N = Vector{Vector{R}}(undef, s)
        for k in 1:s
            Nk = D(1, k - 1)
            for j in 1:(k - 1)
                iszero(A[k, j]) && continue
                Nk = polyadd(Nk, polymul(polymul(z, A[k, j] .* N[j]), D(j + 1, k - 1)))
            end
            N[k] = Nk
        end
        Q = D(1, s)
        P = Q
        for k in 1:s
            iszero(b[k]) && continue
            P = polyadd(P, polymul(polymul(z, b[k] .* N[k]), D(k + 1, s)))
        end
        return P, Q
    end
end

"""
    stability_function(tab, z)

`R(z) = 1 + z bᵀ(I − zA)⁻¹𝟙` directly, by a triangular solve, as the
check of `stability_polynomials`.
"""
function stability_function(tab::IMEXTableau, z)
    return at256() do
        s = length(tab.b)
        x = LowerTriangular(I - z .* tab.A) \ ones(typeof(z), s)
        return 1 + z * dotw(tab.b, x)
    end
end

"""
    R_infinity(tab)

`R(∞) = lim_{z→∞} P(z)/Q(z)`, or `Inf` if the implicit part is unbounded
there. For a nonsingular `A` it is `1 − bᵀA⁻¹𝟙`. At a trivial first stage
`Q` has degree `s − 1`, and `P` must too.
"""
function R_infinity(tab::IMEXTableau)
    P, Q = stability_polynomials(tab)
    dq = degree(Q)
    return at256() do
        any(!iszero, P[(dq + 2):end]) && return oftype(P[1] / Q[1], Inf)
        return (dq + 1 ≤ length(P) ? P[dq + 1] : zero(P[1])) / Q[dq + 1]
    end
end

# `p(iy)·p(−iy) = |p(iy)|²` for real coefficients, as a polynomial in
# `w = y²`: the coefficient of `y^k`, for even `k`, is
# `(−1)^(k/2) Σ_{m+n=k} (−1)^n p_m p_n`, and odd `k` cancel.
function abs2_on_imaginary_axis(p::Vector{R}) where {R}
    d = length(p) - 1
    e = zeros(R, d + 1)
    for m in 0:d, n in 0:d
        k = m + n
        iseven(k) || continue
        e[k ÷ 2 + 1] += (-1)^(k ÷ 2 + n) * p[m + 1] * p[n + 1]
    end
    return e
end

"""
    E_polynomial(tab)

The E-polynomial `E(y) = |Q(iy)|² − |P(iy)|²` of the implicit part, as
ascending coefficients in `w = y²`. With every pole of `R` in the right
half-plane, the implicit part is A-stable if and only if `E(y) ≥ 0` for
all real `y`, that is `E ≥ 0` on `w ≥ 0`.
"""
function E_polynomial(tab::IMEXTableau)
    P, Q = stability_polynomials(tab)
    return at256() do
        eP = abs2_on_imaginary_axis(P)
        eQ = abs2_on_imaginary_axis(Q)
        return polyadd(eQ, -eP)
    end
end

# Roundoff in a 256-bit coefficient that is zero in exact arithmetic is
# about 1e-77. A negative coefficient larger than this is real.
const ROUNDOFF = 1e-60

"""
    is_A_stable(tab)

Whether the implicit part is A-stable: every pole `1/a_kk` of `R` is in
the right half-plane (`a_kk > 0` wherever `a_kk ≠ 0`), and `E(w) ≥ 0` for
`w ≥ 0`. The latter is shown by the coefficients of `E` in `w` all being
non-negative, which is sufficient. If one is negative the helper does not
decide, and throws, so that a new tableau cannot pass by default.
"""
function is_A_stable(tab::IMEXTableau)
    poles_right = all(a -> iszero(a) || a > 0, (tab.A[k, k] for k in eachindex(tab.b)))
    poles_right || return false
    e = E_polynomial(tab)
    all(≥(-ROUNDOFF), e) && return true
    error("is_A_stable($(tab.name)): E(w) has a negative coefficient; \
           decide non-negativity on w ≥ 0 by its roots")
end

"""
    ssp_coefficient(tab; tol = 1e-10)

The SSP coefficient (radius of absolute monotonicity) of the explicit
part, by Kraaijevanger's conditions on `K = [Ã 0; b̃ᵀ 0]`: the largest `r`
with `K(I + rK)⁻¹ ≥ 0` and `(I + rK)⁻¹𝟙 ≥ 0` elementwise. The feasible
`r` form an interval `[0, C]`, found by bisection to `tol`. It is 0 if
`r = 0` is not feasible, that is if `Ã` or `b̃` has a negative entry.
`I + rK` is unit lower triangular, so it is always invertible.
"""
function ssp_coefficient(tab::IMEXTableau; tol = 1e-10)
    return at256() do
        s = length(tab.b̃)
        m = s + 1
        K = zeros(BigFloat, m, m)
        K[1:s, 1:s] .= tab.Ã
        K[m, 1:s] .= tab.b̃
        Id = Matrix{BigFloat}(I, m, m)
        function feasible(r)
            X = LowerTriangular(Id + r .* K) \ Id
            return all(≥(-ROUNDOFF), K * X) && all(≥(-ROUNDOFF), X * ones(BigFloat, m))
        end
        feasible(zero(BigFloat)) || return zero(BigFloat)
        lo, hi = zero(BigFloat), one(BigFloat)
        while feasible(hi)
            lo, hi = hi, 2hi
            hi < 1024 || error("ssp_coefficient($(tab.name)): unbounded")
        end
        while hi - lo > tol
            mid = (lo + hi) / 2
            feasible(mid) ? (lo = mid) : (hi = mid)
        end
        return lo
    end
end

# Rooted trees, for the classical order conditions of any order. A tree is
# its index into `ROOTED_TREES`, and is stored as the sorted indices of its
# children's trees; the leaf `τ` is index 1, with no children. Every tree
# of order `n` is a multiset of trees whose orders add up to `n − 1`, so the
# list is built order by order, and each multiset is generated once, as a
# nondecreasing index sequence.
const ROOTED_TREES = Vector{Int}[Int[]]
const TREE_ORDER = Int[1]
const TREES_UP_TO = Ref(1)

function extend_trees!(p::Integer)
    while TREES_UP_TO[] < p
        n = TREES_UP_TO[] + 1
        known = length(ROOTED_TREES)
        function forests!(acc, prefix, remaining, lo)
            remaining == 0 && return push!(acc, copy(prefix))
            for i in lo:known
                TREE_ORDER[i] ≤ remaining || continue
                push!(prefix, i)
                forests!(acc, prefix, remaining - TREE_ORDER[i], i)
                pop!(prefix)
            end
            return acc
        end
        for children in forests!(Vector{Int}[], Int[], n - 1, 1)
            push!(ROOTED_TREES, children)
            push!(TREE_ORDER, n)
        end
        TREES_UP_TO[] = n
    end
    return nothing
end

"""
    rooted_trees(p)

The indices of the rooted trees of order exactly `p`: 1, 1, 2, 4, 9, 20,
48, 115 and 286 of them for `p = 1, …, 9`.
"""
rooted_trees(p::Integer) = (extend_trees!(p); findall(==(p), TREE_ORDER))

# The density `γ(t) = |t| ∏ γ(child)`.
tree_density(i) = TREE_ORDER[i] * prod(tree_density, ROOTED_TREES[i]; init = 1)

const SUPERSCRIPTS = Dict(zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
superscript(m) = m == 1 ? "" : map(d -> SUPERSCRIPTS[d], string(m))

# Butcher's bracket notation, with repeated children as powers: `τ`,
# `[τ]`, `[τ²]`, `[[τ]]`, `[τ³]`, `[τ[τ]]`, ….
function tree_label(i)
    children = ROOTED_TREES[i]
    isempty(children) && return "τ"
    parts = String[]
    for j in unique(children)
        push!(parts, tree_label(j) * superscript(count(==(j), children)))
    end
    return "[" * join(parts) * "]"
end

"""
    classical_order_residuals(w, X, p)

The residuals, as `label => wᵀΦ(t) − 1/γ(t)` pairs, of the classical
order conditions of exactly order `p` of the Runge–Kutta method with
weights `w` and coefficients `X`, one per rooted tree `t` of order `p`,
labelled in bracket notation (`rooted_trees`). `Φ(τ) = 𝟙`, and for a tree
with children `t₁, …, t_m`, `Φ(t) = (XΦ(t₁)) ∘ ⋯ ∘ (XΦ(t_m))`; so the
abscissae are `X𝟙`. All conditions of every order, which a hand-written
list cannot give past order 4 or 5 (37 up to order 6, 200 up to order 8).
"""
function classical_order_residuals(w::AbstractVector{R}, X::AbstractMatrix{R},
                                   p::Integer) where {R}
    return at256() do
        extend_trees!(p)
        Φ = Dict{Int,Vector{R}}()
        function phi(i)
            get!(Φ, i) do
                v = ones(R, length(w))
                for j in ROOTED_TREES[i]
                    v = v .* (X * phi(j))
                end
                return v
            end
        end
        return Pair{String,R}[tree_label(i) => dotw(w, phi(i)) - R(1 // tree_density(i))
                              for i in rooted_trees(p)]
    end
end

"""
    explicit_order_residuals(tab, p)

The classical order conditions of exactly order `p` of the explicit part
`(Ã, b̃)` alone ([`classical_order_residuals`](@ref)), for a purely
explicit tableau, whose implicit part is zero and so meets none of
`order_residuals`' conditions on `b`.
"""
explicit_order_residuals(tab::IMEXTableau, p::Integer) =
    classical_order_residuals(tab.b̃, tab.Ã, p)

"""
    implicit_order_residuals(tab, p)

The classical order conditions of exactly order `p` of the implicit part
`(A, b)` alone, for a purely implicit tableau, whose explicit part is zero.
"""
implicit_order_residuals(tab::IMEXTableau, p::Integer) =
    classical_order_residuals(tab.b, tab.A, p)

"""
    stiffly_accurate(tab)

`(implicit, explicit)`: whether `b` equals the last row of `A`, and
whether `b̃` equals the last row of `Ã`, exactly.
"""
stiffly_accurate(tab::IMEXTableau) = (tab.b == tab.A[end, :], tab.b̃ == tab.Ã[end, :])

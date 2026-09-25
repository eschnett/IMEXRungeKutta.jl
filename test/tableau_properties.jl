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

"""
    explicit_order_residuals(tab, p)

The residuals, as `label => lhs − rhs` pairs in the tableau's type, of the
classical order conditions of exactly order `p` of the explicit part
`(Ã, b̃)` alone, for a purely explicit tableau, whose implicit part is
zero and so meets none of `order_residuals`' conditions on `b`. For
`p ≤ 4` they are all the conditions (1, 1, 2 and 4 of them); for `p = 5`
only the bushy-tree one, `b̃ᵀc̃⁴ = 1/5`, which is enough to show that
classical RK4 stops at 4.
"""
function explicit_order_residuals(tab::IMEXTableau{R}, p::Integer) where {R}
    return at256() do
        b, X, c = tab.b̃, tab.Ã, tab.c̃
        conditions = if p == 1
            ("b̃ᵀ𝟙" => (sum(b), 1),)
        elseif p == 2
            ("b̃ᵀc̃" => (dotw(b, c), 1 // 2),)
        elseif p == 3
            ("b̃ᵀc̃²" => (dotw(b, c .^ 2), 1 // 3), "b̃ᵀÃc̃" => (dotw(b, X * c), 1 // 6))
        elseif p == 4
            ("b̃ᵀc̃³" => (dotw(b, c .^ 3), 1 // 4),
             "b̃ᵀ(c̃∘Ãc̃)" => (dotw(b, c .* (X * c)), 1 // 8),
             "b̃ᵀÃc̃²" => (dotw(b, X * c .^ 2), 1 // 12),
             "b̃ᵀÃ²c̃" => (dotw(b, X * (X * c)), 1 // 24))
        elseif p == 5
            ("b̃ᵀc̃⁴" => (dotw(b, c .^ 4), 1 // 5),)
        else
            throw(ArgumentError("explicit_order_residuals: order $p is not implemented"))
        end
        return Pair{String,R}[label => lhs - R(rhs) for (label, (lhs, rhs)) in conditions]
    end
end

"""
    stiffly_accurate(tab)

`(implicit, explicit)`: whether `b` equals the last row of `A`, and
whether `b̃` equals the last row of `Ã`, exactly.
"""
stiffly_accurate(tab::IMEXTableau) = (tab.b == tab.A[end, :], tab.b̃ == tab.Ã[end, :])

# `IMEXTableau`, its checks, its nonzero patterns, and the conversion of its
# coefficients to the arithmetic type ("Tableaus", "Tableaus are values"
# and "The stage plan and storage" in `CODE.md`).

# Every closed form, every abscissa and every coefficient quotient is
# computed at this precision, inside `with_coefficient_precision`. BigFloat
# arithmetic rounds to the *global* precision, whatever the precision of
# its operands, so nothing here may rely on it ("What `coefficients`
# holds" in `CODE.md`).
const COEFFICIENT_PRECISION = 256

with_coefficient_precision(f) = setprecision(f, BigFloat, COEFFICIENT_PRECISION)

"""
    IMEXTableau(name, Ã, b̃, A, b)
    IMEXTableau{R}(name, Ã, b̃, A, b)

The pair of Butcher tableaus of an additive implicit–explicit Runge–Kutta
method: `(Ã, b̃)` explicit, strictly lower triangular, and `(A, b)`
implicit, lower triangular (DIRK). The abscissae `c̃ = Ã·𝟙` and `c = A·𝟙`
are computed here, from the stored coefficients, and are fields too. They
differ in general: an explicit evaluation of stage `k` is at
`tⁿ + c̃_k Δt` and its stage solve at `tⁿ + c_k Δt`.

The coefficients are held exactly, as `R = Rational{BigInt}`, when every
one given is an integer or a rational, and otherwise as 256-bit `BigFloat`.
They are converted to the state's arithmetic type once, when an integrator
is built. The ten named tableaus ([`IMEXSSP3433`](@ref), [`RK4`](@ref)
and the others) go through this constructor, and so does a caller's own.
A purely explicit one has `A = 0` and `b = 0`.

The constructor refuses, with an `ArgumentError` that says why:
- parts that are not square and of one size `s ≥ 1`, or weights that are
  not of length `s`;
- a coefficient that is not finite;
- an `Ã` that is not strictly lower triangular, or an `A` that is not
  lower triangular;
- an inadmissible tableau: one where `a_kk = 0` but column `k` of `A`
  below the diagonal, or `b_k`, is nonzero. Such a stage would need the
  implicit tendency `g(U_k)`, and `g` is never evaluated here. This rules
  out ESDIRK-type additive schemes (KenCarp and the like).

See `CODE.md`, "Tableaus" and "Tableaus are values". The tableau's
properties (order, stability, SSP coefficient) are computed in the tests,
not here.
"""
struct IMEXTableau{R<:Union{Rational{BigInt},BigFloat}}
    name::String
    Ã::Matrix{R}
    b̃::Vector{R}
    A::Matrix{R}
    b::Vector{R}
    c̃::Vector{R}
    c::Vector{R}

    function IMEXTableau{R}(name::AbstractString, Ã::AbstractMatrix,
                            b̃::AbstractVector, A::AbstractMatrix,
                            b::AbstractVector) where {R<:Union{Rational{BigInt},BigFloat}}
        check_sizes(name, Ã, b̃, A, b)
        Ã, b̃, A, b, c̃, c = with_coefficient_precision() do
            Ã′ = to_coefficients(R, Ã)
            b̃′ = to_coefficients(R, b̃)
            A′ = to_coefficients(R, A)
            b′ = to_coefficients(R, b)
            # The row sums of the stored coefficients, in `R`: exact for
            # rationals, and one rounding at 256 bits otherwise.
            return Ã′, b̃′, A′, b′, row_sums(Ã′), row_sums(A′)
        end
        check_finite(name, Ã, b̃, A, b)
        check_triangular(name, Ã, A)
        check_admissible(name, A, b)
        return new{R}(String(name), Ã, b̃, A, b, c̃, c)
    end
end

function IMEXTableau(name::AbstractString, Ã::AbstractMatrix, b̃::AbstractVector,
                     A::AbstractMatrix, b::AbstractVector)
    exact = all(x -> x isa Union{Integer,Rational}, Iterators.flatten((Ã, b̃, A, b)))
    R = exact ? Rational{BigInt} : BigFloat
    return IMEXTableau{R}(name, Ã, b̃, A, b)
end

function to_coefficients(::Type{R}, x::AbstractArray) where {R}
    y = similar(Array{R}, size(x))
    for (i, xi) in zip(eachindex(y), x)
        y[i] = R(xi)
    end
    return y
end

function row_sums(M::Matrix{R}) where {R}
    s = size(M, 1)
    c = Vector{R}(undef, s)
    for k in 1:s
        acc = zero(R)
        for j in 1:s
            acc += M[k, j]
        end
        c[k] = acc
    end
    return c
end

function check_sizes(name, Ã, b̃, A, b)
    s = size(Ã, 1)
    s ≥ 1 || throw(ArgumentError("IMEXTableau $name: the explicit part Ã has no \
                                  stages; a tableau needs at least one"))
    size(Ã) == (s, s) || throw(ArgumentError("IMEXTableau $name: Ã must be square, \
                                              but it is $(size(Ã, 1))×$(size(Ã, 2))"))
    size(A) == (s, s) || throw(ArgumentError("IMEXTableau $name: A must be square \
                                              and of the same size as Ã ($s×$s), \
                                              but it is $(size(A, 1))×$(size(A, 2)); \
                                              an additive method has one stage count"))
    length(b̃) == s || throw(ArgumentError("IMEXTableau $name: b̃ must have one weight \
                                           per stage ($s), but it has $(length(b̃))"))
    length(b) == s || throw(ArgumentError("IMEXTableau $name: b must have one weight \
                                           per stage ($s), but it has $(length(b))"))
    return nothing
end

function check_finite(name, parts...)
    for (label, x) in zip(("Ã", "b̃", "A", "b"), parts)
        for i in eachindex(x)
            isfinite(x[i]) || throw(ArgumentError("IMEXTableau $name: $label has the \
                                                   non-finite coefficient $(x[i]) at \
                                                   $(Tuple(CartesianIndices(x)[i]))"))
        end
    end
    return nothing
end

function check_triangular(name, Ã, A)
    s = size(Ã, 1)
    for k in 1:s, j in k:s
        iszero(Ã[k, j]) || throw(ArgumentError("IMEXTableau $name: Ã must be strictly \
                                                lower triangular, so that each explicit \
                                                stage uses only earlier stages, but \
                                                ã[$k,$j] = $(Ã[k, j]) is nonzero"))
    end
    for k in 1:s, j in (k + 1):s
        iszero(A[k, j]) || throw(ArgumentError("IMEXTableau $name: A must be lower \
                                               triangular (DIRK), so that each stage \
                                               is one stage solve, but a[$k,$j] = \
                                               $(A[k, j]) is nonzero"))
    end
    return nothing
end

function check_admissible(name, A, b)
    s = size(A, 1)
    for k in 1:s
        iszero(A[k, k]) || continue
        why = "stage $k has a[$k,$k] = 0, so it makes no stage solve and has no \
               implicit tendency (g is never evaluated), yet"
        for j in (k + 1):s
            iszero(A[j, k]) || throw(ArgumentError("IMEXTableau $name is not \
                                                   admissible: $why a[$j,$k] = \
                                                   $(A[j, k]) would read one. ESDIRK-type \
                                                   tableaus (KenCarp and the like) \
                                                   need g(uⁿ) there; see \
                                                   \"Admissibility\" in CODE.md"))
        end
        iszero(b[k]) || throw(ArgumentError("IMEXTableau $name is not admissible: \
                                             $why b[$k] = $(b[k]) would read one. See \
                                             \"Admissibility\" in CODE.md"))
    end
    return nothing
end

function Base.show(io::IO, tab::IMEXTableau{R}) where {R}
    print(io, "IMEXTableau{", R, "}(\"", tab.name, "\", ", nstages(tab), " stages)")
    return nothing
end

"""
    nstages(tab::IMEXTableau)

The stage count `s`, including a trivial first stage.
"""
nstages(tab::IMEXTableau) = length(tab.b)

"""
    solves(tab::IMEXTableau)

Per stage, whether it makes a stage solve: `a_kk ≠ 0`.
"""
solves(tab::IMEXTableau) = [!iszero(tab.A[k, k]) for k in 1:nstages(tab)]

"""
    explicit_used(tab::IMEXTableau)

Per stage `k`, whether its explicit tendency is read: whether column `k`
of `Ã` or `b̃_k` is nonzero. Only these stages call `f_exp!` and the stage
limiter, and only these have a tendency array ("One step" in `CODE.md`).
"""
function explicit_used(tab::IMEXTableau)
    s = nstages(tab)
    return [any(!iszero, @view tab.Ã[:, k]) || !iszero(tab.b̃[k]) for k in 1:s]
end

"""
    implicit_used(tab::IMEXTableau)

Per stage `k`, whether its implicit increment `d_k = U − u★` is read:
whether column `k` of `A` below the diagonal or `b_k` is nonzero. Only
these stages have an increment array ("One step" in `CODE.md`). By
admissibility, every implicit-used stage solves.
"""
function implicit_used(tab::IMEXTableau)
    s = nstages(tab)
    return [any(!iszero, @view tab.A[(k + 1):s, k]) || !iszero(tab.b[k]) for k in 1:s]
end

"""
    row_empty(tab::IMEXTableau, k)

Whether stage `k`'s row is empty in both parts, `ã_kj = a_kj = 0` for all
`j < k`, so that its `u★` is `uⁿ`. Taken from the exact coefficients.
"""
function row_empty(tab::IMEXTableau, k::Integer)
    return all(iszero, @view tab.Ã[k, 1:(k - 1)]) && all(iszero, @view tab.A[k, 1:(k - 1)])
end

"""
    scratch_count(tab::IMEXTableau)

The number of state-sized scratch arrays the stage plan needs, besides
`integ.u`: `U` if [`needs_U`](@ref), one per implicit-used stage, one per
explicit-used stage, and one more if some solving stage that is not
implicit-used has a nonempty row. `u★` is otherwise formed in the array that then holds `d_k`,
and a stage with an empty row has `u★ = uⁿ`, which is `integ.u` itself
("The stage plan and storage" in `CODE.md`; the empty-row case amended in
step 2).
"""
function scratch_count(tab::IMEXTableau)
    sol = solves(tab)
    imp = implicit_used(tab)
    ex = explicit_used(tab)
    extra = any(k -> sol[k] && !imp[k] && !row_empty(tab, k), 1:nstages(tab)) ? 1 : 0
    return (needs_U(tab) ? 1 : 0) + count(imp) + count(ex) + extra
end

"""
    needs_U(tab::IMEXTableau)

Whether some stage forms its stage value in the scratch `U`: a solving
stage, or an explicit-used stage with a nonempty row. Every IMEX tableau
here has one; explicit Euler, whose one stage reads `uⁿ` itself, has none
(`U` added only where used, amended with the explicit tableaus).
"""
function needs_U(tab::IMEXTableau)
    sol = solves(tab)
    ex = explicit_used(tab)
    return any(k -> sol[k] || (ex[k] && !row_empty(tab, k)), 1:nstages(tab))
end

"""
    coefficients(T, Tt, tab::IMEXTableau)

The tableau's coefficients converted to the arithmetic type `T` and its
abscissae to the time type `Tt`, as the stage plan reads them ("One step"
and "Increments, not tendencies" in `CODE.md`). Returns a named tuple:

- `Ã`, `b̃`: the explicit coefficients;
- `γ`: the diagonal `a_kk`, which the stage solve receives as `γ = a_kk`
  (it is passed `a_kk Δt`);
- `Ā`: `a_kj/a_jj` for `j < k` and `a_jj ≠ 0`, which multiplies the
  increment `d_j` in `u★` of stage `k`, and zero elsewhere;
- `b̄`: `b_j/a_jj` for `a_jj ≠ 0`, which multiplies `d_j` in the update,
  and zero elsewhere;
- `c̃`, `c`: the explicit and implicit abscissae.

Each value, the quotients included, is computed in the tableau's own
type, exactly for a rational tableau and at 256 bits otherwise, then
rounded to 256 bits and converted to `T` (or `Tt`) once. So each is the
correctly rounded 256-bit value. For `T = BigFloat` the result has 256
bits, whatever the global precision.

A zero here is either a structural zero of the tableau, which the stage
plan never reads, or a coefficient that underflowed in `T`.
"""
function coefficients(::Type{T}, ::Type{Tt}, tab::IMEXTableau) where {T,Tt}
    s = nstages(tab)
    return with_coefficient_precision() do
        conv(::Type{S}, x) where {S} = S(BigFloat(x))
        Ã = [conv(T, tab.Ã[k, j]) for k in 1:s, j in 1:s]
        b̃ = [conv(T, tab.b̃[k]) for k in 1:s]
        γ = [conv(T, tab.A[k, k]) for k in 1:s]
        Ā = [j < k && !iszero(tab.A[j, j]) ? conv(T, tab.A[k, j] / tab.A[j, j]) :
             zero(T) for k in 1:s, j in 1:s]
        b̄ = [!iszero(tab.A[j, j]) ? conv(T, tab.b[j] / tab.A[j, j]) : zero(T)
             for j in 1:s]
        c̃ = [conv(Tt, tab.c̃[k]) for k in 1:s]
        c = [conv(Tt, tab.c[k]) for k in 1:s]
        return (; Ã, b̃, γ, Ā, b̄, c̃, c)
    end
end

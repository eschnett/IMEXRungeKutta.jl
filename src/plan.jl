# The stage plan: the tableau compiled, once, in `init`, for the arithmetic
# type `T`, the time type and `Δt` ("One step" and "The stage plan and
# storage" in `CODE.md`).
#
# The plan is a tuple of `Stage`s and a tuple of update terms. Their types
# record the tableau's nonzero pattern: which stages solve, which are
# explicit-used and implicit-used, and which arrays each combination reads.
# Building it is type-unstable, once, behind the function barrier of
# `init`; running it (`step!`) is type-stable and unrolled over the stages.
#
# The pattern is taken from the exact tableau, never from which converted
# coefficients happen to be zero in `T`. A coefficient that underflows in
# `T` keeps its term, multiplying an array that exists and has been
# written, so it neither drops a term from the plan nor reads an
# unallocated array. A structural zero, by contrast, has no term and, if
# nothing else reads its stage, no array: no coefficient ever multiplies
# scratch that was not written in this step, so `0·NaN` cannot occur.

"""
    Stage{Solves,ExplicitUsed,ImplicitUsed}

One stage `k` of the plan:

- `terms`: the `(coefficient, array)` pairs of `u★ = uⁿ + Σ …`, only the
  nonzero ones: first `(Δt ã_kj, k̃_j)` by increasing `j`, then
  `(a_kj/a_jj, d_j)` by increasing `j`. `uⁿ` itself is not a term; it is
  the `x₀` of the combination;
- `ustar`: where `u★` is formed when the stage solves: the stage's own
  increment array `d_k` if it is implicit-used, else the one extra array.
  If the row is empty, `u★ = uⁿ`, and `ustar` is `integ.u` itself;
- `U`: the stage value that `f_exp!` reads. It is the scratch `U`, or
  `integ.u` itself at a trivial stage (no solve, empty row);
- `d`, `k̃`: the increment and tendency arrays, or `nothing` where the
  tableau does not read them;
- `γΔt`: `a_kk Δt`, in `T`, which the stage solve receives;
- `c̃`, `c`: the two abscissae, in the time type: `f_exp!` and the stage
  limiter are called at `tⁿ + c̃ Δt`, and `solve_imp!` at `tⁿ + c Δt`.
"""
struct Stage{Solves,ExplicitUsed,ImplicitUsed,Terms<:Tuple,S,UA,DA,KA,T,Tt}
    k::Int
    terms::Terms
    ustar::S
    U::UA
    d::DA
    k̃::KA
    γΔt::T
    c̃::Tt
    c::Tt
end

function Stage{S,E,I}(k, terms, ustar, U, d, k̃, γΔt, c̃, c) where {S,E,I}
    return Stage{S,E,I,typeof(terms),typeof(ustar),typeof(U),typeof(d),typeof(k̃),
                 typeof(γΔt),typeof(c̃)}(k, terms, ustar, U, d, k̃, γΔt, c̃, c)
end

solves(::Stage{S}) where {S} = S
explicit_used(::Stage{S,E}) where {S,E} = E
implicit_used(::Stage{S,E,I}) where {S,E,I} = I

# A trivial stage makes no solve and has an empty row, so its stage value is
# `uⁿ` itself: `f_exp!` gets `integ.u`, with no copy and no stage limiter
# ("A trivial first stage is uⁿ" in `CODE.md`).
trivial(st::Stage) = !solves(st) && isempty(st.terms)

"""
    StagePlan

The compiled tableau: the tuple of [`Stage`](@ref)s, the tuple of update
terms `(Δt b̃_j, k̃_j)` and `(b_j/a_jj, d_j)` (in that order, each by
increasing `j`, only the nonzero ones), the scratch arrays, and the
partition the stage arithmetic runs through.
"""
struct StagePlan{Stages<:Tuple,Update<:Tuple,Scratch<:Tuple,P}
    stages::Stages
    update::Update
    scratch::Scratch
    partition::P
end

"""
    build_plan(tab, u, u0, T, Δt, partition)

Compile `tab` into a [`StagePlan`](@ref) for the state `u` (which is
`integ.u`), the arithmetic type `T` and the step `Δt`, whose type is the
time type. The scratch arrays come from `similar(u0)`, and each is written
once, through `partition` (`first_touch!`). There are exactly
`scratch_count(tab)` of them.
"""
function build_plan(tab::IMEXTableau, u, u0, ::Type{T}, Δt::Tt, partition) where {T,Tt}
    s = nstages(tab)
    sol = solves(tab)
    ex = explicit_used(tab)
    imp = implicit_used(tab)
    co = coefficients(T, Tt, tab)
    ΔtT = T(Δt)

    scratch = Any[]
    function allocate()
        a = first_touch!(similar(u0), partition)
        push!(scratch, a)
        return a
    end
    U = allocate()
    D = Any[imp[k] ? allocate() : nothing for k in 1:s]
    K = Any[ex[k] ? allocate() : nothing for k in 1:s]

    # A term reads an array that the pattern says exists; if it did not,
    # the pattern functions and this plan would disagree.
    function array_of(arrays, j, what)
        a = arrays[j]
        a === nothing && error("internal error in $(tab.name): the $what of stage $j is \
                                read but not allocated")
        return a
    end
    rows = map(1:s) do k
        terms = Any[]
        for j in 1:(k - 1)
            iszero(tab.Ã[k, j]) || push!(terms, (ΔtT * co.Ã[k, j], array_of(K, j, "k̃")))
        end
        for j in 1:(k - 1)
            iszero(tab.A[k, j]) || push!(terms, (co.Ā[k, j], array_of(D, j, "d")))
        end
        return Tuple(terms)
    end
    @assert all(k -> isempty(rows[k]) == row_empty(tab, k), 1:s)
    E = any(k -> sol[k] && !imp[k] && !isempty(rows[k]), 1:s) ? allocate() : nothing

    stages = map(1:s) do k
        terms = rows[k]
        if sol[k]
            ustar = isempty(terms) ? u : imp[k] ? D[k] : E
            Uk = U
        else
            ustar = nothing
            Uk = !ex[k] ? nothing : isempty(terms) ? u : U
        end
        return Stage{sol[k],ex[k],imp[k]}(k, terms, ustar, Uk, D[k], K[k], ΔtT * co.γ[k],
                                          co.c̃[k], co.c[k])
    end

    update = Any[]
    for j in 1:s
        iszero(tab.b̃[j]) || push!(update, (ΔtT * co.b̃[j], array_of(K, j, "k̃")))
    end
    for j in 1:s
        iszero(tab.b[j]) || push!(update, (co.b̄[j], array_of(D, j, "d")))
    end

    length(scratch) == scratch_count(tab) ||
        error("internal error in $(tab.name): the plan allocated $(length(scratch)) \
               scratch arrays, but scratch_count says $(scratch_count(tab))")
    return StagePlan(Tuple(stages), Tuple(update), Tuple(scratch), partition)
end

"""
    plan_calls(plan::StagePlan)

The callback calls one step of `plan` makes, counted from the plan:
`f_exp` (the explicit-used stages), `solve_imp` (the solving stages) and
`stage_limiter` (the explicit-used stages that are not trivial). For the
tests, which compare it with hand-counted constants and with a mock's log.
"""
function plan_calls(plan::StagePlan)
    st = plan.stages
    return (f_exp = count(explicit_used, st), solve_imp = count(solves, st),
            stage_limiter = count(s -> explicit_used(s) && !trivial(s), st))
end

# Test-only helpers shared by the validation files of step 3: the fitted
# order, and the Kaps problem, which `stiff_tests.jl` and `ap_tests.jl`
# both use.

"""
    fitted_order(dts, errs)

The observed order: the least-squares slope of `log errs` against
`log dts`.
"""
function fitted_order(dts, errs)
    x = collect(log.(dts))
    y = collect(log.(errs))
    x̄ = sum(x) / length(x)
    ȳ = sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

# The Kaps problem, `p = ε`:
#
#     y₁′ = −(2 + 1/ε) y₁ + y₂²/ε,    y₂′ = y₁ − y₂ − y₂²,
#
# split as `PLAN.md` says: the implicit part is `g = ((y₂² − y₁)/ε, 0)` and
# the explicit part `f = (−2y₁, y₁ − y₂ − y₂²)`. The stage solve is exact:
# `U₂ = u★₂`, since `g₂ = 0`, and then `U₁ = (ε u★₁ + γΔt U₂²)/(ε + γΔt)`.
# From `y(0) = (1, 1)` the exact solution is `y₁ = e^{−2t}`, `y₂ = e^{−t}`
# for every `ε`, on the manifold `y₁ = y₂²`. The two-component state is
# indexed here; that is the test problem's, not the integrator's.
kaps_exp!(du, u, ε, t) = (du[1] = -2u[1]; du[2] = u[1] - u[2] - u[2]^2; nothing)
function kaps_imp!(U, u★, γΔt, ε, t)
    U[2] = u★[2]
    U[1] = (ε * u★[1] + γΔt * U[2]^2) / (ε + γΔt)
    return nothing
end

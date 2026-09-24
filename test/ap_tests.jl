using IMEXRungeKutta: IMEXProblem, IMEXTableau
using IMEXRungeKutta: IMEXSSP222, IMEXSSP2322, IMEXSSP2332, IMEXSSP3332, IMEXSSP3433
using IMEXRungeKutta: ARS222, ARS443
using LinearAlgebra: LowerTriangular

# Asymptotic preservation: where one step lands in the stiff limit
# ("Testing", Asymptotic preservation, and "Where a step ends in the stiff
# limit" in `CODE.md`).
#
# Let `S` be the solving stages (`a_kk ≠ 0`; all of them for the IMEX-SSP
# schemes, stages 2–s for ARS). If the stage solves put every `U_k`,
# `k ∈ S`, on the equilibrium `ū`, and `b_Sᵀ A_SS⁻¹ 𝟙 = 1` (that is
# `R(∞) = 0`), the update is
#
#     uⁿ⁺¹ = ū + Δt wᵀF,    w = b̃ − Ã_{S,:}ᵀ A_SS⁻ᵀ b_S,
#
# with `F_j` the explicit tendency of stage `j`. For an `f` of `t` alone,
# `F_j = f(tⁿ + c̃_j Δt)`, so the displacement is `Δt (wᵀ𝟙) f` to leading
# order, then `Δt² (wᵀc̃) f′`. Here `wᵀ𝟙 = 1 − bᵀA⁻¹c̃`.
# - `w = 0` for the ARS schemes: both parts are stiffly accurate.
# - `wᵀ𝟙 = 0` but `wᵀc̃ ≠ 0` for SSP2(3,2,2) and SSP2(3,3,2): the
#   implicit part is stiffly accurate, the explicit part is not.
# - `wᵀ𝟙 ≠ 0` for SSP2(2,2,2), SSP3(3,3,2) and SSP3(4,3,3).

# The stiff-limit weights `w`, in 256-bit arithmetic, and `b_SᵀA_SS⁻¹𝟙`.
function stiff_limit_weights(tab)
    return setprecision(BigFloat, 256) do
        S = [k for k in eachindex(tab.b) if !iszero(tab.A[k, k])]
        A = BigFloat.(tab.A[S, S])
        v = LowerTriangular(A)' \ BigFloat.(tab.b[S])    # A_SS⁻ᵀ b_S
        w = BigFloat.(tab.b̃) .- BigFloat.(tab.Ã[S, :])' * v
        return w, sum(v)
    end
end

# The limit displacement `Δt wᵀF` for `f = f(t)`, one step from `tn`.
function limit_displacement(tab, f, tn, Δt)
    w, _ = stiff_limit_weights(tab)
    return setprecision(BigFloat, 256) do
        F = [f(BigFloat(tn) + BigFloat(tab.c̃[j]) * BigFloat(Δt)) for j in eachindex(w)]
        return Float64(BigFloat(Δt) * sum(w .* F))
    end
end

const AP_ū = 0.5
ap_relax_imp!(U, u★, γΔt, ε, t) = (U .= (ε .* u★ .+ γΔt .* AP_ū) ./ (ε + γΔt); nothing)
ap_const!(du, u, ε, t) = (du .= 1; nothing)
ap_cos!(du, u, ε, t) = (du .= cos(t); nothing)

function ap_displacement(f!, tab, t0, t1, dt, ε; u0 = AP_ū)
    integ = solve(IMEXProblem(f!, ap_relax_imp!, [u0], (t0, t1), ε), tab; dt)
    return integ.u[1] - AP_ū
end

const ARS443_UPSTREAM = let t = ARS443()
    IMEXTableau("ARS(4,4,3), upstream's b̃ = b", t.Ã, t.b, t.A, t.b)
end

# Per tableau, `wᵀ𝟙` and, where it is exact, `wᵀc̃`, as recorded in
# `CODE.md`: `wᵀ𝟙` is −1/√2 for SSP2(2,2,2) and SSP3(3,3,2), and
# −0.28436465 for SSP3(4,3,3) (256-bit values rounded to 8 digits).
const AP_TABLE = [
    (make = IMEXSSP222, w𝟙 = -0.70710678, wc̃ = nothing),
    (make = IMEXSSP2322, w𝟙 = 0, wc̃ = 1 // 2),
    (make = IMEXSSP2332, w𝟙 = 0, wc̃ = 1 // 4),
    (make = IMEXSSP3332, w𝟙 = -0.70710678, wc̃ = nothing),
    (make = IMEXSSP3433, w𝟙 = -0.28436465, wc̃ = nothing),
    (make = ARS222, w𝟙 = 0, wc̃ = 0),
    (make = ARS443, w𝟙 = 0, wc̃ = 0),
]

const AP_ε = 1e-12

# The derivation above rests on `R(∞) = 0` over the solving stages and on
# the weights. A tableau change that broke either would make every claim
# below about a different formula.
@testset "The stiff-limit weights: b_SᵀA_SS⁻¹𝟙 = 1, and wᵀ𝟙 = 1 − bᵀA⁻¹c̃ as recorded" begin
    @test length(AP_TABLE) == 7
    for spec in AP_TABLE
        tab = spec.make()
        w, r = stiff_limit_weights(tab)
        @test abs(r - 1) < 1e-70
        w𝟙 = sum(w)
        if iszero(spec.w𝟙)
            @test abs(w𝟙) < 1e-70
        else
            @test abs(w𝟙 - spec.w𝟙) < 1e-8
        end
        if spec.wc̃ !== nothing
            @test abs(sum(w .* tab.c̃) - spec.wc̃) < 1e-70
        end
        if tab.name in ("ARS(2,2,2)", "ARS(4,4,3)")
            @test all(x -> abs(x) < 1e-70, w)
        end
    end
    # SSP3(4,3,3)'s coefficient, as TreeGRRMHD quotes it.
    @test round(Float64(sum(stiff_limit_weights(IMEXSSP3433())[1])); digits = 4) == -0.2844
end

# The claim of "Testing" in `CODE.md`, that one step lands on the
# equilibrium to O(ε), holds only where `wᵀ𝟙 = 0`. For a constant `f`,
# the three others land exactly `Δt (1 − bᵀA⁻¹c̃) f` off it. A change that
# broke the stage contract (a limiter folded into the tendency, an
# increment scaled wrongly) would move the step off this formula.
@testset "Stiff relaxation, constant f: O(ε) where wᵀ𝟙 = 0, else Δt (1 − bᵀA⁻¹c̃) f" begin
    for spec in AP_TABLE, dt in (0.1, 0.01)
        tab = spec.make()
        d = ap_displacement(ap_const!, tab, 0.0, dt, dt, AP_ε)
        w𝟙 = Float64(sum(stiff_limit_weights(tab)[1]))
        if iszero(spec.w𝟙)
            # On `ū + ε f`, the quasi-steady state, to within 2ε.
            @test abs(d) ≤ 3 * AP_ε
        else
            @test abs(d - dt * w𝟙) ≤ 10 * AP_ε
            @test abs(d / dt - spec.w𝟙) < 1e-8
        end
    end
end

# With an `f` that depends on `t`, SSP2(3,2,2) and SSP2(3,3,2), whose
# explicit part is not stiffly accurate, land `Δt² (wᵀc̃) f′` off the
# equilibrium, not O(ε) (measured in step 3). Only the ARS schemes land on
# it to O(ε). Every tableau lands on the limit formula to O(ε).
@testset "Stiff relaxation, f = cos t: every step lands on ū + Δt wᵀF, to O(ε)" begin
    for spec in AP_TABLE, dt in (0.1, 0.05, 0.025)
        tab = spec.make()
        d = ap_displacement(ap_cos!, tab, 1.0, 1.0 + dt, dt, AP_ε)
        @test abs(d - limit_displacement(tab, cos, 1.0, dt)) ≤ 10 * AP_ε
        if spec.wc̃ !== nothing && iszero(spec.wc̃)
            @test abs(d) ≤ 3 * AP_ε
        elseif spec.wc̃ !== nothing
            # The leading term, `Δt² (wᵀc̃) f′(1)`, to 2% at Δt = 0.025.
            dt == 0.025 && @test abs(d / (dt^2 * -sin(1.0)) / spec.wc̃ - 1) < 0.02
        else
            dt == 0.025 && @test abs(d / (dt * cos(1.0)) / spec.w𝟙 - 1) < 0.05
        end
    end
end

# `CODE.md` claims the displacement does not accumulate, because every
# step starts by relaxing again. If the update carried `uⁿ` forward (an
# `R(∞) ≠ 0`, or a stage that skipped its solve), it would grow with the
# step count.
@testset "Over 100 steps the displacement is the last step's, not accumulated" begin
    for spec in AP_TABLE
        tab = spec.make()
        # ε = 1e−12: exactly the last step's limit displacement.
        d = ap_displacement(ap_cos!, tab, 0.0, 1.0, 0.01, AP_ε)
        @test abs(d - limit_displacement(tab, cos, 0.99, 0.01)) ≤ 10 * AP_ε
        # ε = 1e−6, the README's problem: the last step's displacement,
        # from the quasi-steady state, to 0.2%.
        ε = 1e-6
        d = ap_displacement(ap_cos!, tab, 0.0, 1.0, 0.01, ε)
        one = ap_displacement(ap_cos!, tab, 0.99, 1.0, 0.01, ε; u0 = AP_ū + ε * cos(0.99))
        @test abs(d / one - 1) < 2e-3
    end
end

# The Kaps problem (`problems.jl`) at ε = 1e−12, from `y(0) = (1, 1)`,
# which is on the manifold `y₁ = y₂²`. That manifold is invariant under
# the explicit part, `f₁ − 2y₂f₂ = 0` on it, so the `O(Δt)` term above
# vanishes here, and the three tableaus with `wᵀ𝟙 ≠ 0` land `O(Δt²)` or
# `O(Δt³)` off it, as do SSP2(3,2,2) and SSP2(3,3,2). The ARS schemes land
# on it to O(ε). The residuals and orders are measured in step 3.
kaps_residual(tab, dt; ε = AP_ε, y0 = (1.0, 1.0), t0 = 0.0) =
    let u = solve(IMEXProblem(kaps_exp!, kaps_imp!, collect(y0), (t0, t0 + dt), ε), tab;
                  dt).u
        u[1] - u[2]^2
    end

const KAPS_AP_DTS = (0.1, 0.05, 0.025, 0.0125)
const KAPS_AP_TABLE = [
    (make = IMEXSSP222, r = 1.7046e-2, order = 2),
    (make = IMEXSSP2322, r = 9.9750e-3, order = 2),
    (make = IMEXSSP2332, r = 4.7483e-3, order = 2),
    (make = IMEXSSP3332, r = 5.5907e-4, order = 3),
    (make = IMEXSSP3433, r = -7.1610e-3, order = 2),
    (make = ARS222, r = 0.0, order = nothing),
    (make = ARS443, r = 0.0, order = nothing),
    # Upstream's ARS(4,4,3), with `b̃ = b`: its explicit part is no longer
    # stiffly accurate, and it lands O(Δt⁴) off the manifold.
    (make = () -> ARS443_UPSTREAM, r = 6.4568e-5, order = 4),
]

# A tableau or stage-contract change that moved a step off the manifold,
# or that put a stiffly accurate scheme off it by more than O(ε), would be
# a change in what the stiff limit gives TreeGRRMHD.
@testset "Kaps at ε = 1e−12: one step lands on y₁ = y₂² to O(ε) for ARS, else as recorded" begin
    for spec in KAPS_AP_TABLE
        tab = spec.make()
        rs = [kaps_residual(tab, dt) for dt in KAPS_AP_DTS]
        if spec.order === nothing
            # Measured at most 0.094ε, at Δt = 0.1.
            @test all(r -> abs(r) ≤ AP_ε, rs)
        else
            @test abs(rs[1] / spec.r - 1) < 0.01
            @test abs(fitted_order(KAPS_AP_DTS, abs.(rs)) - spec.order) < 0.15
        end
    end
end

# As for the relaxation above: a residual that grew with the step count
# would be a step that carried the previous one's error forward.
@testset "Kaps at ε = 1e−12: the residual after many steps is the last step's" begin
    for spec in KAPS_AP_TABLE[1:7], dt in (0.1, 0.01)
        tab = spec.make()
        integ = solve(IMEXProblem(kaps_exp!, kaps_imp!, [1.0, 1.0], (0.0, 1.0), AP_ε), tab;
                      dt)
        r = integ.u[1] - integ.u[2]^2
        t = 1 - dt
        one = kaps_residual(tab, dt; y0 = (exp(-2t), exp(-t)), t0 = t)
        if spec.order === nothing
            @test abs(r) ≤ AP_ε
        else
            # Measured within 0.32% of it.
            @test abs(r / one - 1) < 0.01
        end
    end
end

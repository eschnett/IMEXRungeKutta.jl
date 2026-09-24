# IMEXRungeKutta design

This is the design document: what the package is, what has been decided,
and why. Where the implementation shows it wrong or incomplete, amend it
and say so ("(amended in step N)", "(measured in step N)"). Each design
item is marked **(decided)**, **(proposed)** or **(open)**. A decision
that a step proposed and Erik then took keeps its history, as "(proposed
in step N, decided 2026-09-24)".

**Status (2026-09-24):** the implementation plan is complete, and
`PLAN.md` is deleted (amended in step 6). Its steps 0–6 built the
[package design](#package-design): the tableaus, the integrator on the
broadcast path, the validation
([Validation](#validation-measured-in-step-3)), the Metal smoke run
([On a device](#on-a-device-measured-in-step-4)), the stage arithmetic by
owner ([By owner, as built](#by-owner-as-built-measured-in-step-5)) and a
review pass. Erik decided the steps' proposals on 2026-09-24, all but one.
Open or pending:
- where the partition for TreeAMR state vectors comes from (open;
  [Stage arithmetic](#stage-arithmetic-decided));
- SSP2(3,3,2)'s coefficients against Pareschi & Russo (2005) (open;
  [Open questions](#open-questions));
- the Symmetry run of `bench/symmetry_stage_arithmetic.sh`, which is
  Erik's, and the one decision still proposed, fresh tasks rather than
  persistent workers, which waits on it
  ([By owner, as built](#by-owner-as-built-measured-in-step-5));
- the v0.1.0 tag, which is Erik's;
- SciML/OrdinaryDiffEq.jl#4620 upstream, until whose fix two oracle
  comparisons are `@test_broken` ([The oracle](#the-oracle)).

## Purpose

Fixed-step **additive implicit–explicit Runge–Kutta** (IMEX RK) time
integration for method-of-lines PDE systems

    u′ = f(u, t) + g(u, t)

where `f` is non-stiff and treated explicitly, and `g` is stiff and
treated implicitly. The distinguishing feature: **the user solves the
implicit stage equation**, by whatever means suits the problem. The
integrator never evaluates `g` and never forms a Jacobian.

The target problems are hyperbolic systems with stiff **local**
relaxation, where `g` couples only the variables within one grid cell:
- resistive MHD, where the stiff term is the Ohm's-law current and the
  stage solve is coupled to the recovery of primitive variables;
- radiation or neutrino transport with stiff emission and absorption;
- chemistry and reaction networks;
- the relaxation systems of Pareschi & Russo (2005).

For such problems the stage equation is a small nonlinear system per cell.
The best solver for it is problem-specific: a closed form, a fixed point
with a physics fallback, or Newton on a reduced set of unknowns. It often
cannot be written as "evaluate `g`, then Newton on the whole vector"
without losing robustness. A generic Newton–Krylov loop over the whole
state is exactly what this package avoids.

The first intended user is TreeGRRMHD.jl (resistive GRMHD on an adaptive
mesh). Nothing here depends on it.

## Requirements (decided)

These are fixed by the use case, and were decided before the package
existed:

- **A stand-alone package**, not a contribution to OrdinaryDiffEq or an
  extension of ClimaTimeSteppers (see
  [Why not an existing package](#why-not-an-existing-package)).
- **The user stage solver.** Per implicit stage, the integrator calls
  `solve_imp!(U, u★, γΔt, p, t)`. That call must return `U` satisfying
  `U = u★ + γΔt g(U, t)`, computed by any means. The integrator recovers
  the implicit tendency as `k = (U − u★)/(γΔt)`. So no separate
  evaluation of `g` exists or is needed.
- **The tableaus**, seven of them: SSP2(2,2,2), SSP2(3,2,2) and
  SSP2(3,3,2), SSP3(3,3,2) and SSP3(4,3,3) (Pareschi & Russo 2005), and
  ARS(2,2,2) and ARS(4,4,3) (Ascher, Ruuth & Spiteri 1997). SSP3(4,3,3)
  is the intended production scheme, and SSP2(2,2,2) the debugging one.
  - This listed SSP2(3,3,2) alone, as the scheme named `IMEXSSP2322`
    (amended in step 1: Erik decided to have both).
  - `IMEXSSP2322` is SSP2(3,2,2), by the SSPk(s,σ,p) naming rule in
    [Tableaus are values](#tableaus-are-values-decided) (decided). Both
    upstreams implement it under that name (OrdinaryDiffEqSDIRK's
    `IMEXSSP2322`, ClimaTimeSteppers' `SSP322`).
  - `IMEXSSP2332` is SSP2(3,3,2) (decided). It is in neither upstream.
- **A stage limiter hook** and a step limiter hook. These are
  positivity- or atmosphere-type resets of the state, as in the SSPRK
  methods of OrdinaryDiffEqSSPRK.
- **Fixed `Δt`.** The caller chooses the step, typically from a CFL
  condition, and restarts a fresh integrator after a regrid.
- **Generic arrays.** Stage arithmetic is by broadcasting over
  `similar(u0)` arrays, so device arrays work. The package must not
  require a particular array type. A faster path for a CPU `Array` is
  allowed alongside (amended 2026-09-24, see
  [Stage arithmetic](#stage-arithmetic-decided)); it is
  `partition`, since step 5. On
  Metal, a `Float32` `MtlArray` state runs with scalar indexing
  disallowed and agrees with the CPU bitwise (measured in step 4; [On a
  device](#on-a-device-measured-in-step-4)).
- **Julia 1.10 floor**, generic in the scalar type `T` (Float32 must
  work).
- **Minimal dependencies.** Only CommonSolve at run time (amended
  2026-09-24; this was "at most StaticArrays", with SciMLBase open). See
  [Dependencies and names](#dependencies-and-names-decided). It brings
  PrecompileTools and Preferences with it (measured in step 0). Heavier
  packages (OrdinaryDiffEqSDIRK) are test-only, in `test/Project.toml`
  (amended in step 6; [File layout](#file-layout-decided)). With it, the
  test environment has 142 packages on Julia 1.13 and 138 on 1.10,
  standard libraries included (measured in step 3).

## The method

### Tableaus

An IMEX RK method is a pair of Butcher tableaus: `(Ã, b̃, c̃)` explicit,
strictly lower triangular; `(A, b, c)` implicit, lower triangular (DIRK).
The abscissae are the row sums, `c̃ = Ã·𝟙` and `c = A·𝟙`, and **they
differ**. For SSP3(4,3,3), `c̃ = (0, 0, 1, 1/2)` and `c = (α, 0, 1, 1/2)`.
The explicit evaluation of stage `k` is at `tⁿ + c̃_k Δt`, and its stage
solve is at `tⁿ + c_k Δt`.

Pareschi–Russo's name SSPk(s,σ,p) means:
- an explicit part that is SSP of order `k`;
- `s` implicit and `σ` explicit stages;
- overall order `p`.

Their implicit parts have `a_kk ≠ 0` on every stage. The ARS schemes have
`a_11 = 0` (a trivial first implicit stage) and a zero first column in
`A`.

**Admissibility (decided).** Because `g` is never evaluated, a stage
with `a_kk = 0` has no implicit tendency. That is consistent only if
column `k` of `A` and `b_k` are zero. The IMEX-SSP and ARS schemes satisfy
this. ESDIRK-type additive schemes (KenCarp and the like) do not, since
their first column needs `g(uⁿ)`. The tableau constructor checks it and
throws an `ArgumentError` that says why.

**Exact coefficients.** Only SSP2(3,2,2), SSP2(3,3,2) and ARS(4,4,3)
are rational (amended in step 1).
SSP2(2,2,2), SSP3(3,3,2) and ARS(2,2,2) involve `√2`, in closed form
(`γ = 1 − 1/√2`, and so on). SSP3(4,3,3)'s `α, β, η` are printed to 14
digits only (α = 0.24169426078821, β = 0.06042356519705,
η = 0.12915286960590). Both OrdinaryDiffEqSDIRK and ClimaTimeSteppers use
exactly those 14 digits.

**SSP3(4,3,3) in closed form** (computed 2026-09-24, symbolically). With
the explicit part and `b = b̃ = (0, 1/6, 1/6, 2/3)` fixed:
- The third-order implicit condition `bᵀAc = 1/6` and the coupling
  condition `bᵀAc̃ = 1/6` are the only order-3 conditions that involve
  `α, β, η`. Together they give `β = α/4` and `η = (1 − 2α)/4`. Every
  other condition up to order 3 holds identically.
- L-stability, `R(∞) = 1 − bᵀA⁻¹𝟙 = 0`, then reduces to
  `(2α − 1)(3α² − 9α + 2) = 0`. The root in `(0, 1/2)` is
  **`α = (9 − √57)/6`** = 0.24169426078820838…
- The printed digits are these values rounded to 14 digits. Upstream's
  truncated coefficients leave order-condition residuals of about 2e−15.
  Measured in step 1, with the printed digits held as exact decimals, the
  largest is 2.07e−15, in `bᵀAc` and `b̃ᵀAc`. The only other misses are
  1.67e−15, in `bᵀAc̃` and `b̃ᵀAc̃`. Truncation also leaves
  `R(∞) = 3.1e−13`, not 0.

So every tableau here has a closed form. **(decided):**
- the rational ones are held exactly, as `Rational{BigInt}`;
- the others are held as 256-bit `BigFloat`, from their closed forms;
- both are converted to `T` once, when the integrator is built.

**Properties to compute and record per tableau (decided):**
- the order conditions up to order 3, including the IMEX coupling
  conditions;
- stiff accuracy (`b` equal to the last row of `A`);
- L-stability of the implicit part;
- the SSP coefficient of the explicit part.

SSP3(4,3,3) is **not** stiffly accurate: `b = (0, 1/6, 1/6, 2/3)`, while
the last row of `A` is `(β, η, 1/2 − β − η − α, α)`. So its order may
drop in the stiff limit, and the ARS schemes, which are stiffly accurate,
are the alternative if that matters. It does drop, **from 3 to 2**, in
the stiff component (measured in step 3): on the Kaps problem at
`ε = 10⁻⁶` and `10⁻⁹` the observed order is 1.989 and 1.988 in `y₁`,
while the non-stiff `y₂` keeps 3.010 and 3.011. ARS(4,4,3) keeps 3.017
and 3.012 ([The stiff limit](#the-stiff-limit)). Its parameters are
exactly those
that make `R(∞) = 0` (above). Its implicit part is also A-stable, and so
L-stable. That is computed, not quoted (measured in step 1): all three
nonzero coefficients of its E-polynomial are positive (below).
Everything else is computed, not quoted.

**Where a step ends in the stiff limit** (measured in step 2). For a
relaxation `g = −(u − ū)/ε` with `ε → 0`, each stage solve puts its `U`
on the equilibrium, but the update of a tableau whose implicit part is
not stiffly accurate need not. With every `a_kk ≠ 0` and `R(∞) = 0`, the
update is `uⁿ⁺¹ = ū + Δt (b̃ᵀ − bᵀA⁻¹Ã) F`, where `F` is the vector of
the stages' explicit tendencies. For an `f` that depends on `t` alone,
this is `Δt (1 − bᵀA⁻¹c̃) f` to leading order.
- `1 − bᵀA⁻¹c̃` is −0.2844 for SSP3(4,3,3), and −0.7071 for SSP2(2,2,2)
  and SSP3(3,3,2).
- It is 0 for the four schemes with a stiffly accurate implicit part:
  SSP2(3,2,2), SSP2(3,3,2) and the two ARS (for ARS, on stages 2–s).
- On the README's problem, `u′ = cos t − (u − ū)/ε` with `ε = 10⁻⁶` and
  `Δt = 0.01`, SSP3(4,3,3) ends each step `−0.286 Δt cos t` off the
  quasi-steady state `ū + ε cos t`. That is a displacement of 1.5e−3,
  2858 times `ε cos t` and of the other sign. ARS(4,4,3) ends on it to
  within 0.4% of `ε cos t`.
- The displacement does not accumulate, since every step starts by
  relaxing again.

Step 3 measured this per tableau, and found the second item incomplete
(amended in step 3; [Asymptotic preservation](#asymptotic-preservation)):
- **The general form.** With `S` the solving stages (all of them for
  IMEX-SSP, stages 2–s for ARS), `b_SᵀA_SS⁻¹𝟙 = 1` for all seven, and the
  update is `ū + Δt wᵀF` with `w = b̃ − Ã_{S,:}ᵀA_SS⁻ᵀb_S`. For an `f` of
  `t` alone that is `Δt (wᵀ𝟙) f + Δt² (wᵀc̃) f′ + …`, and
  `wᵀ𝟙 = 1 − bᵀA⁻¹c̃`. Every step of every tableau lands on it, to at most
  5.83ε at `ε = 10⁻¹²`, whatever `Δt`.
- **`wᵀ𝟙 = 0` is not `w = 0`.** For SSP2(3,2,2) and SSP2(3,3,2) the
  implicit part is stiffly accurate but the explicit part is not, so
  `w = b̃ − ã_s ≠ 0`, with `wᵀc̃ = 1/2` and `1/4`. With an `f` that depends
  on `t`, they land `Δt² (wᵀc̃) f′` off the equilibrium, not `O(ε)`: from
  `t = 1` with `f = cos t` and `Δt = 0.025`, `−2.65e−4` and `−1.33e−4`,
  that is 0.504 and 0.252 times `−Δt² sin 1`. With a constant `f` they
  land on `ū + ε f` to within 2ε.
- **Only the ARS schemes have `w = 0`**, both parts being stiffly
  accurate, and land on the equilibrium to `O(ε)` whatever `f` is.
- **SSP3(4,3,3)'s coefficient is `wᵀ𝟙` = −0.28436465** (256 bits,
  rounded), exactly what a constant `f` gives, `d/Δt = −0.28436465`; for
  SSP2(2,2,2) and SSP3(3,3,2) it is `−1/√2`.
- **No accumulation, measured.** After 100 steps of `Δt = 0.01`, the
  displacement at `t = 1` is the last step's limit displacement to 5.1ε
  at `ε = 10⁻¹²`, for every tableau. At the README's `ε = 10⁻⁶` it is the
  displacement of one step from the quasi-steady state at `t = 0.99` to
  within 6.8e−4 relative.

**Measured properties** (measured in step 1). These are computed by
`test/tableau_properties.jl` and asserted by `test/tableau_tests.jl`:
- *order*: every additive order condition up to that order holds, and
  some condition of the next order misses by the amount given;
- *stiffly accurate*: `b` is the last row of `A` (implicit), and `b̃` is
  the last row of `Ã` (explicit);
- *A-stable*: every pole `1/a_kk` of `R(z) = 1 + z bᵀ(I − zA)⁻¹𝟙` is in
  the right half-plane, and `E(y) = |Q(iy)|² − |P(iy)|² ≥ 0`;
- *SSP*: the SSP coefficient of the explicit part, by bisection on
  Kraaijevanger's conditions for `K = [Ã 0; b̃ᵀ 0]`, to 1e−10;
- the patterns, as the stages `k` where they hold. The scratch count is
  that of [The stage plan and storage](#the-stage-plan-and-storage-decided).

| | SSP2(2,2,2) | SSP2(3,2,2) | SSP2(3,3,2) | SSP3(3,3,2) | SSP3(4,3,3) | ARS(2,2,2) | ARS(4,4,3) |
|---|---|---|---|---|---|---|---|
| held as | BigFloat | rational | rational | BigFloat | BigFloat | BigFloat | rational |
| order | 2 | 2 | 2 | 2 | 3 | 2 | 3 |
| worst residual up to it | 0 | 0 (exact) | 0 (exact) | 0 | 0 | 8.6e−78 | 0 (exact) |
| next order misses by | 0.167 | 0.167 | 0.083 | 0.069 | 0.083 | 0.187 | 0.097 |
| stiffly accurate, implicit / explicit | no / no | yes / no | yes / no | no / no | no / no | yes / yes | yes / yes |
| `R(∞)` | 1.0e−76 | 0 (exact) | 0 (exact) | 8.6e−77 | −2.4e−76 | 0 | 0 (exact) |
| A-stable | yes | yes | yes | yes | yes | yes | yes |
| so L-stable | yes | yes | yes | yes | yes | yes | yes |
| SSP coefficient | 1 | 1 | 2 | 1 | 1 | 0 | 0 |
| solves (`a_kk ≠ 0`) | 1, 2 | 1–3 | 1–3 | 1–3 | 1–4 | 2, 3 | 2–5 |
| explicit-used | 1, 2 | 2, 3 | 1–3 | 1–3 | 2–4 | 1, 2 | 1–4 |
| implicit-used | 1, 2 | 1–3 | 1–3 | 1–3 | 1–4 | 2, 3 | 2–5 |
| scratch arrays | 5 | 6 | 7 | 7 | 8 | 5 | 9 |

The SSP2(3,3,2) column was added in step 1, after Erik's decision to
have both SSP2 schemes (amended in step 1).

The residuals and `R(∞)` of the BigFloat tableaus are 256-bit
round-off. The next-order miss is at order 3 for the second-order
schemes. For the third-order ones it is in a classical order-4
condition of one part.

The E-polynomials, in `y`. Every coefficient is non-negative, which is
sufficient for `E ≥ 0`, and the diagonals are positive. The test helper
refuses to decide a tableau with a negative coefficient, rather than pass
it (proposed in step 1, decided 2026-09-24):
- SSP2(2,2,2) and ARS(2,2,2): `γ⁴y⁴` (= 0.00736 y⁴), with
  `γ = 1 − 1/√2`. Both implicit parts have the same `R`.
- SSP2(3,2,2): `y⁴/8 + y⁶/64`.
- SSP2(3,3,2): `y⁴/144 + y⁶/2304`.
- SSP3(3,3,2): `0.00736 y⁴ + 0.000631 y⁶`.
- SSP3(4,3,3): `0.00545 y⁴ + 0.000282 y⁶ + α⁸y⁸` (α⁸ = 1.16e−5).
- ARS(4,4,3): `y⁴/24 + 5y⁶/144 + y⁸/256`.

In each, the `y²` coefficient vanishes, as order 2 requires: exactly
for the rational tableaus, and to 256-bit round-off (about 1e−77) for
the others.

The explicit parts of the IMEX-SSP schemes are Heun's method and the
three-stage SSPRK(3,3) of Shu & Osher, with SSP coefficient 1, and for
SSP2(3,3,2) the three-stage second-order SSPRK(3,2), with coefficient 2.
Each is the optimal value for its stages and order. The ARS explicit parts have negative coefficients, and so
coefficient 0: `δ = −1/√2` in ARS(2,2,2), and `ã₄₂ = −5/6` and
`ã₅₄ = b̃₄ = −7/4` in ARS(4,4,3).
Step 3 measured what that means for TVD advection
([SSP and total variation](#ssp-and-total-variation)): on linear upwind
advection only the stability polynomial matters, so ARS(2,2,2) keeps the
threshold 1 of every two-stage second-order method, while ARS(4,4,3),
whose polynomial has the `z⁴` coefficient `−7/288`, has none.

**What the order conditions do not see** (measured in step 1). The
order conditions are the independent check of every transcription. A
test perturbs each coefficient that may be nonzero, one at a time, by
1e−3. Each perturbation breaks an order condition up to the stated
order, or `R(∞) = 0`, or is refused by the constructor, with one
exception. Two coefficients escape the order conditions alone:
- SSP3(4,3,3)'s `a₁₁ = α` is pinned only by `R(∞) = 0`.
- SSP2(3,2,2)'s `a₁₁ = 1/2` is the exception, pinned by nothing
  computed. With `b₁ = b̃₁ = 0` it enters only `Ac`, an order-3 term.
  Stiff accuracy makes `R(∞) = 0` whatever it is. It is asserted
  directly, as the common diagonal 1/2 that both upstreams have.

**Cross-checks** (measured in step 1). Each tableau was compared, by
reading only, with OrdinaryDiffEqSDIRK 2.9.6
(`src/imex_tableaus.jl`) and ClimaTimeSteppers (`main`,
`src/solvers/imex_tableaus.jl`, fetched 2026-09-24).
- The ARS schemes were also checked against the paper itself, from
  Erik's copy.
- Pareschi & Russo's paper was not at hand. Where the citations in
  `src/tableaus.jl` give a table number for it, the number is
  OrdinaryDiffEqSDIRK's, and they say so.
- SSP2(2,2,2), SSP2(3,2,2) (`SSP322` there), SSP3(3,3,2) (`SSP332`) and
  ARS(2,2,2) agree with both, coefficient for coefficient. ARS(2,2,2)
  also agrees with ARS (1997) §2.6, p. 158: `γ = (2 − √2)/2`,
  `δ = 1 − 1/(2γ)`, explicit weights `(δ, 1 − δ, 0)`.
- SSP2(3,3,2) is in neither upstream (amended in step 1).
  - Its coefficients are from the step-1 reviewer's (Claude's)
    recollection of Pareschi & Russo (2005), not a transcription.
  - They are verified only by the order conditions, `R(∞) = 0` and the
    SSP coefficient. These give order 2, `bᵀAc − 1/6 = 1/24` at order 3,
    `R(∞) = 0` and SSP coefficient 2.
  - They are not yet checked against the paper, which is not available
    here (open; see [Open questions](#open-questions)).
  - It has no oracle.
  - Step 3 adds evidence, not a confirmation (measured in step 3): the
    observed order is 2.003 on `u′ = iu − u` and 2.001 on
    `u′ = −u + cos t`; on the Kaps problem it is second order at every
    `ε`; the measured TVD threshold without relaxation is 2.0000, its SSP
    coefficient; and in the stiff limit it lands where a stiffly accurate
    implicit part must (`wᵀ𝟙 = 0`, `wᵀc̃ = 1/4`, in
    [Asymptotic preservation](#asymptotic-preservation)). These test the
    integrated scheme, where the order conditions test the coefficients;
    neither is a comparison with the paper.
- SSP3(4,3,3) (`SSP433`) agrees with both, up to their 14-digit
  `α, β, η`, which are the closed form rounded (a test).
- **ARS(4,4,3) disagrees in `b̃`.**
  - ARS (1997) §2.8, p. 160, prints the explicit tableau with last row
    `(1/4, 7/4, 3/4, −7/4, 0)` and a weight row `b̃` identical to it. It
    prints the implicit last row and `b` both as
    `(0, 3/2, −3/2, 1/2, 1/2)`.
  - So the explicit part is stiffly accurate, and it uses four stages,
    the "4 explicit stages" of the name. ClimaTimeSteppers agrees.
  - Here `b̃` is the paper's (decided: Erik checked the paper, and so did
    step 1).
  - OrdinaryDiffEqSDIRK 2.9.6's `ARS443` has
    `b̃ = b = (0, 3/2, −3/2, 1/2, 1/2)`, which is not the paper's. It is
    also third order, exactly, and misses the classical order-4
    conditions by up to 0.076 (a test). But it reads the explicit
    tendency of stage 5, so it makes five explicit evaluations per step,
    not four.
  - This is a possible upstream issue, for Erik to report. Nothing has
    been filed.
  - Step 3's oracle comparison of ARS(4,4,3) must therefore compare with
    `IMEXTableau("…", Ã, b, A, b)`, built from `ARS443()`'s parts, not
    with `ARS443()`.
  - What the difference amounts to (measured in step 3). On the oracle's
    linear problem the two differ by 1.95e−5 in one step of `Δt = 0.1`
    and by 5.2e−10 at `Δt = 0.00625`, `O(Δt⁴)` per step (local slopes
    3.59, 3.78, 3.89, 3.94), and by 2.5e−5 over ten steps of `Δt = 0.1`.
    In the stiff limit the difference is qualitative: upstream's explicit
    part is not stiffly accurate, so its step lands `O(Δt⁴)` off the
    equilibrium. On the Kaps problem at `ε = 10⁻¹²`, one step of
    `Δt = 0.1` leaves `y₁ − y₂² = 6.46e−5` with upstream's `b̃`, and
    4.2e−14 (0.04ε) with the paper's. A reviewer's reproducer, on a
    problem not recorded here, measured −2.3e−5 against 1.6e−8 at
    `Δt = 0.1`, `ε = 10⁻¹⁰`.

### One step (decided)

Call stage `k` **explicit-used** if column `k` of `Ã` or `b̃_k` is
nonzero, and **implicit-used** if column `k` of `A` below the diagonal or
`b_k` is nonzero. For stages `k = 1, …, s`:

1. `u★ = uⁿ + Δt Σ_{j<k} ã_kj k̃_j + Σ_{j<k} (a_kj/a_jj) d_j`, summing
   only over the explicit-used and implicit-used `j`.
2. If `a_kk ≠ 0`: `solve_imp!(U, u★, a_kk Δt, p, tⁿ + c_k Δt)`, then, if
   stage `k` is implicit-used, `d_k = U − u★`. Otherwise `U = u★`.
3. If stage `k` is explicit-used:
   `stage_limiter!(U, integ, p, tⁿ + c̃_k Δt)`, then
   `f_exp!(k̃_k, U, p, tⁿ + c̃_k Δt)`.

Then `uⁿ⁺¹ = uⁿ + Δt Σ_j b̃_j k̃_j + Σ_j (b_j/a_jj) d_j`, followed by
`step_limiter!(uⁿ⁺¹, integ, p, tⁿ⁺¹)`.

**Increments, not tendencies** (decided). The integrator stores
`d_k = U − u★ = a_kk Δt k_k` and folds `1/a_kk` into the coefficients,
which are computed exactly and then rounded to `T`. This is the tendency
recovery `k_k = (U − u★)/(a_kk Δt)` of the requirements, without the
division by `Δt` and the multiplication back.

**A trivial first stage is `uⁿ`** (decided). In the ARS schemes, stage 1
has `a_11 = 0` and an empty row, so `U = uⁿ`. The integrator passes `uⁿ`
itself to `f_exp!`, with no copy and no stage limiter call. `uⁿ` has
already been through the step limiter, or is the caller's initial state.

**The stage limiter acts only where `f_exp!` reads** (decided). The
limited stage value is read by `f_exp!` and by nothing else: the
increment is taken before the limiter, and the update reads only
increments and tendencies. So at a stage that is not explicit-used,
such as stage 1 of SSP3(4,3,3), a limiter call would be dead work. It is
skipped. This differs from OrdinaryDiffEq's SSPRK methods, which limit
every stage. There, every stage is read by `f`, so they limit exactly
what `f` reads too. A caller who wants the step's result limited as well
passes the same function as `step_limiter!`.

Three consequences of the stage contract, each one a decision:

- **The tendency is taken before the limiter** (decided). A limiter's
  correction is a reset, not a tendency. Folded into `k_k`, it would be
  re-weighted by `a_jk/a_kk` in later stages and by `b_k/a_kk` in the
  update. ClimaTimeSteppers does fold it in: its `constrain_state!` runs
  before its `(U − temp)/dtγ`.
- **One call per implicit stage** (decided). Convergence, fallbacks,
  flags and counters belong to the user's solver. The integrator has no
  nonlinear-solver loop, tolerance or retry.
- **Untouched components stay untouched.** Where `solve_imp!` leaves a
  component of `U` as it entered, a copy of `u★`, its increment is
  exactly zero, and the
  component evolves by the explicit part alone, to the last bit. A
  conservative explicit scheme stays conservative. This is a test.

**Round-off in the recovered increment.** `d_k = U − u★` carries an
absolute error of about `ε|U|`. It enters the update multiplied by
`b_k/a_kk`, which makes it `O(ε|U| b_k/a_kk)` per step, independent of
`Δt`. That is harmless for these tableaus. A tableau with a tiny `a_kk`
would be a poor fit.

**Cost.** The explicit right-hand side is usually the expensive call.
Skipping zero columns, SSP3(4,3,3) makes three explicit evaluations per
step, not four; OrdinaryDiffEqSDIRK makes five (below). Increments are
stored only for implicit-used stages, and explicit tendencies only for
explicit-used ones.

## Package design

Drafted and reviewed 2026-09-24. The first caller is TreeGRRMHD, whose `CODE.md`
("Time integration") and `PLAN.md` (steps 4a–4c) say what it needs:
- one integrator over TreeAMR's flat multi-set state vector, on CPU
  threads or on a device;
- a chunked driver with a fixed `Δt` per chunk and a fresh integrator
  after each regrid;
- a stage solver that flags and counts its own outcomes;
- limiters. TreeGH's are integrator-free (GH-4,
  `gh_stage_limit!(u, p, t)`), and TreeHydro's take OrdinaryDiffEq's
  `(u, integrator, p, t)`;
- stage arithmetic that keeps each block on the thread that owns it, as
  TreeAMR now does (branch `claude/festive-bun-656842`, 2026-09-23).

### Dependencies and names (decided)

The package depends on **CommonSolve.jl** only, and adds methods to its
`init`, `solve!`, `step!` and `solve`. SciMLBase and OrdinaryDiffEq
re-export these same functions, so this package can be loaded beside them
without a name clash. StaticArrays is not needed. This amends the "at
most StaticArrays" requirement.

**CommonSolve's own dependencies** (measured in step 0). This section
said CommonSolve has none. That held up to 0.2.13. From 0.2.14, the
current version on 2026-09-24, it depends on PrecompileTools, which
depends on Preferences (and the TOML standard library). Both are small
and are already in almost every Julia environment. PrecompileTools
1.3 requires Julia 1.12, so on 1.10 the resolver picks 1.2.1; the suite
passes with both. The requirement stands: one direct dependency.

**The compat bound** (proposed in step 0, decided 2026-09-24) is
`CommonSolve = "0.2.14"`, that is `[0.2.14, 0.3)`: the current 0.2 series,
from the one version the suite has run against. A cap below 0.2.14 would
avoid the two transitive packages, but it would hold every environment
that loads this package back from CommonSolve's later releases. SciMLBase
3.56 bounds CommonSolve by `0.2.4 - 0.2`, so either choice resolves beside
it.

The alternatives, not taken:
- **Own the names.** They clash with DifferentialEquations when both are
  loaded, and the caller then has to qualify them.
- **SciMLBase.** It brings problem types, solution types and callbacks,
  none of which a fixed-step chunked driver uses. Its load time is not
  small. TreeHydro depends on it through OrdinaryDiffEqSSPRK, but
  TreeGRRMHD need not.

### The interface (decided)

    prob  = IMEXProblem(f_exp!, solve_imp!, u0, (t0, t1), p = nothing)
    integ = init(prob, IMEXSSP3433(); dt,
                 stage_limiter = nothing, step_limiter = nothing,
                 partition = nothing, alias_u0 = false)
    step!(integ)                       # one step
    solve!(integ)                      # step to t1; returns integ
    integ = solve(prob, IMEXSSP3433(); dt)  # init, then solve!

- **The limiters are `init` keywords**, spelled as OrdinaryDiffEq's
  `solve` keywords, `stage_limiter` and `step_limiter`. `nothing` means
  no call. Below, the functions passed are called `stage_limiter!` and
  `step_limiter!`.
- **Public fields:** `integ.u`, `integ.t`, `integ.dt`, `integ.p`,
  `integ.nstep` (steps taken), `integ.nsteps` (steps to `t1`) and
  `integ.tableau`.
- **`p` is optional** and defaults to `nothing`, as in SciML.
- **`init` copies `u0`**, unless `alias_u0 = true`. Aliasing saves one
  state-sized array.
- **The caller may change `integ.u` in place between steps**
  (decided). Nothing
  carries over from one step to the next: no first-same-as-last stage and
  no cached tendency. So an atmosphere reset or a diagnostic fix-up in
  the driver is always safe.

What step 2 settled (proposed in step 2, decided 2026-09-24):
- **The integrator type** is `IMEXIntegrator`, a mutable struct, not
  exported. Every field but `t` and `nstep` is `const`, so `integ.u = v`
  is an error rather than a silent rebinding that the stage plan, which
  holds `integ.u`, would not see. The internal fields are `t0`, `t1`, the
  four callbacks and the plan. It prints as
  `IMEXIntegrator("SSP3(4,3,3)", t = 0.3, step 3 of 10)`.
- **`step!` returns `nothing`**, as OrdinaryDiffEq's does. `solve!`
  returns the integrator.
- **`solve` is a method of our own**,
  `solve(prob::IMEXProblem, tab::IMEXTableau; kwargs...) =
  solve!(init(prob, tab; kwargs...))`. CommonSolve 0.2.14 has the same
  thing as a generic fallback, `solve(args...; kwargs...) =
  solve!(init(args...; kwargs...))` (checked in step 2). Our own method
  does not rest on it, and carries the docstring.
- **`dt` is a required keyword**, and there is no default tableau.
- **`init` refuses**, each with an `ArgumentError` that says why:
  - a `partition` other than `nothing`, as not implemented yet (step 5).
    Since step 5 it refuses a partition for a state that is not a CPU
    `Array`, and a malformed one ([By owner, as
    built](#by-owner-as-built-measured-in-step-5); amended in step 5);
  - a state whose real element type is not an `AbstractFloat`, since the
    coefficients cannot be converted to it;
  - `t1 ≤ t0`, since the integration runs forward over a nonempty
    interval;
  - a `dt` that is not positive and finite, and a non-finite `tspan`;
  - a `dt` that is not a real number (amended in step 6: the code has
    refused it since step 2, and a test now checks it).
- **`IMEXProblem`** holds `f_exp!`, `solve_imp!`, `u0`, `tspan`, promoted
  to one type, and `p`.

### The callback contracts (decided)

- **`f_exp!(du, u, p, t)`** writes all of `du` and does not change `u`.
  At a trivial first stage, `u` is `integ.u` itself.
- **`solve_imp!(U, u★, γΔt, p, t)`** writes `U` so that
  `U = u★ + γΔt g(U, t)`.
  - `U` and `u★` are distinct arrays, and `u★` must not be changed.
    Where the stage's row is empty, `u★ = uⁿ`, and the integrator passes
    `integ.u` itself as `u★`, with no copy (amended in step 2; see
    [The stage plan and storage](#the-stage-plan-and-storage-decided)).
  - **On entry, `U` holds a copy of `u★`** (decided), so the solver
    writes only the components it solves for.
  - Its return value is ignored.
- **`stage_limiter!(u, integrator, p, t)` and
  `step_limiter!(u, integrator, p, t)`** (decided) change `u` in place.
- **`γΔt` has type `T`**, and **`t` has the time type** (see
  [Time and the step count](#time-and-the-step-count-decided)).
- **`p` is passed through untouched.** No callback may resize its
  arrays.

**Why `U` enters as a copy of `u★`.** The untouched components are then
untouched by construction, not by each caller remembering to copy them.
The copy costs one pass over the state, which a caller would otherwise
pay itself: TreeGRRMHD's Ohm solve changes only `E`, and every other
set, including TreeGH's, has to come from `u★`. The alternative, a `U`
that is undefined on entry, would save that pass only for a solver that
writes every component anyway.

**Why the limiter takes OrdinaryDiffEq's signature.** OrdinaryDiffEq's
SSPRK methods call `(u, integrator, p, t)`, and so the limiters already
written for them, such as TreeHydro's, work unchanged. What the
integrator argument promises:
- **Only the public fields are meaningful** (see
  [The interface](#the-interface-decided)).
- **During a step**, `integrator.u` is `uⁿ` and `integrator.t` is `tⁿ`,
  as in OrdinaryDiffEq. The `t` argument is the time of `u`.
- **The stage limiter's `u` is a scratch array**, never `integrator.u`,
  and `integrator.u` must not be changed.
- **The step limiter's `u` is `integrator.u`**, which then holds
  `uⁿ⁺¹`, and its `t` is `tⁿ⁺¹`. `integrator.t` and `nstep` are advanced
  after it returns.

### Failures and exceptions (decided)

**There is no status** (resolves an earlier open question). The stage
solver owns convergence, fallbacks, flags and counters, and reaches them
through `p`, as TreeGRRMHD's Ohm solve does. The integrator has nothing
to do with a status: with a fixed `Δt` and no retry, only the caller can
decide to shorten the step or stop.

**Exceptions propagate, and a step is atomic up to its update.**
`integ.u` is written only by the final update and by the step limiter.
So an exception from `f_exp!`, `solve_imp!` or the stage limiter leaves
`integ.u = uⁿ` and `integ.t = tⁿ`. The caller can then retry from `uⁿ`
with a smaller `Δt`, in a fresh integrator. After an exception from the
step limiter, `integ.u` is undefined.

### Time and the step count (decided)

- **The step count.** `init` takes `nsteps = ⌈(t1 − t0)/dt⌉` and then
  `Δt = (t1 − t0)/nsteps`, which is at most the requested `dt`. The
  ceiling has a tolerance of a few ulps, so that a chunk meant to be a
  whole number of steps is not given one extra step by round-off.
  - **The tolerance** (proposed in step 2, decided 2026-09-24). With
    `r = (t1 − t0)/dt` and `m` the integer nearest it, `nsteps = m` if
    `m ≥ 1` and `|r − m| ≤ 4 (eps(r) + (eps(t0) + eps(t1))/dt)`, and `⌈r⌉`
    otherwise. The second part is needed: `t1 − t0` inherits the rounding
    of both ends. In the test's sweep of chunks `(kT, (k + 1)T)` with
    `dt = T/m` (four `T`, `k` up to 123456, `m` up to 12), 149 of the 288
    chunks miss `m` by more than `4 eps(r)`, and every one gets exactly
    `m` steps (measured in step 2). Without any tolerance, `0.07/0.01`,
    `2.1/0.3` and `(3·0.1)/0.1` would each get one step too many.
  - So `Δt ≤ dt` holds up to that tolerance, not exactly (amended in
    step 2): in the sweep, `Δt/dt − 1` is at most 1.1e−11.
- **No accumulated time.** `tⁿ = t0 + n Δt` is computed afresh at each
  step, not accumulated, and the last step sets `t = t1` exactly. `step!`
  after the last step throws an `ArgumentError`.
- **Two types.** The time type is that of `t0`, `t1` and `dt` after
  promotion, made `float` (amended in step 2), so that an integer `tspan`
  or a rational `dt` gives `Float64` time. The arithmetic type is
  `T = real(eltype(u0))`, so a complex state works (the order tests use
  `u′ = iu − u`). Coefficients are converted to `T`, and abscissae to
  the time type. A `Float32` state with `Float64` time is allowed.

### Tableaus are values (decided)

- **`IMEXTableau{R}`** holds a name, `Ã`, `b̃`, `A` and `b`, with
  `R = Rational{BigInt}` or `BigFloat`.
- **The constructor checks** that the parts are square and of equal
  size, that `Ã` is strictly lower triangular and `A` lower triangular,
  and the [admissibility](#tableaus) condition. Each failure is an
  `ArgumentError` that says why.
- **Named constructors** (decided): `IMEXSSP222()`, `IMEXSSP2322()`,
  `IMEXSSP2332()`, `IMEXSSP3332()`, `IMEXSSP3433()`, `ARS222()` and
  `ARS443()` (`IMEXSSP2332()` added in step 1).
  - These are OrdinaryDiffEq's names, so an oracle test reads as a
    comparison of like with like. `IMEXSSP2332` has no upstream
    counterpart; it follows the same rule.
  - `IMEXSSPksσp` is Pareschi–Russo's SSPk(s,σ,p). The short
    `IMEXSSP222` is SSP2(2,2,2).
  - They clash with OrdinaryDiffEqSDIRK's exports, so the tests
    `import` it and qualify its names.
  - They are functions returning an `IMEXTableau`, not constants,
    because a `BigFloat` does not survive precompilation reliably.
- **A caller's own tableau** goes through the same constructor.
- **Properties are computed in the tests.** The order conditions,
  L-stability and the SSP coefficient are not package API.

What step 1 settled (proposed in step 1, decided 2026-09-24):
- **The fields** are `name`, `Ã`, `b̃`, `A`, `b`, `c̃` and `c`.
  - The names are Pareschi–Russo's and ARS's, `"SSP3(4,3,3)"` and
    `"ARS(4,4,3)"`.
  - `c̃` and `c` are the row sums of the stored coefficients, taken in
    `R`: exact for a rational tableau, and one 256-bit rounding
    otherwise. For SSP3(4,3,3) they come out exactly `(0, 0, 1, 1/2)`
    and `(α, 0, 1, 1/2)`.
- **`R` follows the input.** If every coefficient given is an integer or
  a rational, `R = Rational{BigInt}`. Otherwise `R` is 256-bit
  `BigFloat`, which holds a `Float64` exactly. `IMEXTableau{R}(…)`
  chooses explicitly.
- **Two more refusals:** a tableau with no stages, and a coefficient
  that is not finite.
- **It prints as** `IMEXTableau{BigFloat}("SSP3(4,3,3)", 4 stages)`
  (amended in step 6, which recorded it and added the test).
- **`IMEXTableau` is exported** beside the seven names, for a caller's own
  tableau.
- **Internal functions for step 2's plan**, in `src/tableau.jl`:
  - `nstages`, and the per-stage patterns `solves`, `explicit_used` and
    `implicit_used`, as `Vector{Bool}`;
  - `scratch_count`, the count of
    [The stage plan and storage](#the-stage-plan-and-storage-decided);
  - `coefficients(T, Tt, tab)`, the named tuple
    `(; Ã, b̃, γ, Ā, b̄, c̃, c)`.
- **What `coefficients` holds.**
  - `γ` is the diagonal `a_kk`.
  - `Ā` holds `a_kj/a_jj` for `j < k` and `a_jj ≠ 0`, and `b̄` holds
    `b_j/a_jj`. Both are zero elsewhere.
  - The coefficients are converted to `T` and the abscissae to the time
    type `Tt`.
  - Each value is formed in `R`, rounded to 256 bits and converted once,
    so it is the correctly rounded 256-bit value (a test, in `Float32`,
    `Float64` and `BigFloat`).
  - For `T = BigFloat` it has 256 bits, whatever the global precision.
    A state at a higher `BigFloat` precision therefore gets coefficients
    good to about 1e−77, not to its own precision.

Values suffice because the stage plan below gives the compiler the
tableau's structure anyway. This resolves "values or types".

### The stage plan and storage (decided)

`init` compiles the tableau for `T` and `Δt` into a **stage plan**:
- per stage, a tuple of the `(coefficient, array)` pairs of its `u★`,
  with only the nonzero terms;
- per stage, whether it solves (`a_kk ≠ 0`), and whether it is
  explicit-used and implicit-used;
- the update, in the same form.

The plan is a heterogeneous tuple whose type records the tableau's
nonzero pattern. `init` is therefore type-unstable, once, behind a
function barrier. `step!` is type-stable and, on the broadcast path,
allocation-free, and it is unrolled over the stages. By owner it allocates
a bound independent of the state size, and nothing at one thread
(amended in step 5; [By owner, as
built](#by-owner-as-built-measured-in-step-5)). Measured in step 2: `@inferred step!` holds, and
a step allocates 0 bytes for every tableau, with `Float64`, `Float32` and
`ComplexF64` states. On an Apple M3 at one thread, with trivial callbacks,
an SSP3(4,3,3) step on 10⁶ `Float64` entries takes 6.1 ms. It makes 53
state-sized reads and writes, so that is about 70 GB/s.

A **structural zero is never read**. The array for a skipped tendency is
not allocated, so no coefficient multiplies it, and `0·NaN` cannot occur
(TreeGRRMHD: "masks branch, never multiply").

**Storage.** All scratch comes from `similar(u0)`. `init` writes it once
through the same partition as the stage arithmetic (below), so that
first touch puts each page on the NUMA domain that will use it. Nothing
reads that initial value. `u★` is formed in the array that will then
hold `d_k`, since `d_k = U − u★` can overwrite `u★` element by element.
So the scratch is:
- `U`;
- one array per implicit-used stage;
- one array per explicit-used stage;
- one more if some solving stage that is not implicit-used has a
  nonempty row.

For SSP3(4,3,3) that is 1 + 4 + 3 = 8 arrays, besides `integ.u`.

**An empty row forms no `u★`** (proposed in step 2, decided 2026-09-24). A
stage whose row is empty in both parts has `u★ = uⁿ`. If it solves, the
integrator passes `integ.u` itself to `solve_imp!` as `u★`: `U` is copied
from it, and `d_k = U − uⁿ`. Forming `u★` in `d_k` first would cost one
more state pass per step, for every IMEX-SSP scheme's stage 1. This amends
the last item of the scratch count, which read "if some solving stage is
not implicit-used", and `scratch_count` with it (amended in step 2). None
of the seven named tableaus has such a stage, so their counts are
unchanged.

What else step 2 settled (proposed in step 2, decided 2026-09-24):
- **The plan's layout.** A `Stage{Solves,ExplicitUsed,ImplicitUsed}` per
  stage holds its terms, where `u★` is formed, the stage value `U` that
  `f_exp!` reads (`integ.u` itself at a trivial stage), the `d_k` and
  `k̃_k` arrays or `nothing`, `γΔt` in `T`, and `c̃_k`, `c_k` in the time
  type. The three flags are type parameters, so `step!` branches at
  compile time.
- **The pattern is the exact tableau's.** A term is present where the
  exact coefficient is nonzero, and an array exists where `solves`,
  `explicit_used` and `implicit_used` say so, whatever the coefficient
  becomes in `T`. A coefficient that underflows to zero in `T` keeps its
  term, which multiplies an array that exists and has been written (a
  test, with `ã₂₁ = 10⁻⁶⁰` in `Float32`).
- **The explicit coefficients are `Δt ã_kj` and `Δt b̃_j`**, formed in
  `T` from `T(Δt)` and the converted `ã_kj`, `b̃_j`: one more rounding,
  of relative size `eps(T)`, below that of the combination itself. The
  increment coefficients `a_kj/a_jj` and `b_j/a_jj` do not involve `Δt`.
  `γΔt = T(Δt)·a_kk` likewise.
- **The terms' order** is the explicit terms by increasing `j`, then the
  implicit ones by increasing `j`, after `x₀ = uⁿ`, summed left to right.
- **A dead stage does nothing.** A stage that makes no solve and is not
  explicit-used is read by nothing, so its `u★` is not formed. A stage
  that solves but is read by neither part still makes its one
  `solve_imp!` call, as "One step" says; its `u★` then takes the extra
  array. No named tableau has either.
- **First touch writes zero.** `init` fills each scratch array with
  `zero(eltype(u0))`, through the partition.
- **`integ.u` is first-touched too** (proposed in step 5, decided
  2026-09-24). By owner, and unless `alias_u0 = true`, `init` makes
  `integ.u` as `similar(u0)` and copies `u0` into it through the
  partition, where the broadcast path calls `copy(u0)`. The state is read
  and written by every combination, as the scratch is. An aliased `u0`
  stays where its caller put it.
- **The plan checks itself.** `init` throws an internal error if the plan
  allocated other than `scratch_count(tab)` arrays, or if a term reads an
  array the pattern did not allocate.

### Stage arithmetic (decided)

**Every combination is one fused linear combination**,
`dst = x₀ + Σ c_j x_j`. That is one pass that reads `m + 1` arrays and
writes one, never a sequence of axpy passes. On the host this arithmetic
is limited by memory bandwidth. TreeWave measured it as 79% of a
64-thread RK4 step, flat at 1.0× at every thread count.

**Where the data lives matters as much as how many threads touch it.**
TreeAMR measured this on a 64-core EPYC 7543 (2026-09-23; `CODE.md`,
"What one process loses", on branch `claude/festive-bun-656842`):
- A block's data streams up to 2.7× faster when the core that last
  touched it touches it again.
- Launching each phase on whichever thread was free cost the RHS 2.4×.
- TreeAMR now runs every per-block pass on the block's owner. Block `b`
  belongs to thread `c` if chunk `c` of `threadchunks(nblocks)` contains
  it. Chunk `c` runs as a sticky task placed on default-pool thread `c`
  (`jl_set_task_tid`), which also nests inside other parallel loops.

The stage arrays are read and written by the caller's kernels, which run
by owner: `f_exp!`, through `scatter!`, and `solve_imp!`. So the stage
arithmetic must give each element to the thread that owns it too.
Otherwise every combination moves the whole state to other cores, twice
per stage. So there are two paths:

- **Broadcast, the default and the first to be implemented.** One fused
  broadcast per combination. It works for any array type, and on a
  device it already is a parallel kernel. On the host it is serial.
- **By owner, for a CPU `Array` with more than one thread.** At one
  thread it is accepted too, and is a plain loop (amended in step 5). The
  caller passes `partition`, a collection of `Threads.nthreads()` elements.
  Element `c` is an iterable of `UnitRange{Int}` index ranges into `u`,
  owned by thread `c`.
  - Each combination is a plain loop over thread `c`'s ranges, run as a
    sticky task on default-pool thread `c`, as TreeAMR places chunk `c`.
  - Forming `u★` writes the `d_k` array and `U` in the same pass.
  - For TreeAMR's multi-set state vector, thread `c` owns one range per
    field set: the entries of its blocks in that set's segment.
  - `init` checks that the ranges are disjoint and cover `u` exactly.
    If they do not, it throws an `ArgumentError` that says which index
    is missing or doubled.
  - `partition = :even` splits `eachindex(u)` into `nthreads()` equal
    contiguous ranges, by TreeAMR's `threadchunks` rule. That keeps the
    integrator's own passes stable from step to step, but it does not
    match anyone else's ownership.
  - This is the one place that indexes into the state, and it is
    confined to `Array`.
- **Keywords:** `partition = nothing` (broadcast), `:even`, or explicit
  ranges. A partition given for a non-`Array` state is an
  `ArgumentError`.

**The result does not depend on the path or on the thread count.** Each
element is computed alone, with its terms summed in a fixed order, so
broadcast and every partition give the same bits. This is a test.

**Where the partition comes from** (open). TreeGRRMHD needs a TreeAMR
function that returns, for a tuple of field sets, the per-thread index
ranges into their state vector. It is the same ownership rule
`launch_by_owner!` already uses. Until TreeAMR has it, the caller builds
the ranges from `threadchunks(nblocks)` and the set layout. This is a
request to add to TreeGRRMHD's upstream list, not a dependency here.
- Step 5 adds a helper that knows nothing of TreeAMR (proposed in step 5,
  decided 2026-09-24): the internal, unexported
  `block_partition(blocks, segments)`. `blocks[c]` is the range of block
  numbers thread `c` owns, one per thread, as `threadchunks(nblocks)`
  gives them, padded with empty ranges to `nthreads()`. `segments` holds
  one `(offset, blocklength)` per segment of equal-sized consecutive
  blocks. Thread `c` then owns one range per segment. A TreeAMR state
  vector is such a layout, one segment per field set with
  `blocklength = N^D · nvars`, if its sets are concatenated; the caller
  asserts that, not this package. The question stays open: the helper is
  not exported until TreeAMR's own function exists or Erik decides to
  export it.

**Rejected:**
- **Polyester** (`RK4(thread = True())`). TreeWave measured it as worse,
  because it starts a second thread pool.
- **KernelAbstractions.** It would be a dependency or an extension. Its
  default CPU schedule is exactly the affinity-losing launch that
  TreeAMR measured, and its static schedule refuses to nest.
- **A caller-supplied kernel.** It would push the bandwidth question onto
  every caller.

The combination sits behind one internal function, so the owner path
can come in a later step without changing the interface. In step 2 it is
`lincomb!(dst, x₀, terms, partition)`, with `terms` a tuple of
`(coefficient, array)` pairs and a per-element kernel that folds them
left to right. `copy_state!`, `increment!` (`d = U − u★`) and
`first_touch!` take the same last argument. `partition === nothing` is
one `broadcast!` each; step 5 adds methods (proposed in step 2, decided
2026-09-24). Step 5 adds `lincomb_copy!(u★, U, x₀, terms, partition)` too,
the fused pass that forms `u★` and `U` together. On the broadcast path it
is `lincomb!` then `copy_state!`, exactly the two broadcasts of step 2, so
that path is unchanged (amended in step 5).

### By owner, as built (measured in step 5)

`src/lincomb.jl`, for a CPU `Array` state. The numbers are from an Apple
M3 Pro (6 performance and 6 efficiency cores, 12 CPU threads, 36 GB),
under Julia 1.13.0 and 1.10.12.

**The keyword** (proposed in step 5, decided 2026-09-24). `partition` is
`nothing`, `:even`, or a collection of `Threads.nthreads()` elements.
Element `c` is a unit range, or an iterable of unit ranges, of linear
indices owned by default-pool thread `c`; it may be empty. Any
`AbstractUnitRange` of integers is accepted and converted to
`UnitRange{Int}`; a `StepRange`, a number or another symbol is refused.
`init` sorts the nonempty ranges by their first index and walks them, so
each refusal names the index: "the partition misses index 41 of the state
(1:100): no thread owns it", "the partition doubles index 60: thread 1's
range 1:60 and thread 2's range 60:100 both own it", a range out of
bounds, or the wrong number of elements. The checked form is an internal
`OwnerPartition`, which `init` also accepts as it is, so that the tests
can give it a hook.

**Placement** (measured in step 5). Thread `c`'s ranges run in one sticky
task placed by `jl_set_task_tid(task, threadpoolsize(:interactive) + c −
1)`: the id is 0-based, and the default pool's ids follow the interactive
pool's. The two versions differ in the default. With `--threads=4`, 1.13
makes one interactive thread, so the default pool is ids 2–5 and the main
task runs on id 1; 1.10 makes none, so the pool is ids 1–4 and the main
task runs on default-pool thread 1. With `--threads=4,1` and
`--threads=2,2` the two agree (ids 2–5, and 3–4). On both, a task placed
on thread `c` reports `Threads.threadid()` equal to the offset plus `c`,
whether it was placed from the main task, from inside `Threads.@spawn`, or
from inside a `Threads.@threads :static` loop. `test/owner_tests.jl`
records the id per range and checks it, at whatever thread count the
suite runs.

**One thread is a plain loop** (proposed in step 5, decided 2026-09-24),
on the calling task, with no task, as in TreeAMR's `threaded_chunks`. On
1.13 the calling task is usually on the interactive thread, not on
default-pool thread 1; the caller's own `threaded_chunks` at one thread
runs there too.

**Fresh tasks, not persistent workers** (proposed in step 5; Erik
(2026-09-24): keep fresh tasks until the Symmetry run of
`bench/symmetry_stage_arithmetic.sh` decides). Each
combination makes one fresh sticky task per thread, waits for all of them,
and then rethrows the first error, unwrapped from its
`TaskFailedException` to what the loop threw. `PLAN.md` asked for
persistent sticky workers instead if they reach zero allocations at no
loss in speed. A prototype in `bench/stage_arithmetic.jl` does, for one
combination: one worker per thread waiting on its own autoreset `Event`,
the job a mutable object built once and called with the thread number,
0 bytes per launch, and within noise of the fresh tasks in time (below).
It is not taken, for what the prototype leaves out:
- zero allocations need the job built once, so one job object per
  combination in the plan; a job that is an immutable struct is boxed on
  every launch (80 bytes, measured);
- the workers must be process-wide, since an integrator per regrid with
  workers of its own would leak them. So they need a lock, which
  serializes concurrent `step!`s of different integrators, and they must
  drop their reference to the last job, or it keeps that integrator's
  arrays alive;
- an interrupt or an error while the caller waits must not leave a
  worker's completion count to the next launch, which fresh tasks get for
  free.

TreeAMR's own passes (`threaded_chunks`, and `@threads :static` under
KernelAbstractions' static schedule) launch fresh tasks the same way, so
a TreeGRRMHD right-hand side already allocates per pass as this does per
combination (read from TreeAMR's source, not measured here).
This is the step-5 decision most worth a second look, with the Symmetry
numbers at 64 threads.

**Forming `u★` writes `d_k` and `U` in one pass**, `lincomb_copy!`. So
an SSP3(4,3,3) step is 9 combinations by owner, where the broadcast path
makes 12 broadcasts, and 39 state-sized reads and writes instead of 42.
At a stage with an empty row, `u★` is `integ.u` itself and only `U` is
written, as on the broadcast path.

**The loop is the broadcast's kernel** (measured in step 5). Each element
is `LinComb(cs)(x₀[i], x₁[i], …)`, the broadcast's own callable, and the
increment is `U[i] − u★[i]`, under `@inbounds @simd ivdep`. `@inbounds` is
safe because every range lies in `1:n` and every array is checked to have
length `n`, once per combination: a resized `integ.u` is a
`DimensionMismatch`, not an out-of-bounds write. `ivdep` holds by the
contract that `dst` may be `x₀` (and `d` may be `u★`) itself and no array
otherwise overlaps another. It matters: without it, LLVM's runtime alias
check sees `dst === x₀` in the update and in every increment and falls
back to a scalar loop. At one thread, on 1.25 million entries, an
in-place combination of 7 terms takes 1.82 ms without it and 1.09 ms with
it, and the in-place increment 0.38 ms and 0.24 ms; the broadcast takes
1.85 ms and 0.38 ms. Every other owner kernel is within a few percent of
its broadcast, or faster.

**Bitwise identity, tested and checked by mutation.** `owner_tests.jl`
runs the seven tableaus and the three corner tableaus of the mechanics
tests on `Float64`, `Float32` and `ComplexF64` states of 203 entries
(`u′ = cos t − u²(1 + u)` with a stiff relaxation, and limiters that
change the state), four steps each. The partitions are `:even`, a
multi-set one from `block_partition` (three segments of 7 blocks, one
range per set per thread) and an irregular one (ranges of 1 to 32
entries, reversed within a thread, interleaved between threads, and a
last thread with none). Every step is bitwise the broadcast's, at one and
at four threads, on 1.10 and on 1.13. A second test folds
`(1 + 2⁵⁴) − 2⁵⁴` and `−1 + (1 + 2⁻³⁰)(1 − 2⁻³⁰)` over 1027 entries,
which are 0 only left to right and without an FMA. Mutations of the loop,
each caught (failures in `owner_tests.jl` and `mechanics_tests.jl`
together, at four threads on 1.13):
- the terms reversed: 179 failures;
- a `muladd` fold: 185;
- a `@fastmath` fold: 185;
- `U` not written in the fused pass: 78. It passed at first, since the
  bitwise tests' solver writes all of `U`; the mock that checks `U = u★`
  on entry, and a solver that writes half of `U`, were added for it;
- the thread ids without the interactive offset: 18. That mutation is
  invisible on 1.10 with `--threads=4`, whose offset is 0;
- the overlap check removed: 2.

**Allocations** (measured in step 5). At one thread a step allocates
nothing, for every tableau and state type (a test). At more, each
combination allocates 64 bytes and, per thread, the task and its
closure: 403–433 bytes on 1.13 and 559–589 on 1.10, growing by about
16 bytes per term of the combination. It is the same for 100 and for 10⁶
entries, and for `Float64` and `ComplexF64`. So an SSP3(4,3,3) step,
9 combinations, allocates `9 (64 + 413 nt)` bytes on 1.13: 15 456 at 4
threads and 45 216 at 12, and 21 072 at 4 threads on 1.10. At 64 threads
that is about 240 KB per step. The test asserts that the
allocation is the same at 100 and at 100 000 entries, and at most
`launches · (128 + 768 nt)` bytes.
- Julia 1.10 needed one change for the one-thread claim: `check_lengths`
  raised its error inline, and building the message allocated 32 bytes
  per combination there even when nothing was wrong. The error is now
  raised by a `@noinline` function.

**Nesting** (a test). A partitioned `step!` called through
`fetch(Threads.@spawn step!(integ))`, from a sticky task placed on the
last thread, and from inside a `Threads.@threads :static` loop gives the
broadcast's bits and runs each range on its owner.

**The Mac numbers** (`bench/stage_arithmetic.jl`, Julia 1.13.0,
`--threads=n`, so with one interactive thread besides; 10⁸ bytes, 12.5
million `Float64`). Each entry is the least time of 20, and the least
of two runs of the sweep; GB/s is from the counted reads and writes,
without write-allocate. `update` is a combination of `x₀` and 7 terms
into an array of its own, 9 passes; `step` is SSP3(4,3,3) with callbacks
that return at once, 42 passes broadcast and 39 by owner; `launch` is
`update` on 1000 entries per thread. The machine was not idle (load
average 7.4–9.6 from other work), and the two runs differ by up to 19%.

| threads | update, broadcast | update, owner | update, persistent | step, broadcast | step, owner | launch, owner / persistent |
|---|---|---|---|---|---|---|
| 1 | 16.1 ms, 56 GB/s | 10.1 ms, 89 GB/s | — | 60.4 ms, 70 GB/s | 47.3 ms, 82 GB/s | 0.5 µs / — |
| 2 | 16.1 ms, 56 GB/s | 8.7 ms, 103 GB/s | 8.7 ms, 104 GB/s | 61.4 ms, 68 GB/s | 40.9 ms, 95 GB/s | 12 / 12 µs |
| 4 | 17.0 ms, 53 GB/s | 9.0 ms, 100 GB/s | 8.7 ms, 103 GB/s | 60.1 ms, 70 GB/s | 38.1 ms, 102 GB/s | 12 / 13 µs |
| 6 | 16.5 ms, 55 GB/s | 7.8 ms, 115 GB/s | 8.4 ms, 108 GB/s | 61.6 ms, 68 GB/s | 36.9 ms, 106 GB/s | 13 / 14 µs |
| 8 | 16.0 ms, 56 GB/s | 8.0 ms, 113 GB/s | 8.3 ms, 108 GB/s | 59.9 ms, 70 GB/s | 35.6 ms, 110 GB/s | 17 / 17 µs |
| 12 | 16.0 ms, 56 GB/s | 7.9 ms, 114 GB/s | 8.1 ms, 111 GB/s | 58.4 ms, 72 GB/s | 34.9 ms, 112 GB/s | 113 / 114 µs |

- **By owner is faster at every thread count, one included.** At one
  thread the owner loop over 8 operands streams 89 GB/s where the
  broadcast streams 56 (the owner loop's LLVM code has vector `fmul`s;
  why the broadcast is slower was not pursued), and the step is 22%
  faster. At 8 and 12 threads the step is 1.7 times faster.
- **The M3 Pro saturates early.** One core already streams 89 GB/s, and
  more threads add at most 30% to that. Past six threads the extra ones
  are efficiency cores, which macOS lets no process pin to, and an equal
  split then waits for the slowest thread. At 12 threads the launch costs
  113 µs, since with the interactive thread there are 13 threads for 12
  cores; at 8 threads one run measured 110 µs and the other 17 µs. So the
  counts past six are the M3's, and Symmetry's pinned run is the one that
  says what the owner path is worth.
- **Persistent workers are within noise of fresh tasks**, in time and in
  launch cost, and allocate nothing; fresh tasks allocate the bytes in
  the last column of the bench output (1040 at 2 threads to 5920 at 12,
  per combination).

**Symmetry** (pending a Symmetry run). `bench/symmetry_stage_arithmetic.sh`
runs the same sweep on one AMD node at 1–64 threads: pinned with first
touch, pinned and interleaved, and unpinned.

### File layout (decided)

- `src/IMEXRungeKutta.jl`: the module and its exports.
- `src/tableau.jl`: `IMEXTableau`, its checks, and the conversion to `T`.
- `src/tableaus.jl`: the seven tableaus, in closed form.
- `src/plan.jl`: the stage plan.
- `src/lincomb.jl`: fused linear combinations, broadcast and threaded.
- `src/integrator.jl`: `IMEXProblem`, `init`, `step!` and `solve!`.
- `test/`: one file per group under [Testing](#testing-decided).
  `test/runtests.jl` includes them into one `@testset`;
  `test/scaffold_tests.jl` checks that the package loads, that its four
  names are CommonSolve's bindings, and that `[deps]` is CommonSolve
  alone (amended in step 0). `test/tableau_properties.jl` holds the
  test-only tableau properties, and `test/tableau_tests.jl` asserts them
  (amended in step 1). `test/mocks.jl` holds the mock callbacks, which
  log their calls into buffers preallocated in `p`; `interface_tests.jl`
  and `mechanics_tests.jl` hold the interface and the mechanics items,
  and `smoke_order_tests.jl` a quick check of the order of SSP2(2,2,2) and
  SSP3(4,3,3) on `u′ = −u + cos t`; and `readme_tests.jl` evaluates the
  README's `julia` blocks and checks their result (amended in step 2).
  Step 3 adds `order_tests.jl`, `stiff_tests.jl`, `ap_tests.jl`,
  `ssp_tests.jl` and `oracle_tests.jl`, one per validation group of
  [Testing](#testing-decided), and `problems.jl`, the helpers they share
  (amended in step 3). Step 5 adds `owner_tests.jl`, the by-owner items of
  Mechanics, in a testset of its own after `mechanics_tests.jl`, whose
  corner tableaus it reuses (proposed in step 5, decided 2026-09-24).
- `bench/`: `stage_arithmetic.jl`, the thread sweep of step 5, and
  `symmetry_stage_arithmetic.sh`, its SLURM job, after TreeAMR's
  `bench/symmetry_affinity.sh` (proposed in step 5, decided 2026-09-24).
  They run in the package's own environment (`--project=.`), with only the
  standard library's `Printf` besides.
- Test-only dependencies are in `test/Project.toml`, with its own
  `[compat]` (amended in step 6: Erik moved test-only dependencies to
  test/Project.toml). Step 0 had proposed `[extras]` and `[targets]` in
  the root `Project.toml`, which now holds only `[deps]` CommonSolve and
  its `[compat]` for CommonSolve and `julia`.
  - Its `[deps]` are CommonSolve, LinearAlgebra, OrdinaryDiffEqSDIRK,
    TOML and Test. TOML is there for the project-file checks, and
    OrdinaryDiffEqSDIRK for the oracle, with the `[compat]` bound
    `"2.9.6"` ([The oracle](#the-oracle)). CommonSolve is there because
    the tests load it by name, which a dependency of the package alone
    does not allow (measured in step 6: without it, `scaffold_tests.jl`
    fails with "Package CommonSolve not found"). Its bound is the root
    `[compat]`'s, which the resolver applies through this package.
  - This package is not listed (measured in step 6). On Julia 1.10 and
    1.13 alike, `Pkg.test()` copies `test/Project.toml` to a temporary
    environment, keeps its `[compat]`, and adds this package to its
    `[deps]`, by path: both record the path in the manifest, and 1.13
    also writes a `[sources]` entry. It writes no `test/Manifest.toml`.
  - `scaffold_tests.jl` checks both files: the root has no `[extras]`,
    `[targets]`, `[weakdeps]` or `[sources]`; `test/Project.toml` has
    exactly those five, the one bound and no `[sources]`; and under
    `Pkg.test()` the active environment is that file's copy with this
    package added.
- The one exception is the device smoke run (proposed in step 4, decided
  2026-09-24): `test/metal_tests.jl` runs in `test/metal/Project.toml`,
  whose dependencies are Metal, Test and this package, developed from
  `../..`. `runtests.jl` does not include it. See
  [On a device](#on-a-device-measured-in-step-4).
- `.github/workflows/CI.yml` (proposed in step 0, decided 2026-09-24) has
  five cells (amended in step 6: Erik added the fifth):
  - Julia 1.10 on Linux, at one thread;
  - the current release on Linux, with `--check-bounds=yes` and
    coverage;
  - the current release on macOS;
  - the current release on Linux at four threads;
  - Julia 1.10 on Linux at four threads, where the owner path's
    placement and allocations differ from the current release's
    ([By owner, as built](#by-owner-as-built-measured-in-step-5)).

  Every cell but the bounds-checked one runs with `--check-bounds=auto`,
  so that the allocation tests run on the floor and at four threads.
  `julia-runtest`'s default, `yes`, would skip them in every cell.
  There is no Metal cell (proposed in step 4, decided 2026-09-24):
  GitHub's hosted macOS arm64 runners are virtual machines without Metal
  support. A push to `main` that changes only Markdown files does not run
  CI, unless one of them is `README.md`, whose `julia` block
  `readme_tests.jl` runs (proposed in step 6, decided 2026-09-24; until
  step 6 every `.md` was ignored, so a README-only change could break the
  suite unseen).

### Documentation (decided)

README and docstrings, no Documenter site for now. Docstrings are
prose-first and point at this document. The README has a worked
example, including a stage solver. A site can be added later without
changing anything else.

The README carries one badge, CI's, for `.github/workflows/CI.yml` on
`main` of `eschnett/IMEXRungeKutta.jl` (proposed in step 4, decided
2026-09-24). CI's bounds-checked cell uploads coverage to Codecov on
`main`, but only with a `CODECOV_TOKEN` secret that this repository may
not have, and with `fail_ci_if_error: false`. So there is no Codecov badge
until an upload has been seen to succeed.

## Why not an existing package

Surveyed 2026-09-24. The tableaus exist elsewhere; the stage contract
does not.

### OrdinaryDiffEq (OrdinaryDiffEqSDIRK 2.9.6)

It has had `IMEXSSP222`, `IMEXSSP2322`, `IMEXSSP3332`, `IMEXSSP3433`,
`ARS222`, `ARS232` and `ARS443` since mid-2026
(SciML/OrdinaryDiffEq.jl#3704, #3705), on `SplitODEProblem`, with
Julia 1.10 compat. Measured:

- **The stage solve is OrdinaryDiffEq's.** It is handed `g` and solves
  each stage on the whole vector:
  - by Newton, with a dense `N×N` Jacobian by default (allocated for
    N = 10⁴);
  - or by fixed-point iteration, which diverges for stiff `g`.

  The one hook for a custom stage problem (`ODENLStepData` with
  `NonlinearSolveAlg`) serves ModelingToolkit. It still runs inside the
  Newton convergence loop, and still builds the Jacobian and
  linear-solver machinery.
- **No stage limiter.** Its IMEX methods reject `stage_limiter` with an
  error; only `step_limiter` exists.
- **Unneeded explicit evaluations.** It evaluates `f` at stage 1 even
  where column 1 of `Ã` and `b̃₁` are zero, and once more per step for
  the interpolant: five evaluations per SSP3(4,3,3) step where three are
  needed.
- **Mistimed last explicit stage.** It is evaluated at `t + Δt` instead
  of `t + c̃_s Δt`. As a result, IMEXSSP3332 and IMEXSSP3433 are only
  first order when `f` depends on `t` (1.01 instead of 2 and 3). The fix
  is one line. Reported as SciML/OrdinaryDiffEq.jl#4620 (2026-09-24).
- A heavy dependency tree (about 180 packages, with ForwardDiff,
  LinearSolve and NonlinearSolve), and serial stage arithmetic.

**Contributing** a user stage-solver algorithm, a stage limiter for the
IMEX methods and zero-column skipping was considered and not pursued
(decided). The downstream user would wait on review of a code path that
had several regressions in 2026. The semantics above (tendency before
limiter, one call per stage) would rest on behaviour upstream does not
promise. Revisit if upstream gains a user stage-solver hook.

**Use as a test oracle** (decided). Upstream agrees with a direct
reference step to 1e−16 on a linear problem with default settings. These
restrictions apply:
- the state must be real (its default AD Jacobian rejects a complex
  state);
- `f` must not depend on `t`, until #4620 is fixed. That holds only for
  the tableaus whose last explicit abscissa `c̃_s` is not 1, SSP3(3,3,2)
  and SSP3(4,3,3) (`c̃_s = 1/2`), since upstream's `t + Δt` is right where
  `c̃_s = 1` (amended in step 3). With a `t`-dependent `f`, the other four
  agree with upstream to 3.1e−16 over ten steps, and those two differ by
  0.0225;
- its `ARS443` has `b̃ = b`, where this package has the last row of `Ã`
  (amended in step 1; see "Cross-checks" under [Tableaus](#tableaus)). So
  the comparison for ARS(4,4,3) is against a tableau built with upstream's
  `b̃`;
- SSP2(3,3,2) is in neither upstream, so it has no oracle (amended in
  step 1). Its coefficients, recalled by the step-1 reviewer, rest on the
  order conditions, `R(∞) = 0` and the SSP coefficient alone, until they
  are checked against the paper (open). Step 3's measurements are further
  evidence ("Cross-checks").

The oracle comparisons are in [The oracle](#the-oracle).

### ClimaTimeSteppers (1.0.1)

The closest match. It has:
- an IMEX-SSPRK algorithm with the Pareschi–Russo tableaus (SSP222,
  SSP322, SSP332, SSP333, SSP433) and the ARS family;
- a monotonicity limiter `lim!`, plus `constrain_state!` hooks;
- zero-column skipping;
- the same tendency recovery, `T_imp = (U − temp)/dtγ`.

It still does not fit:
- **The implicit stage always goes through `solve_newton!`.** A user
  solver could ride in `initialize_imp!`, which is documented as the
  Newton initial guess. But with `max_iters = 0`, the residual (a `g`
  evaluation) and the Jacobian (`Wfact`) are still evaluated once per
  stage, and a Jacobian prototype or a Jacobian-free Krylov method is
  still required.
- **The tendency is taken after `constrain_state!`**, the opposite of
  the decision above.
- Its dependencies (ClimaComms, Krylov, LinearOperators, NVTX) and its
  design centre, climate models on ClimaCore spectral elements.

## Testing (decided)

Testset names are claims, each opening with a comment that names the
failure mode it guards.

- **Tableaus:**
  - each meets its stated order and fails at the next, exactly for the
    rational forms and to a tiny residual in `BigFloat` for the others;
  - the triangularity and admissibility are as stated;
  - the recorded properties (stiff accuracy, L-stability, SSP
    coefficient) are regression-tested;
  - also (amended in step 1):
    - a perturbation of any one coefficient fails a check, except the
      one coefficient named in "What the order conditions do not see";
    - the closed forms are the derived values;
    - the tableaus and their converted coefficients do not depend on the
      global `BigFloat` precision;
    - the 14 printed digits of SSP3(4,3,3) are the closed form rounded;
    - upstream's ARS(4,4,3) variant is third order too.
- **Mechanics:**
  - `step!` is allocation-free after warm-up;
  - `solve_imp!` is called once per implicit stage, with the documented
    arguments (a mock);
  - `f_exp!` is called once per nonzero column, at `tⁿ + c̃_k Δt` (a
    mock): three times per SSP3(4,3,3) step;
  - the stage limiter is called exactly before each `f_exp!` call, on
    the same array, and never on `integ.u` (a mock);
  - at a trivial first stage, `f_exp!` receives `integ.u` itself;
  - scratch filled with NaN before the first step leaves no NaN in the
    result, so no structural zero is read;
  - an exception thrown by `solve_imp!` leaves `integ.u` and `integ.t`
    unchanged;
  - with `g ≡ 0`, the result equals the explicit RK method;
  - untouched components match the explicit-only run bitwise;
  - broadcast, `partition = :even` and an explicit multi-range
    partition give bitwise identical results, at one thread and at four;
  - under a partition, each range is processed on its thread (a mock
    records `Threads.threadid()` per range);
  - a partition with a gap or an overlap is refused;
  - these three are in `test/owner_tests.jl` (amended in step 5), with:
    the traps of reassociation and FMA over a vectorized range; `U = u★`
    on entry to `solve_imp!`, and NaN scratch, by owner; the combination
    count per step and `init`'s first touch, by the hook; nesting inside
    `@spawn`, a sticky task and `@threads :static`; an error on a worker
    reaching the caller as itself, after every worker has finished; a
    resized `integ.u` refused; `@inferred step!`; and the allocations of
    [By owner, as built](#by-owner-as-built-measured-in-step-5);
  - a Float32 run works;
  - also (amended in step 2): three tableaus of a caller's own reach the
    plan's corner cases, the extra array, an empty-row solving stage and a
    dead stage; the call sequence of each step equals one rederived from
    the exact tableau, time by time; the stage limiter's change reaches
    `f_exp!` and nothing else; and `step!` is allocation-free for
    `Float64`, `Float32` and `ComplexF64` states;
  - a device smoke run passes on Metal, gated by
    `IMEXRUNGEKUTTA_TEST_METAL=1`. It is a short `Float32` run on an
    `MtlArray` state with scalar indexing disallowed, so it covers the
    broadcast path and checks that nothing indexes the state. Metal
    enters through an environment of its own, `test/metal/Project.toml`,
    and never through the package's test environment (amended in step
    4; [On a device](#on-a-device-measured-in-step-4)).
- **Order:**
  - on the split linear ODE `u′ = iu − u`, the observed order equals the
    tableau's, ±0.1;
  - likewise on `u′ = −u + cos t` (implicit `−u`, explicit `cos t`),
    which catches a mistimed explicit stage (the #4620 failure mode) and
    is invisible to a problem where `f` does not depend on `t`.
  - Measured in step 2, by mutation: #4620's mistiming, the last explicit
    stage at `tⁿ + Δt`, drops SSP3(3,3,2) and SSP3(4,3,3) to order 1.02
    on it, as upstream measured. But a swap of `c̃` and `c` at every stage
    is invisible to it. The coupling conditions `b̃ᵀc = 1/2` and
    `b̃ᵀc² = 1/3` make an `f` of `t` alone integrate the same with either
    abscissa, up to order 3. The mechanics test of the call times catches
    that swap (61 failures), and so does the `g ≡ 0` comparison with the
    explicit method, whose `f` depends on `u` too (amended in step 2).
- **Stiff limit:** on the Kaps problem with `ε ∈ {1, 1e−3, 1e−6, 1e−9}`,
  the observed order per `ε` is recorded, including any order reduction.
- **Asymptotic preservation:** at `ε = 1e−12`, one step lands on the
  equilibrium manifold to `O(ε)`.
  - That holds only for the ARS schemes, and for SSP2(3,2,2) and
    SSP2(3,3,2) only when `f` does not depend on `t` (amended in step 3).
    For the others the claim is where the step lands: `ū + Δt wᵀF`, to
    `O(ε)`, with `w` as in "Where a step ends in the stiff limit". The
    test asserts `O(ε)` where `w = 0`, the displacement and its
    coefficient `wᵀ𝟙 = 1 − bᵀA⁻¹c̃` for the three with `wᵀ𝟙 ≠ 0`, and
    `wᵀc̃` for the two with only `wᵀ𝟙 = 0`; that the displacement does
    not accumulate over 100 steps; and, on the Kaps problem, the residual
    `y₁ − y₂²` of one step and its order in `Δt`.
- **SSP:** upwind advection with stiff relaxation keeps the total
  variation non-increasing for `Δt ≤ C Δt_FE`; the measured `C` is
  recorded against the explicit part's SSP coefficient.
  - The relaxation is toward the initial square wave itself, and the
    property asked of a step is `TV(uⁿ⁺¹) ≤ max(TV(uⁿ), TV(φ))`, which
    the exact solution has; without relaxation it is `TV(uⁿ⁺¹) ≤ TV(uⁿ)`.
    `C` is measured without relaxation, at `ε = 10⁻²` and at
    `ε = 10⁻¹²` (amended in step 3; [SSP and total
    variation](#ssp-and-total-variation)).
- **Oracle:** each tableau OrdinaryDiffEqSDIRK has matches it to 1e−12
  over ten steps, within the restrictions above. That is all but
  SSP2(3,3,2) (amended in step 1).
  - Also (amended in step 3): with a `t`-dependent `f`, it matches
    where `c̃_s = 1` and is `@test_broken` where not (#4620); the
    14-digit SSP3(4,3,3) is compared on its own; and our ARS(4,4,3)
    differs from upstream's by `O(Δt⁴)` per step.

## Validation (measured in step 3)

The numbers of step 3's validation files, measured on an Apple M3 with
Julia 1.13.0 and identical on 1.10.12. Each is asserted by its test, to
the tolerance given, so that a regression is caught.

**How the tests measure** (proposed in step 3, decided 2026-09-24):
- an observed order is the least-squares slope of `log error` against
  `log Δt` over three step sizes (`fitted_order`, in `test/problems.jl`);
- a test asserts both the theory (the stated order, `O(ε)`, a formula)
  and the measured number, which is recorded here;
- the problems the stiff and the AP tests share, the Kaps problem with its
  split and its closed-form stage solve, are in `test/problems.jl`, beside
  `tableau_properties.jl` and `mocks.jl`.

### Observed orders

`test/order_tests.jl`. Both problems have the implicit part `−u` and run
to `t = 1`, with `Δt = 1/10, …, 1/160`; the fit uses the finest three,
`1/40, 1/80, 1/160`. Each is asserted within 0.1 of the stated order, and
within 0.005 of the number here.

| | stated | `u′ = iu − u` (complex) | `u′ = −u + cos t` |
|---|---|---|---|
| SSP2(2,2,2) | 2 | 2.004 | 2.002 |
| SSP2(3,2,2) | 2 | 2.008 | 2.029 |
| SSP2(3,3,2) | 2 | 2.003 | 2.001 |
| SSP3(3,3,2) | 2 | 1.994 | 1.993 |
| SSP3(4,3,3) | 3 | 3.007 | 3.006 |
| ARS(2,2,2) | 2 | 2.004 | 2.009 |
| ARS(4,4,3) | 3 | 3.005 | 3.001 |

### The stiff limit

`test/stiff_tests.jl`: the Kaps problem,
`y₁′ = −(2 + 1/ε) y₁ + y₂²/ε` and `y₂′ = y₁ − y₂ − y₂²`, whose exact
solution is `y₁ = e^{−2t}`, `y₂ = e^{−t}` for every `ε`, from
`y(0) = (1, 1)` to `t = 1`. The implicit part is `g = ((y₂² − y₁)/ε, 0)`,
whose stage solve is exact, `U₂ = u★₂` and then
`U₁ = (ε u★₁ + γΔt U₂²)/(ε + γΔt)`, and the explicit part is the rest
(amended in step 6: this said "split as `PLAN.md` says"). The error is the
largest over the steps and over both components; the order is fitted over
`Δt = 1/40, 1/80, 1/160`. Each number is asserted to ±0.15.

| | stated | `ε = 1` | `ε = 10⁻³` | `ε = 10⁻⁶` | `ε = 10⁻⁹` |
|---|---|---|---|---|---|
| SSP2(2,2,2) | 2 | 2.025 | 2.225 | 2.003 | 2.002 |
| SSP2(3,2,2) | 2 | 2.023 | 2.149 | 2.004 | 2.004 |
| SSP2(3,3,2) | 2 | 2.016 | 2.117 | 1.998 | 1.998 |
| SSP3(3,3,2) | 2 | 3.025 | 3.386 | 2.997 | 2.996 |
| SSP3(4,3,3) | 3 | 3.026 | 2.511 | **1.989** | **1.988** |
| ARS(2,2,2) | 2 | 2.020 | 2.003 | 2.010 | 2.010 |
| ARS(4,4,3) | 3 | 3.019 | 1.493 | 3.017 | 3.012 |

- **SSP3(4,3,3) drops to order 2 in the stiff limit**, in the stiff
  component: `y₂` alone keeps 3.010 and 3.011 at `ε = 10⁻⁶` and `10⁻⁹`
  (asserted to ±0.05). Not stiff, at `ε = 1`, it is third order.
- **SSP3(3,3,2) shows order 3**, above its stated 2, at every `ε` but
  `10⁻³`. The Kaps problem cannot see its failing order-3 condition: its
  exact solution lies on `y₁ = y₂²`, where `g = 0`, so every elementary
  differential with `g` at a leaf vanishes. The order tests, which can,
  give 1.994 and 1.993.
- **At `ε = 10⁻³`**, `Δt/ε` runs from 25 to 6, between the regimes, and
  the slope is not an asymptotic order. Over `Δt = 1/10, …, 1/640` the
  local slopes of ARS(4,4,3) run from 0.7 to 2.3, and SSP3(3,3,2)'s from
  1.75 to 5.0. The numbers are regression values.
- ARS(4,4,3), stiffly accurate in both parts, keeps order 3 in the stiff
  limit; ARS(2,2,2) and the three second-order SSP schemes keep 2.

### Asymptotic preservation

`test/ap_tests.jl`, at `ε = 10⁻¹²`. Two problems:
- **Relaxation**, `u′ = f(t) − (u − ū)/ε`, from `u = ū`. Every step of
  every tableau lands on `ū + Δt wᵀF` to at most 5.83ε (SSP2(2,2,2) and
  SSP3(3,3,2) with `f = 1`; at most 3.15ε with `f = cos t`), with `w`
  from "Where a step ends in the stiff limit"; asserted to 10ε. With `f = 1`
  the displacement is `Δt wᵀ𝟙`, with `f = cos t` from `t = 1` its leading
  term is `Δt (wᵀ𝟙) cos 1`, or `−Δt² (wᵀc̃) sin 1` where `wᵀ𝟙 = 0`.
- **Kaps**, from `y(0) = (1, 1)`, the residual `r = y₁ − y₂²` after one
  step. The manifold is invariant under the explicit part
  (`f₁ − 2y₂f₂ = 0` on it), so the `O(Δt)` term of the relaxation
  vanishes here, and the residual is higher order.

| | `wᵀ𝟙` | `wᵀc̃` | `f = 1`: `d/Δt` | `f = cos t`: `d/(Δt² (−sin 1))`, `Δt = 0.025` | Kaps `r`, `Δt = 0.1` | Kaps `r`, order in `Δt` |
|---|---|---|---|---|---|---|
| SSP2(2,2,2) | −1/√2 | — | −0.70710678 | — | 1.7046e−2 | 1.999 |
| SSP2(3,2,2) | 0 | 1/2 | 2.0ε/Δt | 0.50396 | 9.9750e−3 | 1.999 |
| SSP2(3,3,2) | 0 | 1/4 | 1.0ε/Δt | 0.25231 | 4.7483e−3 | 1.979 |
| SSP3(3,3,2) | −1/√2 | — | −0.70710678 | — | 5.5907e−4 | 2.993 |
| SSP3(4,3,3) | −0.28436465 | — | −0.28436465 | — | −7.1610e−3 | 1.962 |
| ARS(2,2,2) | 0 (`w = 0`) | 0 | 1.0ε/Δt | 0 | 9.4e−14 (0.094ε) | — |
| ARS(4,4,3) | 0 (`w = 0`) | 0 | 1.0ε/Δt | 0 | 4.2e−14 (0.042ε) | — |
| ARS(4,4,3), upstream's `b̃` | — | — | — | — | 6.4568e−5 | 3.974 |

- The Kaps residuals are asserted to 1% and their orders, over
  `Δt = 0.1, 0.05, 0.025, 0.0125`, to ±0.15 of 2, 2, 2, 3, 2 and 4. For
  ARS the residual is asserted below ε at every `Δt`.
- **Over many steps nothing accumulates.** For the relaxation, see "Where
  a step ends in the stiff limit". For Kaps, after 10 steps of `Δt = 0.1`
  or 100 of `Δt = 0.01`, the residual at `t = 1` is that of one step from
  the exact solution at `1 − Δt`, to within 0.32%; it decays with the
  solution, `∝ e^{−2t}`. For ARS it stays below ε.

### SSP and total variation

`test/ssp_tests.jl`: `u_t + u_x = −(u − φ)/ε` on the periodic `[0, 1)`,
100 cells, first-order upwind explicit and the relaxation implicit, from
`u⁰ = φ`, a square wave with `TV = 2`. `C = Δt/Δt_FE`, with
`Δt_FE = Δx`, is the largest value at which 50 steps keep
`TV(uⁿ⁺¹) ≤ max(TV(uⁿ), TV(φ))` to a relative 1e−12 (without relaxation,
`TV(uⁿ⁺¹) ≤ TV(uⁿ)`), by bisection on `[0, 4]` to 1e−4 (proposed in step
3, decided 2026-09-24). `C` is asserted to 1e−3.

| | SSP coefficient (step 1) | `C`, no relaxation | `C`, `ε = 10⁻²` | `C`, `ε = 10⁻¹²` | stiff limit: TV rise in step 1 |
|---|---|---|---|---|---|
| SSP2(2,2,2) | 1 | 1.0000 | 0.7158 | 0 | `4κC`, `κ = 1/√2` |
| SSP2(3,2,2) | 1 | 1.0000 | 0.7088 | 1.0024 | `8(C − 1)ε/Δx` for `C > 1` |
| SSP2(3,3,2) | 2 | 2.0000 | 1.4709 | ≥ 4 | 0 |
| SSP3(3,3,2) | 1 | 1.0000 | 0.8613 | 0 | `4κC`, `κ = 1/√2` |
| SSP3(4,3,3) | 1 | 1.0000 | 0.6840 | 0 | `4κC`, `κ = 0.28436465` |
| ARS(2,2,2) | 0 | 1.0000 | 0.7286 | ≥ 4 | 0 |
| ARS(4,4,3) | 0 | 0.0021 | 0.0021 | ≥ 4 | 0 |

- **Without relaxation** `C` is the linear threshold of the explicit
  stability polynomial, and equals the SSP coefficient for the five
  IMEX-SSP schemes. ARS(2,2,2)'s polynomial is `1 + z + z²/2`, threshold
  1, although its SSP coefficient is 0. ARS(4,4,3)'s has the `z⁴`
  coefficient `−7/288`, so no positive threshold; its 0.0021 is where the
  `O(C⁴)` rise falls below the tolerance, and `C = 0.01` fails (a test).
- **With relaxation as fast as the advection** (`ε = 10⁻²`, `Δt/ε = C`),
  every `C` is below the explicit threshold.
- **In the stiff limit** the non-stiffly-accurate three have `C = 0`: each
  step ends `Δt wᵀF` off `φ`, an overshoot of `κC` (`κ = −wᵀ𝟙`) on each
  side of both jumps, so the first step raises the total variation by
  exactly `4κC` (asserted to 1e−6 relative, at `C = 0.5, 1, 2, 4`). For
  SSP3(4,3,3) at `C = 1` that is 1.137 on a total variation of 2, with
  the solution reaching 1.284 and −0.284. The four with a stiffly accurate
  implicit part end on `φ` to `O(ε)`: SSP2(3,2,2)'s rise, `8(C − 1)ε/Δx`,
  is 2.4e−9 at `C = 4` (asserted below 10⁴ε), and puts its bisected `C`
  at 1.0024 for `ε = 10⁻¹²` but 3.4999 for `ε = 10⁻¹⁵`. The other three
  stay at `TV(u⁰)` to round-off up to `C = 4`.

This is the stiff-limit displacement of
[Asymptotic preservation](#asymptotic-preservation) seen at a
discontinuity: where the equilibrium is not a steady state of the
explicit part, SSP2(2,2,2), SSP3(3,3,2) and SSP3(4,3,3) put an `O(1)`
overshoot, `κ Δt |f|` with `|f| ~ jump/Δx`, next to each jump of it.

### The oracle

`test/oracle_tests.jl`: `SplitODEProblem(g, f, …)` with upstream's
defaults, ten steps of `Δt = 0.1` of `u′ = Lu + Mu + a cos(3t) v` on three
real components, `Lu` implicit. Upstream is OrdinaryDiffEqSDIRK 2.9.6.

| | `c̃_s` | `a = 0` | `a = 1` |
|---|---|---|---|
| SSP2(2,2,2) | 1 | 5.6e−17 | 1.4e−16 |
| SSP2(3,2,2) | 1 | 2.5e−16 | 2.6e−16 |
| SSP3(3,3,2) | 1/2 | 1.4e−16 | 0.0225 (#4620, `@test_broken`) |
| SSP3(4,3,3) | 1/2 | 5.3e−16 | 0.0225 (#4620, `@test_broken`) |
| ARS(2,2,2) | 1 | 2.8e−17 | 1.4e−16 |
| ARS(4,4,3), upstream's `b̃` | 1 | 8.3e−17 | 3.1e−16 |

- The 1e−12 tolerance absorbs upstream's 14-digit SSP3(4,3,3): ten steps
  with the printed digits, held exactly as decimals, differ from the
  closed form by 4.9e−16, and from upstream by 1.9e−16.
- Our own ARS(4,4,3), the paper's, differs from upstream's by 2.5e−5 over
  the ten steps ("Cross-checks").
- `test/Project.toml` adds OrdinaryDiffEqSDIRK with the compat bound
  `"2.9.6"`, that is `[2.9.6, 3)` (proposed in step 3, decided
  2026-09-24). A release that fixes #4620 turns the two `@test_broken`
  into unexpected passes, which fail the suite, and so is noticed.

## On a device (measured in step 4)

`test/metal_tests.jl`, on an Apple M3, with Metal 1.11.1 (GPUArrays
11.5.14), under Julia 1.13.0 and 1.10.12. The numbers are the same on both
unless given for each.

**How Metal gets in** (proposed in step 4, decided 2026-09-24). `PLAN.md`
offered a separate environment or a conditional `Pkg.add` in the gated
file. It is the separate environment, `test/metal/Project.toml`:
- `[deps]` Metal, Test and this package; `[compat]` Metal `"1.11"`;
  `[sources]` points this package at `../..`. Julia 1.11 and later read
  `[sources]`; 1.10 ignores it, so the command in `CLAUDE.md` runs
  `Pkg.develop(path = ".")` first, which works on both and leaves the
  tracked file unchanged. The root `Project.toml` still has no `[sources]`.
- Its manifest has 99 packages on 1.13 and 96 on 1.10, standard
  libraries included, and not OrdinaryDiffEqSDIRK. The run takes 14 s on
  1.13 and 11 s on 1.10, most of it compiling kernels.
- A `Pkg.add` inside the gated file would instead change `Pkg.test()`'s
  sandbox from within the test run, and resolve Metal against the whole
  test environment, oracle included.

**The gate** (proposed in step 4, decided 2026-09-24). Without
`IMEXRUNGEKUTTA_TEST_METAL=1` the file logs that it is skipped and exits 0
before loading anything. With it, a Metal that is not functional fails the
run: the run was asked for. `runtests.jl` does not include the file.

**The ordinary suite never sees Metal** (tests, in `scaffold_tests.jl`
and at the end of `runtests.jl`): Metal is in none of the `[deps]`,
`[weakdeps]` or `[extras]` of the root `Project.toml` and of
`test/Project.toml` (amended in step 6, when the test environment moved
there); it is not in the resolved test environment's manifest; under
`Pkg.test()`, whose load path is that environment alone,
`Base.find_package("Metal")` is `nothing` (Erik's global environment has
Metal, so a plain `julia --project=.` run would find it there, and the
check is skipped when the load path includes `@v#.#`); and after the last
file no loaded module is Metal. A test also checks that
`test/metal/Project.toml` names this package's UUID and points at this
checkout.

**The run.** `u′ = cos t − κu − (u − ū)/ε` in 4096 cells, `κ = 1/2`,
`ū = 1 + x`, `u(0) = ū + sin 2πx` and `ε = 10^{−3+3x}` for
`x = 0, 1/4096, …`, so that `Δt/ε` runs from 100 to 0.1. `f_exp!`, the
closed-form stage solve and both limiters, a floor far below the solution
that changes nothing, are one broadcast each. Ten steps of `Δt = 0.1`, of
SSP2(2,2,2) and SSP3(4,3,3), with `Float64` and with `Float32` time, on an
`MtlArray{Float32}` state with `Metal.allowscalar(false)` (a test that
scalar indexing then throws), against the same run on the CPU in
`Float32`.
- **Float64 is not needed on the device.** An `MtlArray{Float64}` is
  refused ("Metal does not support Float64 values"). The time stays on
  the host in its own type; every coefficient in a kernel, `γΔt`
  included, is `T`. A callback converts what it takes from `t` before its
  kernel, as `f_exp!` does with `T(cos t)`.
- **The device agrees with the CPU bitwise**: 0 ulps after every step, in
  all four runs, on both Julia versions. So Metal contracted nothing to an
  FMA here, and its division rounded as the CPU's does.
- **The tolerance is 4 ulps per step** (proposed in step 4, decided
  2026-09-24), in units of `eps(Float32) · max|u|`, cumulative: `4n` after
  step `n`. Contraction or a differently rounded division changes
  roundings, not the arithmetic, so the two runs differ by at most the sum
  of their rounding errors, and those do not grow here (the stiff cells
  contract; the others grow by `1 + O(Δt)`). The CPU run in `Float32` is
  11.5 (SSP2(2,2,2)) and 15.4 (SSP3(4,3,3)) of these units from the same
  run in `Float64` after ten steps, about 1.5 per step. A wrong
  coefficient is far outside it: the two tableaus differ by 41349.
- **The test has teeth**, checked by mutation of `lincomb!`: a scalar
  loop in place of the broadcast fails every device testset with
  "Scalar indexing is disallowed", and `Float64` coefficients fail them
  with an `InvalidIRError` naming `Float64`.

**Host allocations of a step** (measured, not asserted zero). Each
broadcast on Metal is a kernel launch, which allocates on the host. At
steady state (after three steps; the least over five more):

| | launches per step | Julia 1.13 | Julia 1.10 |
|---|---|---|---|
| SSP2(2,2,2) | 13 | 22 656 B (1.74 KB per launch) | 36 880 B (2.84 KB) |
| SSP3(4,3,3) | 23 | 42 112 B (1.83 KB per launch) | 68 640 B (2.98 KB) |
| one `a .= b`, for scale | 1 | 1 280 B | 2 352 B |

- The launches are the plan's combinations (`u★`, the copy into `U`, the
  increment, the update) plus one per callback call, limiters included.
  A launch with more operands captures more, hence above `a .= b`.
- At 256 times the state, 2²⁰ cells, a step allocates the same, 22 656 B
  and 41 760 B on 1.13, so nothing is state-sized. The test asserts that
  (to 25%); a host copy of that state alone would be 4 MiB.

**Metal compiles per state size** (measured in step 4). Metal specializes
a broadcast kernel on the array's shape once it has launched that shape
more than ten times (`BROADCAST_SPECIALIZATION_THRESHOLD` in its
`broadcast.jl`). So the second step at a new state length compiles every
kernel of the step again: for SSP2(2,2,2) on 1.13, 0.90–1.12 s and about
253 MB of host allocation, at 65 536 and at 100 000 cells after 4096; a
length seen before costs nothing. After that, enqueueing a step takes
40–50 µs of host time at 4096 cells. For TreeGRRMHD, where every regrid
changes the state length, that is about a second per new length on
Metal, from Metal's broadcast and not from this package.

## Open questions

For the package design:

- Where a TreeAMR state vector's ownership partition comes from
  ([Stage arithmetic](#stage-arithmetic-decided)).
- Confirm SSP2(3,3,2) against Pareschi & Russo (2005) (open, added in
  step 1). Its coefficients are the step-1 reviewer's recollection of the
  paper, verified only by the order conditions, `R(∞) = 0` and the SSP
  coefficient. See "Cross-checks" under [Tableaus](#tableaus). Step 3's
  observed orders, stiff-limit behaviour and TVD threshold agree with
  them, which is evidence, not the check against the paper (amended in
  step 3).

Deferred:

- **Scope beyond fixed steps.** Embedded error estimates (the IMEX-SSP
  schemes have none), dense output, low-storage forms and multirate are
  out of scope unless a use case appears.
- **Using a stiffly accurate previous stage.** For a stiffly accurate
  implicit part, the last stage's tendency is `g(uⁿ⁺¹)`, which could make
  ESDIRK-type tableaus admissible. It is invalid once a step limiter has
  changed `u`. Not planned.

## References

- L. Pareschi and G. Russo, *Implicit–explicit Runge–Kutta schemes and
  applications to hyperbolic systems with relaxation*, J. Sci. Comput.
  25 (2005) 129–155.
- U. M. Ascher, S. J. Ruuth and R. J. Spiteri, *Implicit–explicit
  Runge–Kutta methods for time-dependent partial differential
  equations*, Appl. Numer. Math. 25 (1997) 151–167.
- C. Palenzuela, L. Lehner, O. Reula and L. Rezzolla, *Beyond ideal MHD:
  towards a more realistic modelling of relativistic astrophysical
  plasmas*, MNRAS 394 (2009) 1727–1740 — IMEX-SSP for resistive MHD.
- SciML/OrdinaryDiffEq.jl#2065 (the IMEX-SSP request), #3704, #3705
  (the tableaus), #4620 (the abscissa bug).
- ClimaTimeSteppers.jl, `src/solvers/imex_ssprk.jl` and
  `src/solvers/imex_ark.jl`.

# IMEXRungeKutta design

This is the design document: what the package is, what has been decided,
and why. Where the implementation shows it wrong or incomplete, amend it
and say so ("(amended in step N)", "(measured in step N)"). Each design
item is marked **(decided)**, **(proposed)** or **(open)**.

**Status (2026-09-24):** the [package design](#package-design) is
complete, except where the partition for TreeAMR state vectors comes
from (open). The implementation plan is `PLAN.md`. Steps 0 and 1, the
scaffolding and the tableaus, are done; step 2, the integrator, is next.

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
- **The tableaus:** SSP2(2,2,2), SSP2(3,2,2), SSP3(3,3,2) and SSP3(4,3,3)
  (Pareschi & Russo 2005), and ARS(2,2,2) and ARS(4,4,3) (Ascher, Ruuth &
  Spiteri 1997). SSP3(4,3,3) is the intended production scheme, and
  SSP2(2,2,2) the debugging one.
  - This said SSP2(3,3,2) (amended in step 1). The name decided for it,
    `IMEXSSP2322`, is SSPk(s,σ,p) = SSP2(3,2,2) by the naming rule in
    [Tableaus are values](#tableaus-are-values-decided). Both upstreams
    implement SSP2(3,2,2) under that name (OrdinaryDiffEqSDIRK's
    `IMEXSSP2322`, ClimaTimeSteppers' `SSP322`). So that is the one here
    (proposed in step 1).
  - If the requirement meant a scheme of the paper's that is really
    named SSP2(3,3,2), that one is not included. Neither upstream
    implements one, and without the paper at hand its coefficients could
    not be checked.
- **A stage limiter hook** and a step limiter hook. These are
  positivity- or atmosphere-type resets of the state, as in the SSPRK
  methods of OrdinaryDiffEqSSPRK.
- **Fixed `Δt`.** The caller chooses the step, typically from a CFL
  condition, and restarts a fresh integrator after a regrid.
- **Generic arrays.** Stage arithmetic is by broadcasting over
  `similar(u0)` arrays, so device arrays work. The package must not
  require a particular array type. A faster path for a CPU `Array` is
  allowed alongside (amended 2026-09-24, see
  [Stage arithmetic](#stage-arithmetic-decided-details-proposed)).
- **Julia 1.10 floor**, generic in the scalar type `T` (Float32 must
  work).
- **Minimal dependencies.** Only CommonSolve at run time (amended
  2026-09-24; this was "at most StaticArrays", with SciMLBase open). See
  [Dependencies and names](#dependencies-and-names-decided). It brings
  PrecompileTools and Preferences with it (measured in step 0). Heavier
  packages (OrdinaryDiffEqSDIRK) are test-only.

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

**Exact coefficients.** Only SSP2(3,2,2) and ARS(4,4,3) are rational
(SSP2(3,3,2) before, amended in step 1).
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
are the alternative if that matters. Its parameters are exactly those
that make `R(∞) = 0` (above). Its implicit part is also A-stable, and so
L-stable. That is computed, not quoted (measured in step 1): all three
nonzero coefficients of its E-polynomial are positive (below).
Everything else is computed, not quoted.

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

| | SSP2(2,2,2) | SSP2(3,2,2) | SSP3(3,3,2) | SSP3(4,3,3) | ARS(2,2,2) | ARS(4,4,3) |
|---|---|---|---|---|---|---|
| held as | BigFloat | rational | BigFloat | BigFloat | BigFloat | rational |
| order | 2 | 2 | 2 | 3 | 2 | 3 |
| worst residual up to it | 0 | 0 (exact) | 0 | 0 | 8.6e−78 | 0 (exact) |
| next order misses by | 0.167 | 0.167 | 0.069 | 0.083 | 0.187 | 0.097 |
| stiffly accurate, implicit / explicit | no / no | yes / no | no / no | no / no | yes / yes | yes / yes |
| `R(∞)` | 1.0e−76 | 0 (exact) | 8.6e−77 | −2.4e−76 | 0 | 0 (exact) |
| A-stable | yes | yes | yes | yes | yes | yes |
| so L-stable | yes | yes | yes | yes | yes | yes |
| SSP coefficient | 1 | 1 | 1 | 1 | 0 | 0 |
| solves (`a_kk ≠ 0`) | 1, 2 | 1–3 | 1–3 | 1–4 | 2, 3 | 2–5 |
| explicit-used | 1, 2 | 2, 3 | 1–3 | 2–4 | 1, 2 | 1–4 |
| implicit-used | 1, 2 | 1–3 | 1–3 | 1–4 | 2, 3 | 2–5 |
| scratch arrays | 5 | 6 | 7 | 8 | 5 | 9 |

The residuals and `R(∞)` of the BigFloat tableaus are 256-bit
round-off. The next-order miss is at order 3 for the second-order
schemes. For the third-order ones it is in a classical order-4
condition of one part.

The E-polynomials, in `y`. Every coefficient is non-negative, which is
sufficient for `E ≥ 0`, and the diagonals are positive. The test helper
refuses to decide a tableau with a negative coefficient, rather than pass
it (proposed in step 1):
- SSP2(2,2,2) and ARS(2,2,2): `γ⁴y⁴` (= 0.00736 y⁴), with
  `γ = 1 − 1/√2`. Both implicit parts have the same `R`.
- SSP2(3,2,2): `y⁴/8 + y⁶/64`.
- SSP3(3,3,2): `0.00736 y⁴ + 0.000631 y⁶`.
- SSP3(4,3,3): `0.00545 y⁴ + 0.000282 y⁶ + α⁸y⁸` (α⁸ = 1.16e−5).
- ARS(4,4,3): `y⁴/24 + 5y⁶/144 + y⁸/256`.

In each, the `y²` coefficient vanishes, as order 2 requires: exactly
for the rational tableaus, and to 256-bit round-off (about 1e−77) for
the others.

The explicit parts of the IMEX-SSP schemes are Heun's method and the
three-stage SSPRK(3,3) of Shu & Osher. Their SSP coefficient is the
optimal 1. The ARS explicit parts have negative coefficients, and so
coefficient 0: `δ = −1/√2` in ARS(2,2,2), and `ã₄₂ = −5/6` and
`ã₅₄ = b̃₄ = −7/4` in ARS(4,4,3).
Step 3 measures what that means for TVD advection.

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
`src/solvers/imex_tableaus.jl`, fetched 2026-09-24). The papers were not
at hand. Where the citations in `src/tableaus.jl` give a table number,
it is OrdinaryDiffEqSDIRK's, and they say so.
- SSP2(2,2,2), SSP2(3,2,2) (`SSP322` there), SSP3(3,3,2) (`SSP332`) and
  ARS(2,2,2) agree with both, coefficient for coefficient.
- SSP3(4,3,3) (`SSP433`) agrees with both, up to their 14-digit
  `α, β, η`, which are the closed form rounded (a test).
- **ARS(4,4,3) disagrees in `b̃`.** ClimaTimeSteppers takes `b̃` as the
  last row of `Ã`, `(1/4, 7/4, 3/4, −7/4, 0)`, so the explicit part is
  stiffly accurate and uses four stages, the "4 explicit stages" of the
  name. OrdinaryDiffEqSDIRK has `b̃ = b = (0, 3/2, −3/2, 1/2, 1/2)`.
  - That is also third order, exactly, and it misses the classical
    order-4 conditions by up to 0.076 (a test).
  - But it reads the explicit tendency of stage 5, so it makes five
    explicit evaluations per step, not four.
  - Here `b̃` is the last row of `Ã`, as in ClimaTimeSteppers and as
    the name requires (proposed in step 1). It should be confirmed
    against the paper, §2.8.
  - Step 3's oracle comparison of ARS(4,4,3) must therefore compare with
    `IMEXTableau("…", Ã, b, A, b)`, built from `ARS443()`'s parts, not
    with `ARS443()`.

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

**The compat bound** (proposed in step 0) is `CommonSolve = "0.2.14"`,
that is `[0.2.14, 0.3)`: the current 0.2 series, from the one version
the suite has run against. A cap below 0.2.14 would avoid the two
transitive packages, but it would hold every environment that loads this
package back from CommonSolve's later releases. SciMLBase 3.56 bounds
CommonSolve by `0.2.4 - 0.2`, so either choice resolves beside it.

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

### The callback contracts (decided)

- **`f_exp!(du, u, p, t)`** writes all of `du` and does not change `u`.
  At a trivial first stage, `u` is `integ.u` itself.
- **`solve_imp!(U, u★, γΔt, p, t)`** writes `U` so that
  `U = u★ + γΔt g(U, t)`.
  - `U` and `u★` are distinct arrays, and `u★` must not be changed.
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
- **No accumulated time.** `tⁿ = t0 + n Δt` is computed afresh at each
  step, not accumulated, and the last step sets `t = t1` exactly. `step!`
  after the last step throws an `ArgumentError`.
- **Two types.** The time type is that of `t0`, `t1` and `dt` after
  promotion. The arithmetic type is `T = real(eltype(u0))`, so a complex
  state works (the order tests use `u′ = iu − u`). Coefficients are
  converted to `T`, and abscissae to the time type. A `Float32` state
  with `Float64` time is allowed.

### Tableaus are values (decided)

- **`IMEXTableau{R}`** holds a name, `Ã`, `b̃`, `A` and `b`, with
  `R = Rational{BigInt}` or `BigFloat`.
- **The constructor checks** that the parts are square and of equal
  size, that `Ã` is strictly lower triangular and `A` lower triangular,
  and the [admissibility](#tableaus) condition. Each failure is an
  `ArgumentError` that says why.
- **Named constructors** (decided): `IMEXSSP222()`, `IMEXSSP2322()`,
  `IMEXSSP3332()`, `IMEXSSP3433()`, `ARS222()` and `ARS443()`.
  - These are OrdinaryDiffEq's names, so an oracle test reads as a
    comparison of like with like.
  - `IMEXSSPksσp` is Pareschi–Russo's SSPk(s,σ,p). The short
    `IMEXSSP222` is SSP2(2,2,2).
  - They clash with OrdinaryDiffEqSDIRK's exports, so the tests
    `import` it and qualify its names.
  - They are functions returning an `IMEXTableau`, not constants,
    because a `BigFloat` does not survive precompilation reliably.
- **A caller's own tableau** goes through the same constructor.
- **Properties are computed in the tests.** The order conditions,
  L-stability and the SSP coefficient are not package API.

What step 1 settled (proposed in step 1):
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
- **`IMEXTableau` is exported** beside the six names, for a caller's own
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
function barrier. `step!` is type-stable and allocation-free, and it is
unrolled over the stages.

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
- one more if some solving stage is not implicit-used.

For SSP3(4,3,3) that is 1 + 4 + 3 = 8 arrays, besides `integ.u`.

### Stage arithmetic (decided, details proposed)

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
- **By owner, for a CPU `Array` with more than one thread.** The caller
  passes `partition`, a collection of `Threads.nthreads()` elements.
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

**Rejected:**
- **Polyester** (`RK4(thread = True())`). TreeWave measured it as worse,
  because it starts a second thread pool.
- **KernelAbstractions.** It would be a dependency or an extension. Its
  default CPU schedule is exactly the affinity-losing launch that
  TreeAMR measured, and its static schedule refuses to nest.
- **A caller-supplied kernel.** It would push the bandwidth question onto
  every caller.

The combination sits behind one internal function, so the owner path
can come in a later step without changing the interface.

### File layout (decided)

- `src/IMEXRungeKutta.jl`: the module and its exports.
- `src/tableau.jl`: `IMEXTableau`, its checks, and the conversion to `T`.
- `src/tableaus.jl`: the six tableaus, in closed form.
- `src/plan.jl`: the stage plan.
- `src/lincomb.jl`: fused linear combinations, broadcast and threaded.
- `src/integrator.jl`: `IMEXProblem`, `init`, `step!` and `solve!`.
- `test/`: one file per group under [Testing](#testing-decided).
  `test/runtests.jl` includes them into one `@testset`;
  `test/scaffold_tests.jl` checks that the package loads, that its four
  names are CommonSolve's bindings, and that `[deps]` is CommonSolve
  alone (amended in step 0). `test/tableau_properties.jl` holds the
  test-only tableau properties, and `test/tableau_tests.jl` asserts them
  (amended in step 1).
- Test-only dependencies are `[extras]` and `[targets]` in the root
  `Project.toml`, not a `test/Project.toml` (proposed in step 0). That
  is what `PLAN.md` specifies, and it keeps one file to read for the
  whole dependency picture. TOML is among them, for the `[deps]` check.
- `.github/workflows/CI.yml` (proposed in step 0) has four cells: Julia
  1.10 on Linux; the current release on Linux, with
  `--check-bounds=yes` and coverage; the current release on macOS; and
  the current release on Linux at four threads. Every cell but the
  bounds-checked one runs with `--check-bounds=auto`, so that the
  allocation tests run on the floor and at four threads.
  `julia-runtest`'s default, `yes`, would skip them in every cell.

### Documentation (decided)

README and docstrings, no Documenter site for now. Docstrings are
prose-first and point at this document. The README has a worked
example, including a stage solver. A site can be added later without
changing anything else.

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
reference step to 1e−16 on a linear problem with default settings. Two
restrictions apply:
- the state must be real (its default AD Jacobian rejects a complex
  state);
- `f` must not depend on `t`, until #4620 is fixed;
- its `ARS443` has `b̃ = b`, where this package has the last row of `Ã`
  (amended in step 1; see "Cross-checks" under [Tableaus](#tableaus)). So
  the comparison for ARS(4,4,3) is against a tableau built with upstream's
  `b̃`.

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
  - a Float32 run works;
  - a device smoke run passes on Metal, gated by
    `IMEXRUNGEKUTTA_TEST_METAL=1`. It is a short `Float32` run on an
    `MtlArray` state with scalar indexing disallowed, so it covers the
    broadcast path and checks that nothing indexes the state. How Metal
    enters the test environment without being installed everywhere is
    settled in the plan.
- **Order:**
  - on the split linear ODE `u′ = iu − u`, the observed order equals the
    tableau's, ±0.1;
  - likewise on `u′ = −u + cos t` (implicit `−u`, explicit `cos t`),
    which catches a mistimed explicit stage (the #4620 failure mode) and
    is invisible to a problem where `f` does not depend on `t`.
- **Stiff limit:** on the Kaps problem with `ε ∈ {1, 1e−3, 1e−6, 1e−9}`,
  the observed order per `ε` is recorded, including any order reduction.
- **Asymptotic preservation:** at `ε = 1e−12`, one step lands on the
  equilibrium manifold to `O(ε)`.
- **SSP:** upwind advection with stiff relaxation keeps the total
  variation non-increasing for `Δt ≤ C Δt_FE`; the measured `C` is
  recorded against the explicit part's SSP coefficient.
- **Oracle:** each tableau OrdinaryDiffEqSDIRK has matches it to 1e−12
  over ten steps, within the restrictions above.

## Open questions

For the package design:

- Where a TreeAMR state vector's ownership partition comes from
  ([Stage arithmetic](#stage-arithmetic-decided-details-proposed)).

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

# IMEXRungeKutta design

This is the design document: what the package is, what has been decided,
and why. Where the implementation shows it wrong or incomplete, amend it
and say so ("(amended in step N)", "(measured in step N)"). Each design
item is marked **(decided)**, **(proposed)** or **(open)**.

**Status (2026-09-24):** requirements and background only. The package
design (API, types, file layout, the implementation plan) is next.

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
- **The tableaus:** SSP2(2,2,2), SSP2(3,3,2), SSP3(3,3,2) and SSP3(4,3,3)
  (Pareschi & Russo 2005), and ARS(2,2,2) and ARS(4,4,3) (Ascher, Ruuth &
  Spiteri 1997). SSP3(4,3,3) is the intended production scheme, and
  SSP2(2,2,2) the debugging one.
- **A stage limiter hook** and a step limiter hook. These are
  positivity- or atmosphere-type resets of the state, as in the SSPRK
  methods of OrdinaryDiffEqSSPRK.
- **Fixed `Δt`.** The caller chooses the step, typically from a CFL
  condition, and restarts a fresh integrator after a regrid.
- **Generic arrays.** Stage arithmetic is by broadcasting over
  `similar(u0)` arrays, so device arrays work. The package must not
  require a particular array type.
- **Julia 1.10 floor**, generic in the scalar type `T` (Float32 must
  work).
- **Minimal dependencies.** At most StaticArrays at run time. SciMLBase
  is (open), see [Open questions](#open-questions). Heavier packages
  (OrdinaryDiffEqSDIRK) are test-only.

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

**Admissibility (proposed).** Because `g` is never evaluated, a stage
with `a_kk = 0` has no implicit tendency. That is consistent only if
column `k` of `A` and `b_k` are zero. The IMEX-SSP and ARS schemes satisfy
this. ESDIRK-type additive schemes (KenCarp and the like) do not, since
their first column needs `g(uⁿ)`. The tableau constructor checks it and
throws an `ArgumentError` that says why.

**Exact coefficients.** Only SSP2(3,3,2) and ARS(4,4,3) are rational.
SSP2(2,2,2), SSP3(3,3,2) and ARS(2,2,2) involve `√2`, in closed form
(`γ = 1 − 1/√2`, and so on). SSP3(4,3,3)'s `α, β, η` are printed to 14
digits only (α = 0.24169426078821, β = 0.06042356519705,
η = 0.12915286960590; β = α/4 to those digits). Both OrdinaryDiffEqSDIRK
and ClimaTimeSteppers use exactly those 14 digits. So (proposed):
- every tableau is held in extended precision (e.g. 256-bit `BigFloat`)
  and converted to `T` once, when the integrator is built;
- the rational ones are also held exactly;
- SSP3(4,3,3)'s parameters are computed from the conditions in the paper
  that fix them. If the paper does not fix them, the printed digits are
  the definition, and that is recorded.

**Properties to compute and record per tableau (proposed):**
- the order conditions up to order 3, including the IMEX coupling
  conditions;
- stiff accuracy (`b` equal to the last row of `A`);
- L-stability of the implicit part;
- the SSP coefficient of the explicit part.

SSP3(4,3,3) is **not** stiffly accurate: `b = (0, 1/6, 1/6, 2/3)`, while
the last row of `A` is `(β, η, 1/2 − β − η − α, α)`. So its order may
drop in the stiff limit, and the ARS schemes, which are stiffly accurate,
are the alternative if that matters. Its implicit part is expected to be
L-stable (Pareschi & Russo 2005). Everything else is computed, not
quoted.

### One step (proposed)

For stages `k = 1, …, s`:

1. `u★ = uⁿ + Δt Σ_{j<k} (ã_kj k̃_j + a_kj k_j)`.
2. If `a_kk ≠ 0`: `solve_imp!(U_k, u★, a_kk Δt, p, tⁿ + c_k Δt)`, then
   `k_k = (U_k − u★)/(a_kk Δt)`. Otherwise `U_k = u★`.
3. `stage_limiter!(U_k, p, tⁿ + c̃_k Δt)`.
4. `f_exp!(k̃_k, U_k, p, tⁿ + c̃_k Δt)`, **only if** column `k` of `Ã`
   or `b̃_k` is nonzero.

Then `uⁿ⁺¹ = uⁿ + Δt Σ_j (b̃_j k̃_j + b_j k_j)`, followed by
`step_limiter!`.

Three consequences of the stage contract, each one a decision:

- **The tendency is taken before the limiter** (decided). A limiter's
  correction is a reset, not a tendency. Folded into `k_k`, it would be
  re-weighted by `a_jk/a_kk` in later stages and by `b_k/a_kk` in the
  update. ClimaTimeSteppers does fold it in: its `constrain_state!` runs
  before its `(U − temp)/dtγ`.
- **One call per implicit stage** (decided). Convergence, fallbacks,
  flags and counters belong to the user's solver. The integrator has no
  nonlinear-solver loop, tolerance or retry.
- **Untouched components stay untouched.** Where `solve_imp!` copies a
  component of `u★` into `U`, its tendency is exactly zero, and the
  component evolves by the explicit part alone, to the last bit. A
  conservative explicit scheme stays conservative. This is a test.

**Round-off in the recovered tendency.** `k_k` carries an absolute error
of about `ε|U|/(a_kk Δt)`, which enters the update multiplied by
`b_k Δt`. That makes it `O(ε|U| b_k/a_kk)` per step, independent of
`Δt`: harmless for these tableaus. A tableau with a tiny `a_kk` would be
a poor fit.

**Cost.** The explicit right-hand side is usually the expensive call.
Skipping zero columns, SSP3(4,3,3) makes three explicit evaluations per
step, not four; OrdinaryDiffEqSDIRK makes five (below). Implicit
tendencies are stored only for the columns of `A` and entries of `b` that
use them.

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

**Use as a test oracle** (proposed). Upstream agrees with a direct
reference step to 1e−16 on a linear problem with default settings. Two
restrictions apply:
- the state must be real (its default AD Jacobian rejects a complex
  state);
- `f` must not depend on `t`, until #4620 is fixed.

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

## Testing (proposed)

Testset names are claims, each opening with a comment that names the
failure mode it guards.

- **Tableaus:**
  - each meets its stated order and fails at the next, exactly for the
    rational forms and to a tiny residual in `BigFloat` for the others;
  - the triangularity and admissibility are as stated;
  - the recorded properties (stiff accuracy, L-stability, SSP
    coefficient) are regression-tested.
- **Mechanics:**
  - `step!` is allocation-free after warm-up;
  - `solve_imp!` is called once per implicit stage, with the documented
    arguments (a mock);
  - `f_exp!` is called once per nonzero column, at `tⁿ + c̃_k Δt` (a
    mock): three times per SSP3(4,3,3) step;
  - with `g ≡ 0`, the result equals the explicit RK method;
  - untouched components match the explicit-only run bitwise;
  - a Float32 run works;
  - a device smoke run passes, gated by an environment variable.
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

- **The API.** A sketch carried over from the requirements:

      prob  = IMEXProblem(f_exp!, solve_imp!, u0, (t0, t1), p;
                          stage_limiter! = nothing, step_limiter! = nothing)
      integ = init(prob, tableau; dt)   # dt adjusted to hit t1 in whole steps
      step!(integ); solve!(integ)       # integ.u, integ.t

  Open points:
  - whether to use SciMLBase's `init`/`solve!`/`step!` generics and
    problem types, or to own the names (SciMLBase is not light);
  - the limiter signatures: OrdinaryDiffEq passes the integrator,
    `(u, integrator, p, t)`, and the sketch passes `(u, p, t)`;
  - whether tableaus are values or types.
- **Stage arithmetic.** Plain broadcasting is serial on the CPU; a
  downstream code measured that serial integrator arithmetic caps a
  threaded step at 3.6×. The options are:
  - fused broadcasts only;
  - an optional threaded or KernelAbstractions path (a dependency);
  - letting the caller supply the linear-combination kernel.
- **Failure reporting.** The solver owns convergence, but a step may
  still need to report that a stage failed. Should `solve_imp!` return a
  status, and does the integrator count, stop, or ignore it?
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

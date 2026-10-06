# IMEXRungeKutta history

How the package got where it is: who decided what and when, what was
tried or rejected, measurements that later ones replaced, upstream bugs
since fixed, and the releases. The current design, and the reasons that
still hold, are in [`CODE.md`](CODE.md); this file follows its sections.

**The steps** are those of the implementation plan, `PLAN.md`, deleted
when it was complete:
- step 0, the scaffolding;
- step 1, the tableaus;
- step 2, the integrator on the broadcast path;
- step 3, the validation;
- step 4, the Metal smoke run;
- step 5, the stage arithmetic by owner;
- step 6, the review pass.

## Contents

- [The implementation plan](#the-implementation-plan)
- [Requirements](#requirements)
- [The method](#the-method)
  - [Tableaus](#tableaus)
  - [Explicit tableaus](#explicit-tableaus)
  - [Implicit Euler](#implicit-euler)
  - [One step](#one-step)
  - [Limiters in other codes](#limiters-in-other-codes)
- [Package design](#package-design)
  - [Dependencies and names](#dependencies-and-names)
  - [The interface](#the-interface)
  - [The callback contracts](#the-callback-contracts)
  - [Failures and exceptions](#failures-and-exceptions)
  - [Time and the step count](#time-and-the-step-count)
  - [Tableaus are values](#tableaus-are-values)
  - [The stage plan and storage](#the-stage-plan-and-storage)
  - [Scratch reuse](#scratch-reuse)
  - [Stage arithmetic](#stage-arithmetic)
  - [By owner, as built](#by-owner-as-built)
  - [File layout](#file-layout)
  - [Documentation](#documentation)
- [Why not an existing package](#why-not-an-existing-package)
  - [OrdinaryDiffEq](#ordinarydiffeq)
- [Testing](#testing)
  - [The test environment](#the-test-environment)
- [Validation](#validation)
  - [The oracle](#the-oracle)
- [On a device](#on-a-device)
- [Releases](#releases)
- [TreeGRRMHD](#treegrrmhd)

## The implementation plan

- The package design was drafted and reviewed on 2026-09-24.
- **The implementation plan is complete** (2026-09-24). Its steps 0–6
  built the package design: the tableaus, the integrator on the broadcast
  path, the validation, the Metal smoke run, the stage arithmetic by
  owner and a review pass. `PLAN.md` was deleted in step 6.
- Each design item of `CODE.md` was marked **(decided)**, **(proposed)**
  or **(open)**, and a decision that a step proposed and Erik then took
  kept its history, as "(proposed in step N, decided 2026-09-24)". Those
  markers now live here, and `CODE.md` marks only what is open or
  proposed.
- Erik decided every proposal the steps made on 2026-09-24. The last one,
  fresh tasks rather than persistent workers, was decided by the Symmetry
  run he asked for ([By owner, as built](#by-owner-as-built)).
- The documents were reorganized into `README.md`, `CLAUDE.md`,
  `CODE.md` and this file on 2026-10-06.

## Requirements

- The requirements were decided before the package existed.
- **The tableaus.** The list first had SSP2(3,3,2) alone, as the scheme
  named `IMEXSSP2322`. In step 1 Erik decided to have both:
  `IMEXSSP2322` is SSP2(3,2,2), by the naming rule (decided), and
  `IMEXSSP2332` is SSP2(3,3,2) (decided).
- **The three purely explicit tableaus**, `Euler()`, `RK4()` and
  `SSPRK33()`, were added on 2026-09-25, at Erik's request.
- **Butcher62, CooperVerner8 and ImplicitEuler** were added on
  2026-10-05, at Erik's request.
- **The stage limiter hook.** That what a stage limiter's correction
  reaches differs from OrdinaryDiffEqSSPRK's was added on 2026-09-25.
- **Generic arrays.** The faster path for a CPU `Array`, alongside the
  broadcast, was allowed on 2026-09-24; it is `partition` since step 5.
  That a `Float32` `MtlArray` state runs on Metal with scalar indexing
  disallowed and agrees with the CPU bitwise was measured in step 4.
- **MultiFloats' double-floats** were added to the requirements on
  2026-09-28, at Erik's request.
- **Minimal dependencies.** This read "at most StaticArrays", with
  SciMLBase open, until 2026-09-24, when it became "only CommonSolve at
  run time". That CommonSolve brings PrecompileTools and Preferences was
  measured in step 0. Heavier packages moved to `test/Project.toml` in
  step 6.
- **The size of the test environment.** With OrdinaryDiffEqSDIRK, 142
  packages on Julia 1.13 and 138 on 1.10, standard libraries included
  (measured in step 3). With the explicit tableaus' oracles, 144 and 140
  (measured 2026-09-25). With MultiFloats, 148 on 1.13, and the Metal
  environment 103 on 1.13 and 101 on 1.10 (measured 2026-09-28).

## The method

### Tableaus

- That only SSP2(3,2,2), SSP2(3,3,2) and ARS(4,4,3) are rational was
  amended in step 1.
- SSP3(4,3,3)'s closed form was computed symbolically on 2026-09-24. Its
  order-condition residuals with the printed 14 digits, and `R(∞)`, were
  measured in step 1.
- The holding of the coefficients, rational exactly and otherwise as
  256-bit `BigFloat`, converted to `T` once, and the properties to
  compute and record per tableau, were marked decided.
- That SSP3(4,3,3) drops from order 3 to 2 in the stiff component was
  measured in step 3; that its implicit part is A-stable, and so
  L-stable, was computed in step 1.
- **Where a step ends in the stiff limit** was measured in step 2. Step 3
  measured it per tableau, and found its second item ("It is 0 for the
  four schemes with a stiffly accurate implicit part") incomplete: `wᵀ𝟙 =
  0` is not `w = 0`.
- **The measured properties** were measured in step 1. The SSP2(3,3,2)
  column of their table was added in step 1, after Erik's decision to
  have both SSP2 schemes.
- **The E-polynomials.** That the test helper refuses to decide a
  tableau with a negative coefficient, rather than pass it, was proposed
  in step 1 and decided 2026-09-24.
- What the ARS explicit parts' coefficient 0 means for TVD advection was
  measured in step 3.
- **What the order conditions do not see** was measured in step 1.
- **Cross-checks.** The comparison with OrdinaryDiffEqSDIRK 2.9.6 and
  ClimaTimeSteppers was made in step 1. That OrdinaryDiffEqSDIRK 2.9.7's
  `imex_tableaus.jl` differs from 2.9.6's only in `ARS443`'s `b̃` was
  added on 2026-10-05.
  - The check of the five Pareschi–Russo schemes against the paper itself
    was added on 2026-09-24.
  - SSP2(3,3,2)'s coefficients were first the step-1 reviewer's
    (Claude's) recollection of the paper. Whether they were right was
    open until 2026-09-24, when they were checked against it: they are
    Table 4 of arXiv:1009.2757 exactly. Step 3's measurements agree.
  - **ARS(4,4,3)'s `b̃`.** That `b̃` is the paper's was decided: Erik
    checked the paper, and so did step 1. Erik reported OrdinaryDiffEqSDIRK
    2.9.6's `b̃ = b` upstream, to SciML/OrdinaryDiffEq.jl, on 2026-09-24.
    Step 3's oracle comparison of ARS(4,4,3) therefore compared with
    `IMEXTableau("…", Ã, b, A, b)`, built from `ARS443()`'s parts, not
    with `ARS443()`. 2.9.7, resolved on 2026-10-05, has the paper's `b̃`
    and agrees with `ARS443()` to 2.8e−16 over the oracle's ten steps;
    from then on the oracle compares with `ARS443()` itself, and the
    `b̃ = b` variant remains as a third-order method of its own, named for
    2.9.6. What the difference amounts to was measured in step 3.

### Explicit tableaus

- The explicit tableaus were decided on 2026-09-25: `Euler`, `RK4` and
  `SSPRK33` then, and `Butcher62` and `CooperVerner8` added on
  2026-10-05, with their cross-checks against OrdinaryDiffEqExplicitTableaus
  and their total-variation threshold (measured 2026-10-05).
- **The classical order conditions** were a hand-written list, complete
  to order 4, until 2026-10-05, when they came to be generated from the
  rooted trees, one per tree. RK4's miss was recorded as 1/120, the bushy
  tree's `b̃ᵀc̃⁴ − 1/5`, while the list had only that condition at order
  5; the largest of all nine is 1/80.
- That only the step limiter limits an explicit tableau's first stage
  was marked decided.

### Implicit Euler

- Decided on 2026-10-05. Erik chose backward Euler over the IMEX
  forward–backward Euler, ARS(1,1,1), on 2026-10-05. Its numbers were
  measured on 2026-10-05.

### One step

- The stage loop, increments rather than tendencies, a trivial first
  stage read as `uⁿ`, the stage limiter acting only where `f_exp!` reads,
  the tendency taken before the limiter and one call per implicit stage
  were each marked decided.
- **Where a stage limiter's correction goes** was amended on 2026-09-25.
  The section said that OrdinaryDiffEq's SSPRK methods "limit exactly
  what `f` reads"; that is true of where they limit, not of where the
  correction goes. TreeHydro measured the weights in its step 9.
- What folding a correction into the tendency would cost was computed
  on 2026-09-25 from `src/tableaus.jl`.

### Limiters in other codes

- The survey was made on 2026-09-25.

## Package design

### Dependencies and names

- Depending on CommonSolve amended the "at most StaticArrays"
  requirement, on 2026-09-24.
- **CommonSolve's own dependencies** were measured in step 0. The section
  had said CommonSolve has none; that held up to 0.2.13. 0.2.14, which
  brings PrecompileTools and Preferences, was the current version on
  2026-09-24.
- **The compat bound** `CommonSolve = "0.2.14"` was proposed in step 0
  and decided 2026-09-24.

### The interface

- The interface, and that the caller may change `integ.u` in place
  between steps, were marked decided.
- `reuse` was added on 2026-09-26.
- Step 2 settled the integrator type, `step!`'s return value, `solve` as
  a method of our own, the required `dt` and `init`'s refusals (proposed
  in step 2, decided 2026-09-24). That CommonSolve 0.2.14 has the same
  generic `solve` fallback was checked in step 2.
- **`init`'s refusals.** Until step 5, `init` refused any `partition`
  other than `nothing`, as not implemented yet; since step 5 it refuses
  a partition for a state that is not a CPU `Array`, and a malformed
  one. The refusal of a `tspan` and `dt` that promote to no concrete
  float was added on 2026-09-28. The refusal of a `dt` that is not a real
  number has been in the code since step 2; step 6 recorded it and added
  a test.
- **The public API.** Version 1.0.0 was tagged on 2026-09-24
  ([Releases](#releases)); since then the interface is under semantic
  versioning.

### The callback contracts

- The contracts were marked decided, among them that `U` enters as a
  copy of `u★` and the limiters' signature.
- That `f_exp!` may be `nothing` for `ImplicitEuler()` was added on
  2026-10-05; that `solve_imp!` may be `nothing` for the explicit
  tableaus, on 2026-09-25.
- That the integrator passes `integ.u` itself as `u★` at an empty row
  was amended in step 2.
- That `u★` may be inadmissible, and that the stage limiter's correction
  reaches the result only as the step limiter, were amended on
  2026-09-25.

### Failures and exceptions

- Marked decided. "There is no status" resolved an earlier open question
  about a status the integrator would return.

### Time and the step count

- Marked decided. **The tolerance** of the step count was proposed in
  step 2 and decided 2026-09-24; the chunk sweep was measured in step 2.
  That `Δt ≤ dt` holds only up to that tolerance, and that the time type
  is made `float`, were amended in step 2.
- **Software floats** were added on 2026-09-28. Until then `Int(n)` of
  the step count had no method for a MultiFloat time, so `init` refused
  every MultiFloat time.

### Tableaus are values

- Marked decided, with the named constructors. `IMEXSSP2332()` was
  added in step 1, the explicit `Euler()`, `RK4()` and `SSPRK33()` on
  2026-09-25, and `Butcher62()`, `CooperVerner8()` and `ImplicitEuler()`
  on 2026-10-05.
- Step 1 settled the fields, `R`, the two more refusals, the export, the
  internal functions for step 2's plan and what `coefficients` holds
  (proposed in step 1, decided 2026-09-24).
- How a tableau prints was recorded in step 6, which added the test.
- The double-float rounding of the coefficients was measured on
  2026-09-28.
- "Tableaus are values" resolved the open question "values or types".

### The stage plan and storage

- Marked decided. The broadcast path's `@inferred step!`, its zero
  allocations and the 6.1 ms SSP3(4,3,3) step were measured in step 2.
  The by-owner bound was added in step 5.
- That scratch taken over with `reuse` is not written again was added on
  2026-09-26. That `U` is needed only for a stage with a nonempty row
  (explicit Euler has none) was amended on 2026-09-25.
- **An empty row forms no `u★`** was proposed in step 2 and decided
  2026-09-24. It amended the last item of the scratch count, which read
  "if some solving stage is not implicit-used", and `scratch_count` with
  it. None of the seven named tableaus then had such a stage, so their
  counts were unchanged.
- Step 2 settled the plan's layout, the exact pattern, the explicit
  coefficients, the terms' order, dead stages, first touch and the
  plan's self-check (proposed in step 2, decided 2026-09-24).
- That `integ.u` is first-touched too was proposed in step 5 and decided
  2026-09-24.

### Scratch reuse

- Decided on 2026-09-26, from TreeGeneralizedHarmonic's measurement on
  Symmetry of the same day. Refusing a misfit, over a silent fallback to
  fresh arrays, was Erik's decision. What it saves was measured on
  2026-09-26.

### Stage arithmetic

- Marked decided. The broadcast path was the first to be implemented;
  that one thread is accepted by owner too was amended in step 5.
- `block_partition` was proposed in step 5 and decided 2026-09-24.
- The internal combination function was `lincomb!` from step 2, with the
  owner path's methods to come in step 5 (proposed in step 2, decided
  2026-09-24). Step 5 added `lincomb_copy!`; on the broadcast path it is
  exactly the two broadcasts of step 2, so that path was unchanged.

### By owner, as built

- Built and measured in step 5. The keyword, one thread as a plain loop,
  the bench files and the test file's place were proposed in step 5 and
  decided 2026-09-24.
- **Fresh tasks, not persistent workers.** Proposed in step 5. `PLAN.md`
  asked for persistent sticky workers instead if they reach zero
  allocations at no loss in speed; the prototype in
  `bench/stage_arithmetic.jl` was written for that. Erik (2026-09-24):
  keep fresh tasks until the Symmetry run of
  `bench/symmetry_stage_arithmetic.sh` decides. It was decided by that
  run, job 563504 on 2026-09-24: persistent workers are no faster at any
  thread count.
- The explicit tableaus joined the bitwise-identity test on 2026-09-25.
- **Julia 1.10 needed one change** for the one-thread allocation claim:
  `check_lengths` raised its error inline, and building the message
  allocated 32 bytes per combination there even when nothing was wrong.
  The error has been raised by a `@noinline` function since.

### File layout

- Marked decided.
- `scaffold_tests.jl`'s check of `[deps]` was added in step 0,
  `tableau_properties.jl` and `tableau_tests.jl` in step 1, the mocks,
  the interface, mechanics, smoke-order and README tests in step 2, the
  validation files and `problems.jl` in step 3, `owner_tests.jl` in step
  5 (proposed in step 5, decided 2026-09-24), `reuse_tests.jl` on
  2026-09-26, `multifloat_tests.jl` on 2026-09-28, and `jin_xin_tests.jl`
  with `examples/jin_xin_2d.jl` on 2026-09-24, after the plan.
- **Test-only dependencies.** Step 0 had proposed `[extras]` and
  `[targets]` in the root `Project.toml`; in step 6 Erik moved the
  test-only dependencies to `test/Project.toml`, and the test environment
  has been that file since. OrdinaryDiffEqLowOrderRK and
  OrdinaryDiffEqSSPRK were added on 2026-09-25, and OrdinaryDiffEqSDIRK's
  bound was raised from `"2.9.6"` on 2026-10-05. That the test
  environment needs CommonSolve listed, and how `Pkg.test()` adds this
  package, were measured in step 6.
- **The device smoke run's environment** was proposed in step 4 and
  decided 2026-09-24.
- **CI** was proposed in step 0 and decided 2026-09-24, with four cells;
  in step 6 Erik added the fifth, Julia 1.10 at four threads. No Metal
  cell was proposed in step 4 and decided 2026-09-24. Running CI on a
  README-only push to `main` was proposed in step 6 and decided
  2026-09-24: until step 6 every `.md` was ignored, so a README-only
  change could break the suite unseen.

### Documentation

- Marked decided. CI's badge was proposed in step 4 and decided
  2026-09-24; Codecov's was added on 2026-09-24.
- The first Codecov uploads, on 2026-09-24, failed with "Token required -
  not valid tokenless upload" while the step stayed green; Erik then
  added the `CODECOV_TOKEN` secret and the badge.

## Why not an existing package

- The survey was made on 2026-09-24. Not contributing to OrdinaryDiffEq,
  and using it as a test oracle, were marked decided.

### OrdinaryDiffEq

- **#4620.** The mistimed last explicit stage was reported as
  SciML/OrdinaryDiffEq.jl#4620 on 2026-09-24, while open, and is fixed in
  OrdinaryDiffEqSDIRK 2.9.7, which a fresh `Pkg.test()` resolved on
  2026-10-05.
- **The oracle's restrictions until 2.9.7.** `f` had to be independent
  of `t` (#4620). That held only for the tableaus whose last explicit
  abscissa `c̃_s` is not 1, SSP3(3,3,2) and SSP3(4,3,3) (`c̃_s = 1/2`),
  since upstream's `t + Δt` is right where `c̃_s = 1` (amended in step
  3). With a `t`-dependent `f`, 2.9.6 agreed with the other four to
  3.1e−16 over ten steps, and differed from those two by 0.0225. And its
  `ARS443` had `b̃ = b`, where this package has the last row of `Ã`
  (amended in step 1), so the comparison for ARS(4,4,3) was against a
  tableau built with that `b̃`. 2.9.7 lifted both on 2026-10-05.
- That SSP2(3,3,2) has no oracle was amended in step 1, and its check
  against the paper on 2026-09-24.

## Testing

- The test groups were marked decided. Additions: in step 1, the
  perturbation, closed-form, precision, 14-digit and ARS(4,4,3)-variant
  tests; on 2026-09-25, the explicit tableaus' tests and mechanics; on
  2026-10-05, the rooted-tree conditions, Butcher62, CooperVerner8 and
  ImplicitEuler; in step 2, the corner tableaus, the call sequence, the
  stage limiter's reach and the allocation tests; in step 5, the by-owner
  items of `test/owner_tests.jl`; on 2026-09-26, `reuse`; on 2026-09-28,
  the double-floats and the `Float32x2` Metal run; in step 4, the Metal
  smoke run's environment of its own.
- **The order tests by mutation** were measured in step 2, which also
  added the remark that a swap of `c̃` and `c` is invisible to them.
- **Asymptotic preservation.** That the `O(ε)` claim holds only for the
  ARS schemes, and for SSP2(3,2,2) and SSP2(3,3,2) only when `f` does not
  depend on `t`, was amended in step 3.
- **SSP.** The relaxation toward the square wave, and `C` measured at
  three relaxations, were amended in step 3.
- **Oracle.** That SSP2(3,3,2) has no oracle was amended in step 1; the
  explicit three were added on 2026-09-25, and `ImplicitEuler` on
  2026-10-05. From step 3 to 2026-10-05, with a `t`-dependent `f`, the
  comparison matched where `c̃_s = 1` and was `@test_broken` where not
  (#4620), and our ARS(4,4,3) differed from upstream's by `O(Δt⁴)` per
  step. Against 2.9.7 (2026-10-05) every tableau matches.
- **A PDE**, the Jin–Xin example, was added on 2026-09-24, after the
  plan, and its numbers measured then.

### The test environment

Measured on the M3 in step 3, with the test environment of 142
packages on 1.13, from a fresh `JULIA_DEPOT_PATH`, so including the
downloads:
- 1.13.0: 238 s in all. `Pkg.instantiate()` of the package itself, with
  the registry, 49 s; then `Pkg.test()` 189 s, of which resolving and
  downloading the test environment about 10 s, precompiling its 156
  packages 116 s, and the suite 64 s.
- 1.10.12, `julia_args=["--check-bounds=auto"]`: 169 s in all.
  Instantiate 7 s; then `Pkg.test()` 162 s, of which about 5 s resolve and
  download, 110 s precompiling 153 packages, and the suite 47 s.
- With the depot warm and only `Manifest.toml` deleted, `Pkg.test()`
  takes the suite time plus 5–10 s. A `--check-bounds=yes` run
  precompiles the whole environment again for that flag the first time:
  267 s on 1.13 and 331 s on 1.10, of which the suite is 99 s and 129 s.
- The suite alone, at one thread: 64 s on 1.13 and 44–57 s on 1.10, of
  which `oracle_tests.jl` is 38–42 s and 28 s, almost all of it compiling
  upstream's six solvers. Under load (a load average of 14) the 1.13
  suite took 98 s. Every other file is under 11 s.

That on 1.10 `Pkg.test()` forces `--check-bounds=yes` on the test
process, ignores the parent's `--check-bounds=auto` and is overridden by
a test-process argument, and that `Pkg.test()` passes the parent's
`--threads` on, were measured in step 0.

## Validation

- Measured in step 3. How the tests measure, and the total-variation
  bisection, were proposed in step 3 and decided 2026-09-24. The SSP
  coefficients in the total-variation table are step 1's.
- The explicit tableaus' observed orders and total variation were
  measured on 2026-09-25; Butcher62's, CooperVerner8's and
  ImplicitEuler's on 2026-10-05.
- **The stiff limit.** The Kaps split was described as "split as
  `PLAN.md` says" until step 6.

### The oracle

- Upstream was OrdinaryDiffEqSDIRK 2.9.6 until 2026-10-05, and is 2.9.7
  since.
- Against 2.9.6 the `a = 1` column was 0.0225 for SSP3(3,3,2) and
  SSP3(4,3,3), the mistimed last explicit stage of #4620, and those two
  were `@test_broken`. Its ARS(4,4,3) was the `b̃ = b` variant, which
  agreed with a tableau built that way to 8.3e−17 and 3.1e−16, and
  differs from 2.9.7's, and from ours, by 2.5e−5 at `a = 0` and 2.5e−4 at
  `a = 1`.
- **The bound.** `test/Project.toml` added OrdinaryDiffEqSDIRK with the
  compat bound `"2.9.6"`, that is `[2.9.6, 3)` (proposed in step 3,
  decided 2026-09-24). A release that fixes #4620 turns the two
  `@test_broken` into unexpected passes, which fail the suite, and so is
  noticed. 2.9.7 did, and changed `ARS443`'s `b̃` too: a fresh
  `Pkg.test()` resolved it on 2026-10-05 and `oracle_tests.jl` failed,
  as intended. The bound became `"2.9.7"`, and the two `@test_broken`s
  plain `@test`s, every tableau compared with a `t`-dependent explicit
  part too.
- The explicit tableaus' comparisons were measured on 2026-09-25, and
  `ImplicitEuler()`'s on 2026-10-05, the same with 2.9.6 and 2.9.7.

## On a device

- Measured in step 4, on an Apple M3 with Metal 1.11.1.
- **How Metal gets in** was proposed in step 4 and decided 2026-09-24.
  `PLAN.md` offered a separate environment or a conditional `Pkg.add` in
  the gated file. MultiFloats joined the environment on 2026-09-28.
- In step 4 the Metal environment's manifest had 99 packages on 1.13 and
  96 on 1.10, standard libraries included. That the run passes on both,
  and how long it takes, was measured in step 4 too.
- The gate and the 4-ulp tolerance were proposed in step 4 and decided
  2026-09-24. The checks that the ordinary suite never sees Metal moved
  to `test/Project.toml` in step 6, with the test environment.
- That Metal compiles per state size was measured in step 4.
- **Float32x2 on Metal** was added and measured on 2026-09-28.

## Releases

Each release is tagged at the commit that bumped `Project.toml`, and has
a GitHub release. No 0.1.0 was ever tagged.

- `v1.0.0`, 2026-09-24, Erik's decision.
- `v1.1.0`, the explicit tableaus.
- `v1.2.0`, scratch reuse.
- `v1.3.0`, MultiFloats, 2026-09-28.
- `v1.4.0`, Butcher62, CooperVerner8 and ImplicitEuler, 2026-10-05.

The tags up to `v1.2.0` were in place by 2026-09-26. The GitHub releases
of the first three were created on 2026-09-28, from their tags.

## TreeGRRMHD

The first intended user, and how its plan's steps map onto this one's:
- **TreeGRRMHD's step 5 may start.** It needs its 4b, which is step 2
  here. It adds the package by URL, now the remote's,
  `Pkg.add(url = "https://github.com/eschnett/IMEXRungeKutta.jl", rev =
  "v1.4.0")`, or the latest tag. On Metal, each new state length costs
  about a second of kernel compilation in Metal's broadcast (`CODE.md`,
  "On a device"), which a regrid pays.
- **TreeGRRMHD's 4c is step 3 here.** The numbers it needs for
  SSP3(4,3,3), its L-stability, its order in the stiff limit and where
  its step ends there, are in `CODE.md`, "Where a step ends in the stiff
  limit" and "Validation".
- The repository's remote, `eschnett/IMEXRungeKutta.jl` on GitHub, exists
  since 2026-09-24.

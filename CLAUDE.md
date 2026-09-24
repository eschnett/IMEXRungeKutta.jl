# Working notes for Claude in IMEXRungeKutta.jl

Read `CODE.md` first. It is the design document and states *why* things
are the way they are. This file is only about mechanics.

## What this package is

Fixed-step additive implicit–explicit Runge–Kutta integrators
(IMEX-SSP of Pareschi & Russo, and ARS) in which **the user solves the
implicit stage equation**. Per implicit stage the integrator calls
`solve_imp!(U, u★, γΔt, p, t)` once, recovers the implicit tendency as
`(U − u★)/(γΔt)`, and never evaluates the stiff term or forms a
Jacobian. It is aimed at hyperbolic systems with stiff relaxation that
is local to a grid cell.

Rules that follow from `CODE.md` and govern every change:

- **The stage contract is the package.** Exactly one `solve_imp!` call
  per implicit stage. The tendency is taken before any limiter. No
  nonlinear-solver loop, tolerance or retry lives here.
- **Two abscissae.** An explicit evaluation is at `tⁿ + c̃_k Δt`, and a
  stage solve at `tⁿ + c_k Δt`; they differ (SSP3(4,3,3) at stage 1).
  Mixing them up is invisible on any problem where the right-hand side
  does not depend on `t` (see SciML/OrdinaryDiffEq.jl#4620), so every
  order test includes one where it does.
- **Skip what the tableau does not use.** Evaluate the explicit part only
  where column `k` of `Ã` or `b̃_k` is nonzero. Store an implicit
  tendency only where it is read.
- **Coefficients in extended precision.** Most tableaus are irrational.
  Hold them in `BigFloat` (exactly, where rational) and convert to `T`
  once. Never paste Float64 literals.
- **Spec-first.** When the implementation shows `CODE.md` wrong or
  incomplete, amend it and say so ("(amended in step N)", "(measured in
  step N)").

## Current state

**Steps 0 and 1 done: scaffolding and the tableaus** (2026-09-24).
`CODE.md` records the requirements, the method, the survey of
OrdinaryDiffEq and ClimaTimeSteppers, the package design, the test plan
and the open questions. One question is still open: where a TreeAMR
state vector's ownership partition comes from. `PLAN.md` splits the work
into steps 0–6. What exists:
- `Project.toml` with CommonSolve as the one run-time dependency, and
  Test, LinearAlgebra and TOML as test-only extras;
- `src/IMEXRungeKutta.jl`, the module, which re-exports CommonSolve's
  `init`, `solve`, `solve!` and `step!` (no methods yet) and exports
  `IMEXTableau` and the six named tableaus;
- `src/tableau.jl`: `IMEXTableau{R}`, its checks, and the internals step
  2's plan reads: `solves`, `explicit_used`, `implicit_used`,
  `scratch_count` and `coefficients(T, Tt, tab)`;
- `src/tableaus.jl`: `IMEXSSP222`, `IMEXSSP2322` (SSP2(3,2,2)),
  `IMEXSSP3332`, `IMEXSSP3433`, `ARS222` and `ARS443`, in closed form;
- `test/runtests.jl`, `test/scaffold_tests.jl`,
  `test/tableau_properties.jl` (test-only helpers: order conditions,
  `R(z)`, the E-polynomial, the SSP coefficient) and
  `test/tableau_tests.jl`;
- `.github/workflows/CI.yml` and `.github/dependabot.yml`, which run once
  there is a remote;
- `README.md`.

Step 2, the integrator, is next. `CODE.md` ("Tableaus", "Measured
properties") has each tableau's patterns and scratch count, which the
stage plan must reproduce. Upstream's `ARS443` differs from ours in `b̃`
("Cross-checks"), which step 3's oracle test must allow for.

## Commands

The suite, at one thread, at four threads, and with `--check-bounds=yes`,
on the current release (`julia`, 1.13 on Erik's Mac):

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. --check-bounds=yes -e 'using Pkg; Pkg.test()'
```

On the floor, Julia 1.10, **`Pkg.test()` forces `--check-bounds=yes`**
on the test process whatever the parent was started with, and ignores the
parent's `--check-bounds=auto` (measured in step 0). The allocation tests
would then skip themselves. Pass the mode as a test-process argument,
which does override it; this is also how CI's `julia-runtest` passes it:

```bash
julia +1.10 --project=. -e 'using Pkg; Pkg.test(julia_args=["--check-bounds=auto"])'
```

```bash
julia +1.10 --project=. --threads=4 -e 'using Pkg; Pkg.test(julia_args=["--check-bounds=auto"])'
```

`Pkg.test()` passes the parent's `--threads` on to the test process on
both versions (measured in step 0). The suite prints the thread count and
`CHECK_BOUNDS_FORCED` as it starts: check them.

`Manifest.toml` is untracked and shared between Julia versions. `Pkg.test`
re-resolves a manifest written by the other version by itself, but a
plain `julia +1.10 --project=. -e 'using IMEXRungeKutta'` after a 1.13
resolve fails, because PrecompileTools 1.3 (CommonSolve's one dependency)
requires Julia 1.12. Delete `Manifest.toml` when switching.

The clean-archive check, run before a step is reported done:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" &&
    julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Allocation tests are meaningless under `--check-bounds=yes`; skip them
there, by `Base.JLOptions().check_bounds == 1` (`CHECK_BOUNDS_FORCED` in
`test/runtests.jl`). Measure allocations with a top-level helper that
takes concrete arguments, not a closure inside a `@testset`, which
allocates itself.

## Conventions

These match the sibling packages in `~/src/jl` (TreeAMR, TreeHydro,
EntropyEOS):

- 4-space indent, wrap at about 80–90 columns; `return` on the last line
  of any non-trivial function.
- **Julia 1.10** (`julia = "1.10"` in `[compat]`). No `[sources]`, since
  that needs 1.11.
- Generic in the scalar type `T` and in the array type. No float literals
  in `T`-generic code: write `T(1//2)` or `oftype(x, 2)`. Float32 must
  work.
- Stage arithmetic by one fused broadcast per combination over
  `similar(u0)` arrays, so that device arrays work. No scalar indexing
  into the state, except in the by-owner path for a CPU `Array`
  (`CODE.md`, "Stage arithmetic").
- The only run-time dependency is CommonSolve (which brings
  PrecompileTools and Preferences with it). Test-only dependencies go in
  `[extras]` and `[targets]`, e.g. OrdinaryDiffEqSDIRK as an oracle.
  `test/scaffold_tests.jl` asserts the `[deps]` list, so adding one means
  amending `CODE.md` and that test together.
- Unicode in mathematical contexts (`Δt`, `u★`, `γ`, `Ã`, `b̃`, `c̃`).
  `ArgumentError`s say *why*.
- Docstrings are prose-first and point at `CODE.md`.
- **Testset names are claims**, each opening with a comment naming the
  failure mode it guards.

## Repository facts

- No remote yet. Work on a branch, and do not commit,
  push or merge without being asked.
- `.gitignore` is the siblings':
  `Manifest.toml` everywhere, `/docs/build/`, `/bin/output/`, editor
  leftovers, `TODO.md`.
- If Erik adds a `TODO.md`, it is his personal list: **do not modify
  it.**
- The upstream reference implementations are read, never edited, and are
  not run-time dependencies:
  - OrdinaryDiffEqSDIRK: `lib/OrdinaryDiffEqSDIRK/src/imex_tableaus.jl`
    and `generic_imex_perform_step.jl` in SciML/OrdinaryDiffEq.jl;
  - ClimaTimeSteppers: `src/solvers/imex_ssprk.jl`, `imex_ark.jl` and
    `imex_tableaus.jl` in CliMA/ClimaTimeSteppers.jl.
- OrdinaryDiffEqSDIRK #4620 (the mistimed last explicit stage) was open
  on 2026-09-24. Until it is fixed, oracle comparisons must use an
  explicit part that does not depend on `t`.

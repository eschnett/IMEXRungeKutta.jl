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

**Requirements and background only** (2026-09-24). `CODE.md` records
the requirements, the method, the survey of OrdinaryDiffEq and
ClimaTimeSteppers, the test plan and the open questions. `src/` is still
the Pkg template. The package design, and a `PLAN.md` if the work is
split into steps, are next.

## Commands

Nothing to run yet beyond the template. Once there is a test suite, run
it at one thread, at four threads, and with `--check-bounds=yes`:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. --threads=4 -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. --check-bounds=yes -e 'using Pkg; Pkg.test()'
```

Allocation tests are meaningless under `--check-bounds=yes`; skip them
there. Measure allocations with a top-level helper that takes concrete
arguments, not a closure inside a `@testset`, which allocates itself.

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
- Stage arithmetic by broadcasting over `similar(u0)` arrays, so that
  device arrays work. No scalar indexing into the state.
- Run-time dependencies at most StaticArrays. SciMLBase is an open
  question in `CODE.md`. Test-only dependencies go in `[extras]`, e.g.
  OrdinaryDiffEqSDIRK as an oracle.
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

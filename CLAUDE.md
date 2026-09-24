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

**The implementation plan is complete** (2026-09-24). Its steps 0–6,
the scaffolding, the tableaus, the integrator on the broadcast path, the
validation, the Metal smoke run, the stage arithmetic by owner and the
review pass, are done, and `PLAN.md` is deleted. `CODE.md` records the
requirements, the method, the survey of OrdinaryDiffEq and
ClimaTimeSteppers, the package design as built and measured, the tests
and the open questions. Erik decided every proposal the steps made but
one on 2026-09-24. Open or pending:
- where a TreeAMR state vector's ownership partition comes from (open);
- whether SSP2(3,3,2)'s coefficients, recalled by the step-1 reviewer
  rather than transcribed, match Pareschi & Russo (2005) (open);
- the Symmetry run of `bench/symmetry_stage_arithmetic.sh`, which is
  Erik's, and the one decision still proposed, fresh sticky tasks rather
  than persistent workers for the owner path, which waits on it
  (`CODE.md`, "By owner, as built");
- the v0.1.0 tag, which is Erik's (below);
- #4620 upstream ("Repository facts").

What exists:
- `Project.toml` with CommonSolve as the one run-time dependency, and
  `test/Project.toml`, the test environment: CommonSolve, LinearAlgebra,
  OrdinaryDiffEqSDIRK (compat `2.9.6`, the oracle), TOML and Test;
- `src/IMEXRungeKutta.jl`, the module, which re-exports CommonSolve's
  `init`, `solve`, `solve!` and `step!` and exports `IMEXProblem`,
  `IMEXTableau` and the seven named tableaus;
- `src/tableau.jl`: `IMEXTableau{R}`, its checks, and the internals the
  plan reads: `solves`, `explicit_used`, `implicit_used`, `row_empty`,
  `scratch_count` and `coefficients(T, Tt, tab)`;
- `src/tableaus.jl`: `IMEXSSP222`, `IMEXSSP2322` (SSP2(3,2,2)),
  `IMEXSSP2332` (SSP2(3,3,2), in neither upstream, so no oracle),
  `IMEXSSP3332`, `IMEXSSP3433`, `ARS222` and `ARS443`, in closed form;
- `src/lincomb.jl`: `lincomb!`, `lincomb_copy!`, `copy_state!`,
  `increment!`, `first_touch!` and `copy_initial`, each with a last
  `partition` argument: `nothing` is one fused broadcast, and an
  `OwnerPartition` the by-owner path of step 5, a loop per range on a
  sticky task placed on the owning thread (`by_owner`). Also the
  partition's checks (`owner_partition`), `even_partition` (`:even`) and
  `block_partition`, a helper for segmented block layouts;
- `src/plan.jl`: `Stage`, `StagePlan`, `build_plan` and `plan_calls`;
- `src/integrator.jl`: `IMEXProblem`, `IMEXIntegrator`, and the methods
  of `init` (with `partition`, checked by `resolve_partition`), `step!`,
  `solve!` and `solve`;
- `test/runtests.jl`, `test/scaffold_tests.jl`,
  `test/tableau_properties.jl` (test-only helpers: order conditions,
  `R(z)`, the E-polynomial, the SSP coefficient), `test/tableau_tests.jl`,
  `test/mocks.jl` (logging mock callbacks), `test/interface_tests.jl`,
  `test/mechanics_tests.jl`, `test/smoke_order_tests.jl` and
  `test/readme_tests.jl` (which runs the README's example);
- the validation of step 3: `test/problems.jl` (test-only helpers: the
  fitted order and the Kaps problem), `test/order_tests.jl`,
  `test/stiff_tests.jl` (Kaps), `test/ap_tests.jl` (the stiff limit),
  `test/ssp_tests.jl` (total variation) and `test/oracle_tests.jl`
  (OrdinaryDiffEqSDIRK); the numbers are in `CODE.md`, "Validation";
- the device smoke run of step 4: `test/metal_tests.jl`, gated by
  `IMEXRUNGEKUTTA_TEST_METAL=1` and run in its own environment,
  `test/metal/Project.toml` (Metal, Test, and this package from `../..`);
  it is not part of `Pkg.test()`, and the ordinary suite checks that it
  never sees Metal. The numbers are in `CODE.md`, "On a device";
- the owner path of step 5: `test/owner_tests.jl` (bitwise identity with
  the broadcast for every tableau, placement per range, nesting, the
  refusals, allocations), `bench/stage_arithmetic.jl` (a thread sweep,
  broadcast against by owner, with a persistent-worker prototype for
  comparison) and `bench/symmetry_stage_arithmetic.sh`, its SLURM job. The
  numbers are in `CODE.md`, "By owner, as built";
- `.github/workflows/CI.yml`, with five cells (`CODE.md`, "File
  layout"), and `.github/dependabot.yml`;
- `README.md`, with the CI badge, installation by URL, the worked example
  and the 0.1.0 status.

**TreeGRRMHD's step 5 may start.** It needs 4b, which is this step 2. It
adds the package by URL, now the remote's,
`Pkg.add(url = "https://github.com/eschnett/IMEXRungeKutta.jl")`, and,
once Erik has tagged it, with `rev = "v0.1.0"`. On Metal, each new state
length costs about a second of kernel compilation in Metal's broadcast
(`CODE.md`, "On a device"), which a regrid pays.

**TreeGRRMHD's 4c is this step 3.** The numbers it needs for
SSP3(4,3,3), its L-stability, its order in the stiff limit and where its
step ends there, are in `CODE.md`, "Where a step ends in the stiff limit"
and "Validation".

**0.1.0 is prepared, not tagged.** Before the tag: Erik commits his
`LICENSE.md` (MIT); the remote's `main`, which has steps 0–2, gets steps 3
and 4; and CI is green there. The tag, and any registration, are Erik's.
Steps 5 and 6 are on local `main` too, after the 0.1.0 commit (ea555c6),
while `Project.toml` still says 0.1.0: whether the tag goes on that commit
or a later one, and the next version number, are Erik's.

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

**The test environment is `test/Project.toml`** (since step 6). On 1.10
and on 1.13, `Pkg.test()` copies it to a temporary environment, keeps its
`[compat]`, and adds this package itself, so the file does not list it;
it writes no `test/Manifest.toml` (measured in step 6). A package that the
tests load by name must be listed there even when this package depends on
it: without CommonSolve in it, `using CommonSolve` in `scaffold_tests.jl`
fails with "Package CommonSolve not found".

**The test environment is large** since step 3 added OrdinaryDiffEqSDIRK
as the oracle: 142 packages on 1.13, 138 on 1.10. Measured on the M3
(step 3), from a fresh `JULIA_DEPOT_PATH`, so including the downloads:
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

`Manifest.toml` is untracked and shared between Julia versions. `Pkg.test`
re-resolves a manifest written by the other version by itself, but a
plain `julia +1.10 --project=. -e 'using IMEXRungeKutta'` after a 1.13
resolve fails, because PrecompileTools 1.3 (CommonSolve's one dependency)
requires Julia 1.12. Delete `Manifest.toml` when switching.

**The device smoke run**, on an Apple-silicon Mac, in its own
environment. Set it up once per Julia version (delete
`test/metal/Manifest.toml` when switching), then run it:

```bash
julia --project=test/metal -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

```bash
IMEXRUNGEKUTTA_TEST_METAL=1 julia --project=test/metal test/metal_tests.jl
```

The same two with `julia +1.10` run it on the floor; both pass (measured
in step 4, Metal 1.11.1). `Pkg.develop` is needed on 1.10, which ignores
the environment's `[sources]`, and harmless on 1.11 and later, which read
it. Without the variable the file does nothing; with it and no functional
Metal, it fails. The run takes 14 s on 1.13 and 11 s on 1.10; the first
setup on 1.10 precompiled Metal in 25 s. Never add Metal to the root
`Project.toml` or to `test/Project.toml`: every `Pkg.test()`, on Linux
too, would then install a GPU stack, and `scaffold_tests.jl` refuses it.

The clean-archive check, run before a step is reported done:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" &&
    julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The stage-arithmetic benchmark runs its own thread sweep, one Julia
process per thread count (`IMEXRK_BENCH_THREADS`, default
`1,2,4,6,8,12`), on a 10⁸-byte state; it takes about two minutes on the
M3:

```bash
julia --project=. bench/stage_arithmetic.jl
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
- **Julia 1.10** (`julia = "1.10"` in `[compat]`). No `[sources]` in the
  root `Project.toml`, since that needs 1.11. The one `[sources]` is the
  Metal environment's, which 1.10 ignores and `Pkg.develop` replaces
  ("Commands").
- Generic in the scalar type `T` and in the array type. No float literals
  in `T`-generic code: write `T(1//2)` or `oftype(x, 2)`. Float32 must
  work.
- Stage arithmetic by one fused broadcast per combination over
  `similar(u0)` arrays, so that device arrays work. No scalar indexing
  into the state, except in the by-owner path for a CPU `Array`
  (`CODE.md`, "Stage arithmetic").
- The only run-time dependency is CommonSolve (which brings
  PrecompileTools and Preferences with it). Test-only dependencies go in
  `test/Project.toml`, with their bounds in its `[compat]`, e.g.
  OrdinaryDiffEqSDIRK as an oracle; the root `Project.toml` has no
  `[extras]` or `[targets]` (Erik's decision in step 6).
  `test/scaffold_tests.jl` asserts both files' `[deps]` lists and the root
  `[compat]`, so adding a dependency of either kind means amending
  `CODE.md` and that test together.
- Unicode in mathematical contexts (`Δt`, `u★`, `γ`, `Ã`, `b̃`, `c̃`).
  `ArgumentError`s say *why*.
- Docstrings are prose-first and point at `CODE.md`, by a section's
  heading or a bold paragraph label that exists there.
- `README.md`'s ` ```julia ` blocks are extracted and run by
  `test/readme_tests.jl`, so they must run as written; code that must
  not run, such as the installation, is indented instead. CI runs on a
  README-only push to `main` for that reason.
- **Testset names are claims**, each opening with a comment naming the
  failure mode it guards.

## Sharp edges

Traps that the design makes easy to fall into, carried over from the
implementation plan. The why is in `CODE.md`.

- **Increments, not tendencies.** The integrator stores `d_k = U − u★`
  and uses the coefficients `a_kj/a_jj` and `b_j/a_jj`, computed in
  extended precision and rounded to `T` once. Never form
  `(U − u★)/(a_kk Δt)` and multiply back.
- **Structural zeros are never read.** A skipped tendency has no array,
  and scratch filled with NaN must not reach the result (`0·NaN = NaN`).
  The stage plan branches on the pattern; it never multiplies by zero.
- **Coefficient precision.** Compute every closed form inside
  `setprecision(BigFloat, 256) do … end`, never at the global precision.
  Compute the abscissae `c` and `c̃` exactly and convert the result, not a
  sum of converted entries. Convert to `T` from the 256-bit value, the
  rational tableaus included: `T(BigFloat(r))`.
- **`BigFloat` and precompilation.** The named tableaus are functions,
  not `const`s, and `init` builds its stage plan from them at run time.
- **Type instability is confined to `init`.** The stage plan's type
  depends on the tableau's nonzero pattern. `step!` must pass `@inferred`,
  and on the broadcast path allocate nothing.
- **CommonSolve's names.** `using CommonSolve: CommonSolve, init, solve,
  solve!, step!`, add methods, and re-export the four, so that they are
  SciMLBase's and OrdinaryDiffEq's bindings too; a test asserts
  `IMEXRungeKutta.init === CommonSolve.init`.
- **The tableau names clash with OrdinaryDiffEqSDIRK's** (`IMEXSSP3433`,
  `ARS222`, …). The oracle test does `import OrdinaryDiffEqSDIRK as ODE`
  and qualifies everything.
- **The oracle.**
  - SciML's `SplitODEProblem(f1, f2, u0, tspan)` treats **`f1`
    implicitly** and `f2` explicitly: the opposite order from
    `IMEXProblem(f_exp!, solve_imp!, …)`.
  - Upstream's state must be real.
  - Upstream's SSP3(4,3,3) uses the 14-digit coefficients, which differ
    from the closed form by about 1e−15.
  - Upstream's `ARS443` has `b̃ = b`, not the last row of `Ã`. Compare
    ARS(4,4,3) with `IMEXTableau("…", Ã, b, A, b)`, built from
    `ARS443()`'s parts.
  - #4620 is under "Repository facts".
- **Mocks** record their calls into buffers preallocated in `p`, so that
  the mechanics tests can also run under the allocation helper.
- **Metal** (`CODE.md`, "On a device").
  - The device has no `Float64`. Only `T` may reach a kernel; the time
    stays on the host, and a callback converts what it takes from `t`.
  - Metal compiles a shape-specialized kernel once a shape has been
    broadcast more than ten times, so the second step at a new state
    length compiles for about a second and allocates about 250 MB. Warm
    up three steps before measuring anything.

## Repository facts

- The remote is `origin`, `eschnett/IMEXRungeKutta.jl` on GitHub (from
  2026-09-24). Work on a branch, and do not commit, push, merge or tag
  without being asked; pushes, tags and releases are Erik's.
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
  on 2026-09-24. Until it is fixed, oracle comparisons of the tableaus
  with `c̃_s ≠ 1`, SSP3(3,3,2) and SSP3(4,3,3), must use an explicit part
  that does not depend on `t`; `test/oracle_tests.jl` marks theirs
  `@test_broken`. A fix upstream makes them unexpected passes, which fail
  the suite: then turn them into plain `@test`s.

# Working notes for Claude in IMEXRungeKutta.jl

## Contents

- [Read first](#read-first)
- [What this package is](#what-this-package-is)
- [Commands](#commands)
- [Conventions](#conventions)
- [Sharp edges](#sharp-edges)
- [Repository facts](#repository-facts)

## Read first

Read `CODE.md` first. It is the design document and states *why* things
are the way they are. This file is only about mechanics.

Where content goes:
- **`README.md`**, for users: what the package does, installation, a
  short example, the status, and a pointer to `CODE.md`.
- **`CLAUDE.md`**, this file, loaded into every session, so as short as
  it can be: the map of the files, commands, conventions, sharp edges
  and repository facts, each a short rule pointing at `CODE.md`.
- **`CODE.md`**, the current design and the reasons for it, in the
  present tense, without dates or step numbers on decisions; only what
  is not settled is marked, as **(open)** or **(proposed)**.
- **`PLAN.md`**, a concrete plan, when there is one: steps, each with
  what it delivers and how one knows it is done. There is none now.
- **`HISTORY.md`**, how the package got here: who decided what and when,
  rejected alternatives, replaced measurements, fixed upstream bugs and
  the releases, by topic.

The test between `CODE.md` and `HISTORY.md`: would someone changing the
code today need this? Then `CODE.md`, rationale included.

- **Spec-first.** When the implementation shows `CODE.md` wrong or
  incomplete, fix `CODE.md` to describe the current state, and record in
  `HISTORY.md` what changed, when and why.
- **Keep the tables of contents current**, in every one of these files:
  `##` and `###` headings in `CODE.md`, `PLAN.md` and `HISTORY.md`, `##`
  only in `README.md` and here.

## What this package is

Fixed-step additive implicit–explicit Runge–Kutta integrators
(IMEX-SSP of Pareschi & Russo, and ARS) in which **the user solves the
implicit stage equation** (`CODE.md`, "Purpose" and "Requirements").
Per implicit stage the integrator calls `solve_imp!(U, u★, γΔt, p, t)`
once and never evaluates the stiff term or forms a Jacobian. Open:
where a TreeAMR state vector's ownership partition comes from
(`CODE.md`, "Stage arithmetic").

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
parent's `--check-bounds=auto`. The allocation tests
would then skip themselves. Pass the mode as a test-process argument,
which does override it; this is also how CI's `julia-runtest` passes it:

```bash
julia +1.10 --project=. -e 'using Pkg; Pkg.test(julia_args=["--check-bounds=auto"])'
```

```bash
julia +1.10 --project=. --threads=4 -e 'using Pkg; Pkg.test(julia_args=["--check-bounds=auto"])'
```

`Pkg.test()` passes the parent's `--threads` on to the test process on
both versions. The suite prints the thread count and
`CHECK_BOUNDS_FORCED` as it starts: check them.

**The test environment is `test/Project.toml`** (`CODE.md`, "File
layout"). A package that the tests load by name must be listed there even
when this package depends on it.

**The test environment is large** (`CODE.md`, "Requirements"). A fresh
`Pkg.test()` spends about two minutes precompiling it, and the suite
takes about one minute at one thread, most of it `oracle_tests.jl`; a
first `--check-bounds=yes` run precompiles again and runs slower. The
measurements are in `HISTORY.md`, "The test environment".

`Manifest.toml` is untracked and shared between Julia versions. `Pkg.test`
re-resolves a manifest written by the other version by itself, but a
plain `julia +1.10 --project=. -e 'using IMEXRungeKutta'` after a 1.13
resolve fails, because of PrecompileTools 1.3 (`CODE.md`, "Dependencies
and names"). Delete `Manifest.toml` when switching.

**The device smoke run**, on an Apple-silicon Mac, in its own
environment. Set it up once per Julia version (delete
`test/metal/Manifest.toml` when switching), then run it:

```bash
julia --project=test/metal -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

```bash
IMEXRUNGEKUTTA_TEST_METAL=1 julia --project=test/metal test/metal_tests.jl
```

The same two with `julia +1.10` run it on the floor; both pass (Metal
1.11.1). `Pkg.develop` is needed on 1.10, which ignores
the environment's `[sources]`, and harmless on 1.11 and later, which read
it. Without the variable the file does nothing; with it and no functional
Metal, it fails. The run takes 14 s on 1.13 and 11 s on 1.10, most of it
compiling kernels; the first setup on 1.10
precompiled Metal in 25 s. Never add Metal to the root
`Project.toml` or to `test/Project.toml`: every `Pkg.test()`, on Linux
too, would then install a GPU stack, and `scaffold_tests.jl` refuses it
(`CODE.md`, "On a device").

The clean-archive check, run before work is reported done:

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
  `[extras]` or `[targets]` (`CODE.md`, "File layout").
  `test/scaffold_tests.jl` asserts both files' `[deps]` lists and the root
  `[compat]`, so adding a dependency of either kind means amending
  `CODE.md` and that test together.
- **The public API** is `CODE.md`, "The interface" and "The callback
  contracts", under semantic versioning: a change to it is a major
  version.
- Unicode in mathematical contexts (`Δt`, `u★`, `γ`, `Ã`, `b̃`, `c̃`).
  `ArgumentError`s say *why*.
- Docstrings are prose-first and point at `CODE.md`, by a section's
  heading or a bold paragraph label that exists there.
- `README.md`'s ` ```julia ` blocks are extracted and run by
  `test/readme_tests.jl`, so they must run as written; code that must
  not run, such as the installation, is indented instead (`CODE.md`,
  "File layout", on CI).
- **Testset names are claims**, each opening with a comment naming the
  failure mode it guards.

## Sharp edges

Traps that the design makes easy to fall into. The why is in `CODE.md`.

- **The stage contract is the package.** Exactly one `solve_imp!` call
  per implicit stage. The tendency is taken before any limiter. No
  nonlinear-solver loop, tolerance or retry lives here (`CODE.md`, "One
  step").
- **Two abscissae.** An explicit evaluation is at `tⁿ + c̃_k Δt`, and a
  stage solve at `tⁿ + c_k Δt`; they differ (SSP3(4,3,3) at stage 1).
  Mixing them up is invisible on any problem where the right-hand side
  does not depend on `t` (see SciML/OrdinaryDiffEq.jl#4620), so every
  order test includes one where it does (`CODE.md`, "Tableaus" and
  "Testing").
- **Skip what the tableau does not use.** Evaluate the explicit part only
  where column `k` of `Ã` or `b̃_k` is nonzero. Store an implicit
  tendency only where it is read (`CODE.md`, "One step", Cost).
- **Increments, not tendencies.** The integrator stores `d_k = U − u★`
  and uses the coefficients `a_kj/a_jj` and `b_j/a_jj`, computed in
  extended precision and rounded to `T` once. Never form
  `(U − u★)/(a_kk Δt)` and multiply back (`CODE.md`, "One step").
- **Structural zeros are never read.** A skipped tendency has no array,
  and scratch filled with NaN must not reach the result (`0·NaN = NaN`).
  The stage plan branches on the pattern; it never multiplies by zero
  (`CODE.md`, "The stage plan and storage").
- **Coefficient precision.** Hold coefficients in `BigFloat` (exactly,
  where rational) and convert to `T` once; never paste Float64 literals,
  since most tableaus are irrational. Compute every closed form inside
  `setprecision(BigFloat, 256) do … end`, never at the global precision.
  Compute the abscissae `c` and `c̃` exactly and convert the result, not a
  sum of converted entries. Convert to `T` from the 256-bit value, the
  rational tableaus included: `T(BigFloat(r))` (`CODE.md`, "Tableaus"
  and "Tableaus are values").
- **`BigFloat` and precompilation.** The named tableaus are functions,
  not `const`s, and `init` builds its stage plan from them at run time
  (`CODE.md`, "Tableaus are values").
- **Type instability is confined to `init`.** The stage plan's type
  depends on the tableau's nonzero pattern. `step!` must pass `@inferred`,
  and on the broadcast path allocate nothing (`CODE.md`, "The stage plan
  and storage").
- **CommonSolve's names.** `using CommonSolve: CommonSolve, init, solve,
  solve!, step!`, add methods, and re-export the four, so that they are
  SciMLBase's and OrdinaryDiffEq's bindings too; a test asserts
  `IMEXRungeKutta.init === CommonSolve.init` (`CODE.md`, "Dependencies
  and names").
- **The tableau names clash with OrdinaryDiffEqSDIRK's** (`IMEXSSP3433`,
  `ARS222`, `ImplicitEuler`, …), and the explicit ones with
  OrdinaryDiffEqLowOrderRK's (`Euler`, `RK4`) and OrdinaryDiffEqSSPRK's
  (`SSPRK33`). The oracle test imports them `as ODE`, `as LowRK` and
  `as SSPRK`, and qualifies everything (`CODE.md`, "Tableaus are values").
- **High orders need `BigFloat`.** Butcher62 (order 6) and CooperVerner8
  (order 8) reach `Float64` round-off within one or two halvings of `Δt`;
  their observed order is measured in 256-bit `BigFloat`, and no
  explicit method above order 4 is SSP (`CODE.md`, "Explicit tableaus").
  `Butcher6` upstream is a different method; ours is `Butcher62`.
- **An explicit tableau's first stage is trivial**: `f_exp!` reads `uⁿ`
  there with no stage limiter call. Every right-hand-side input is limited
  only with the same function as both limiters, and `u0` limited before
  `init` (`CODE.md`, "Explicit tableaus").
- **The oracle** (`CODE.md`, "Why not an existing package" and "The
  oracle").
  - SciML's `SplitODEProblem(f1, f2, u0, tspan)` treats **`f1`
    implicitly** and `f2` explicitly: the opposite order from
    `IMEXProblem(f_exp!, solve_imp!, …)`.
  - Upstream's state must be real.
  - Upstream's SSP3(4,3,3) uses the 14-digit coefficients, which differ
    from the closed form by about 1e−15.
  - `ARS443_2_9_6` in the tests is OrdinaryDiffEqSDIRK 2.9.6's `ARS443`,
    with `b̃ = b`, not the paper's: a third-order method of its own, in
    the `O(Δt⁴)` test and the stiff-limit table. Upstream from 2.9.7 is
    compared with `ARS443()` (`CODE.md`, "Cross-checks").
- **MultiFloats converts only through `BigFloat`** (`CODE.md`,
  "Software floats"). A double-float has no `Int`, `Float64`, `round(Int,
  …)` or `cos`, so a `T(x)` between two float types in `src/` goes
  through `convert_float`. Its identity method is bounded,
  `where {T<:AbstractFloat}`: unbounded, it is not more specific than the
  `x::AbstractFloat` method, which is then picked even for `x::T` and
  allocates a `BigFloat` in every callback that uses it.
- **Mocks** record their calls into buffers preallocated in `p`, so that
  the mechanics tests can also run under the allocation helper.
- **Metal** (`CODE.md`, "On a device").
  - The device has no `Float64`. Only `T` may reach a kernel; the time
    stays on the host, and a callback converts what it takes from `t`.
  - Metal compiles a shape-specialized kernel once a shape has been
    broadcast more than ten times, so the second step at a new state
    length compiles for about a second. Warm up three steps before
    measuring anything.

## Repository facts

- The remote is `origin`, `eschnett/IMEXRungeKutta.jl` on GitHub. Work
  on a branch, and do not commit, push, merge or tag without being asked;
  pushes, tags and releases are Erik's. The package is not registered;
  any registration is Erik's too. The releases are in `HISTORY.md`.
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

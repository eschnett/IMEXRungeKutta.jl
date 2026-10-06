# IMEXRungeKutta history

The decisions that shaped the package, with who took them, when and why,
the alternatives they rejected, and the releases. The current design, and
every reason that still governs the code, is in [`CODE.md`](CODE.md); the
git log has the rest.

## Contents

- [The implementation plan](#the-implementation-plan)
- [Decisions](#decisions)
  - [Scope](#scope)
  - [Tableaus](#tableaus)
  - [Implicit Euler](#implicit-euler)
  - [One step](#one-step)
  - [Package design](#package-design)
  - [Tests and CI](#tests-and-ci)
  - [Documents](#documents)
- [Releases](#releases)

## The implementation plan

The package was built by a plan, `PLAN.md`, in seven steps: step 0, the
scaffolding; step 1, the tableaus; step 2, the integrator on the
broadcast path; step 3, the validation; step 4, the Metal smoke run;
step 5, the stage arithmetic by owner; step 6, a review pass. The plan
was complete on 2026-09-24, and `PLAN.md` was deleted. TreeGRRMHD's plan
cites two of them: its step 4b is step 2 here, and its 4c is step 3.

Erik decided every proposal the steps made on 2026-09-24, the last one
through a Symmetry run he asked for ([Package design](#package-design)).

## Decisions

### Scope

- **The run-time dependency** (2026-09-24): CommonSolve alone. The
  requirement had been "at most StaticArrays", with SciMLBase open. Owning
  the names and depending on SciMLBase were rejected (`CODE.md`,
  "Dependencies and names").
- **Both SSP2 schemes** (step 1, Erik). The requirements listed
  SSP2(3,3,2) alone, under the name `IMEXSSP2322`; Erik decided to have
  SSP2(3,2,2) as `IMEXSSP2322` too, by the naming rule, and SSP2(3,3,2)
  as `IMEXSSP2332`.
- **Explicit tableaus** (2026-09-25, Erik's request): `Euler`, `RK4` and
  `SSPRK33`, so that non-stiff problems and debugging run through the
  same integrator.
- **MultiFloats' double-floats** (2026-09-28, Erik's request), as state
  and time types, on the CPU and on Metal.
- **Butcher62, CooperVerner8 and ImplicitEuler** (2026-10-05, Erik's
  request).

### Tableaus

- **ARS(4,4,3)'s `b̃` is the paper's** (step 1): Erik checked the paper,
  and so did the step-1 review.
- **SSP2(3,3,2)'s coefficients** were first the step-1 reviewer's
  recollection of the paper; they were checked against it, Table 4 of
  arXiv:1009.2757, on 2026-09-24.
- **The E-polynomial test refuses** a tableau with a negative
  coefficient rather than pass it (step 1, decided 2026-09-24):
  non-negative coefficients are sufficient for `E ≥ 0`, not necessary,
  so a negative one leaves the question undecided.
- **The classical order conditions** come from the rooted trees since
  2026-10-05, with Butcher62 and CooperVerner8; until then they were a
  hand-written list, complete to order 4.

### Implicit Euler

- **Backward Euler** (2026-10-05): Erik chose it over the IMEX
  forward–backward Euler, ARS(1,1,1).

### One step

- **A stage limiter's correction reaches `uⁿ⁺¹` only through `f_exp!`**
  (2026-09-25). This was first described as limiting "exactly what `f`
  reads", like OrdinaryDiffEq's SSPRK methods; TreeHydro (its step 9)
  measured that their correction persists in the state and this
  package's does not. Folding the correction into the tendency was
  rejected for the weights it gets (`CODE.md`, "One step"); a caller
  whose correction must persist passes it as the step limiter too.

### Package design

- **The interface** (steps 0–2, decided 2026-09-24): CommonSolve's
  names, the integrator type with `const` fields, `dt` required, no
  default tableau, the refusals of `init`, the step-count tolerance, and
  values rather than types for tableaus ("values or types" had been
  open).
- **No status** from a step (2026-09-24): an earlier open question; the
  stage solver owns its outcomes.
- **Fresh tasks, not persistent workers**, by owner (step 5). `PLAN.md`
  asked for persistent sticky workers if they reached zero allocations at
  no loss in speed; a prototype did, for one combination. Erik
  (2026-09-24): keep fresh tasks until the Symmetry run of
  `bench/symmetry_stage_arithmetic.sh` decides. It did, job 563504:
  persistent workers are no faster at any thread count.
- **Scratch reuse** (2026-09-26), after TreeGeneralizedHarmonic measured
  0.13–0.36 s per `init` on Symmetry. Erik decided that a misfit is
  refused, over a silent fallback to fresh arrays. A `reinit!` and a
  workspace object were rejected (`CODE.md`, "Scratch reuse").
- **The partition is the caller's** (2026-10-06, Erik). Where a TreeAMR
  state vector's partition comes from was the one open question; it is
  settled as the caller's job, with `block_partition` an unexported
  helper.

### Tests and CI

- **The test environment is `test/Project.toml`** (step 6, Erik), rather
  than `[extras]` and `[targets]` in the root `Project.toml`, as step 0
  had proposed.
- **OrdinaryDiffEqSDIRK's floor is 2.9.7** (2026-10-05), the release
  that fixed SciML/OrdinaryDiffEq.jl#4620 (reported 2026-09-24) and gave
  `ARS443` the paper's `b̃`.
- **The Metal run has an environment of its own** (step 4, decided
  2026-09-24), rather than a conditional `Pkg.add` in the gated file, and
  no CI cell, since GitHub's hosted macOS runners have no Metal.
- **CI's fifth cell**, Julia 1.10 at four threads (step 6, Erik). CI runs
  on a README-only push to `main` since step 6, when it turned out a
  README change could break the suite unseen.

### Documents

- **2026-10-06.** The documents were reorganized into `README.md`,
  `CLAUDE.md`, `CODE.md` and this file, and then streamlined to describe
  the package as it is. OrdinaryDiffEqSDIRK 2.9.6's ARS(4,4,3) variant
  was removed from the tests with its documentation (Erik's decision).

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

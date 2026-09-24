# IMEXRungeKutta implementation plan

This file is for the sessions that implement IMEXRungeKutta, one step at
a time. The **design** is in `CODE.md`. Read it first, in full: it is
authoritative, and this file is only the work breakdown. For each step
it says what changes, what must not change, and what must be measured
and recorded. `CLAUDE.md` has the mechanics. Delete this file when the
last step is marked *(Done.)*.

**Steps 0–5 are done. Step 6 is next.** The 0.1.0 tag of step 4 is
Erik's, and waits on his `LICENSE.md` and a green CI on the remote. Step
5's Symmetry run is Erik's too.

Each step ends in a green suite and a `CODE.md` update, and each is a
brief that a single session can carry. The steps are in dependency
order. TreeGRRMHD's `PLAN.md` calls three of them by its own names:

| step | here | TreeGRRMHD | what TreeGRRMHD needs from it |
|---|---|---|---|
| 1 | the tableaus | 4a | — |
| 2 | the integrator, broadcast path | 4b | its step 5 waits for this |
| 3 | validation | 4c | SSP3(4,3,3) and SSP2(2,2,2), validated; L-stability and stiff-limit order recorded |
| 4 | the Metal smoke test, then v0.1.0 | "a tag" | the release |
| 5 | stage arithmetic by owner | — | host performance |
| 6 | review pass | — | — |

## Ground rules, every step

- **Read `CLAUDE.md` first**, then this file's ground rules, sharp edges
  and the step, then the `CODE.md` sections the step names.
- **Work on a branch** named `claude/step-N-<slug>`, off `main`.
  - Commit in TreeAMR's style: implement, then measure, then record, with
    the measured numbers in the commit body.
  - **Do not merge into `main` and do not push.** Report the branch and
    its commits. There is a remote now, `origin` on GitHub, and the rule
    stands: pushes, tags and releases are Erik's.
- **Never edit a sibling checkout.** That includes TreeAMR, TreeGRRMHD,
  TreeHydro, and the upstream reference implementations in
  OrdinaryDiffEq and ClimaTimeSteppers. If a step needs something
  upstream, describe exactly what and why, and report it.
- **Do not modify `TODO.md`**, if one exists.
- **Julia 1.10 floor, generic from the first line.**
  - Generic in `T` and in the array type.
  - No float literals where `T` is in play.
  - No scalar indexing into the state outside the step-5 owner path.
  - `Float32` must work.
- **Run the full suite before and after each step**: at one thread, at
  four threads, and with `--check-bounds=yes` (commands in `CLAUDE.md`).
  Skip allocation tests under `--check-bounds=yes`.
- **Spec-first.**
  - Amend `CODE.md` and say so: "(amended in step N)", "(measured in
    step N)".
  - Decide each **(open)** the step owns.
  - Never loosen an assertion to get green. Report the failure instead.
- **Testset names are claims.** Each opens with a comment naming the
  failure mode it guards. Orders, residuals and counts are asserted as
  numbers with tolerances.
- **Tests are small.** Each test file takes under about 60 s. The whole
  suite takes under about 5 min at one thread.
- **One step at a time.** The next starts from a green suite on `main`.
- **At the end of each step**, update `CLAUDE.md`'s "Current state", and
  mark the step *(Done.)* here, with the date.

## Running a step as an agent

This is TreeGRRMHD's pattern.
- **One implementation agent per step**, in its own worktree, with the
  brief below. Only one runs at a time.
- **The agent does not ask questions mid-step.** Where it would have
  asked, it decides, marks the decision "(proposed in step N)" in
  `CODE.md`, and lists it in the report.
- **The reviewer**:
  - checks out the branch and runs the suite at one and four threads;
  - reads the `CODE.md` diff;
  - walks the acceptance list against the report;
  - merges only then.

  A failed item goes back to the same agent with the finding, not to a
  new agent.

**The brief** (replace `N`):

    You are implementing step N of PLAN.md in the Julia package at
    ~/src/jl/IMEXRungeKutta. Read CLAUDE.md, then PLAN.md's "Ground
    rules", "Sharp edges" and step N, then the CODE.md sections step N
    names. Work on the branch claude/step-N-<slug> in this worktree;
    commit in TreeAMR's style with measured numbers in the commit body;
    do not merge, push, or touch TODO.md or any sibling checkout. Do all
    of step N. If part of it is blocked (a device you do not have, a
    cluster run), finish everything else and say exactly what is
    blocked. Never loosen an assertion to get green. When you would ask
    a question, decide, mark it "(proposed in step N)" in CODE.md, and
    list it in the report. End with the report PLAN.md specifies.

**The report**, in this order:
1. the branch and its commits, one line each;
2. the acceptance list, each item marked *done*, *measured* (with the
   number), or *blocked* (with the reason);
3. the suite's summary lines at one and at four threads, and with
   `--check-bounds=yes`, verbatim;
4. every `CODE.md` change, quoted, with its marker;
5. the decisions it took, each marked (proposed);
6. anything the next step should know.

## Sharp edges to know before starting

- **Two abscissae.** An explicit evaluation is at `tⁿ + c̃_k Δt`, and a
  stage solve at `tⁿ + c_k Δt`. Every test that could tell them apart
  has a right-hand side that depends on `t`. A problem autonomous in `t`
  hides the #4620 failure mode completely.
- **Increments, not tendencies.** The integrator stores `d_k = U − u★`
  and uses the coefficients `a_kj/a_jj` and `b_j/a_jj`. These are
  computed in extended precision and rounded to `T` once. Never form
  `(U − u★)/(a_kk Δt)` and multiply back.
- **Structural zeros are never read.** A skipped tendency has no array,
  and scratch filled with NaN must not reach the result (`0·NaN = NaN`).
  The stage plan branches on the pattern; it never multiplies by zero.
- **Coefficient precision.**
  - Compute every closed form inside `setprecision(BigFloat, 256) do …
    end`. Never rely on the global precision.
  - Compute the abscissae `c` and `c̃` exactly, and convert the result.
    Do not sum converted entries.
  - Convert to `T` from the 256-bit value, including for the rational
    tableaus: `T(BigFloat(r))`.
- **`BigFloat` and precompilation.** The named tableaus are functions,
  not `const`s. `init` builds its stage plan from them at run time.
- **CommonSolve.** Write `using CommonSolve: CommonSolve, init, solve,
  solve!, step!`, add methods, and re-export the four names. They are
  then the same bindings as SciMLBase's and OrdinaryDiffEq's, so there
  is no clash when both are loaded. A test asserts that
  `IMEXRungeKutta.init === CommonSolve.init`.
- **The names clash with OrdinaryDiffEqSDIRK** (`IMEXSSP3433`, `ARS222`,
  …). The oracle test does `import OrdinaryDiffEqSDIRK as ODE` and
  qualifies everything.
- **The oracle.**
  - SciML's `SplitODEProblem(f1, f2, u0, tspan)` treats **`f1`
    implicitly** and `f2` explicitly: the opposite order from
    `IMEXProblem(f_exp!, solve_imp!, …)`.
  - Upstream's explicit part must not depend on `t` until #4620 is
    fixed, for the tableaus with `c̃_s ≠ 1`, SSP3(3,3,2) and SSP3(4,3,3).
    Where `c̃_s = 1` the comparison passes (measured in step 3).
  - Upstream's state must be real.
  - Upstream's SSP3(4,3,3) uses the 14-digit coefficients, which differ
    from the closed form by about 1e−15.
  - Upstream's `ARS443` has `b̃ = b`, not the last row of `Ã` (measured in
    step 1). Compare ARS(4,4,3) with `IMEXTableau("…", Ã, b, A, b)`, built
    from `ARS443()`'s parts.
- **Type instability is confined to `init`.** The stage plan's type
  depends on the tableau's nonzero pattern. `step!` must be inferred
  (`@inferred`) and allocation-free.
- **Allocation tests** go through a top-level helper that takes concrete
  arguments, never a closure inside a `@testset`, which allocates
  itself. They are skipped under `--check-bounds=yes`.
  - On Julia 1.10, `Pkg.test()` forces `--check-bounds=yes` whatever
    the parent was started with (measured in step 0). Run the floor with
    `Pkg.test(julia_args=["--check-bounds=auto"])`, as in `CLAUDE.md`, or
    the allocation tests never run there. `CHECK_BOUNDS_FORCED` in
    `test/runtests.jl` is the flag to skip on.
- **Mocks** record their calls into preallocated buffers in `p`, so the
  mechanics tests can also run under the allocation helper.
- **Metal** (measured in step 4; `CODE.md`, "On a device").
  - It is never in the root `Project.toml`; the smoke run has its own
    environment, `test/metal/Project.toml`, and commands in `CLAUDE.md`.
  - The device has no `Float64`. Only `T` may reach a kernel; the time
    stays on the host, and a callback converts what it takes from `t`.
  - Metal compiles a shape-specialized kernel once a shape has been
    broadcast more than ten times, so the second step at a new state
    length compiles for about a second and allocates about 250 MB. Warm
    up three steps before measuring anything.

## Step 0 — Scaffolding *(Done, 2026-09-24.)*

`CODE.md`: "Requirements", "Dependencies and names", "File layout".

Changes:
- **`Project.toml`:**
  - `[deps]`: CommonSolve;
  - `[compat]`: CommonSolve (the current 0.2 series) and
    `julia = "1.10"`;
  - `[extras]` and `[targets]`: Test and LinearAlgebra for now.
- **`src/IMEXRungeKutta.jl`:** the module shell, with its docstring
  (prose-first, pointing at `CODE.md`), the CommonSolve imports and
  re-exports, and includes for the files that exist. It replaces
  `hello`/`domath`.
- **`test/runtests.jl`**, and **`test/scaffold_tests.jl`**: the package
  loads; the four CommonSolve names are CommonSolve's bindings.
- **CI:** `.github/workflows/CI.yml` after TreeHydro's, with cells for
  1.10 on Linux, the current release on Linux and macOS, and one
  four-thread cell. Also `.github/dependabot.yml`. They run once there
  is a remote.
- **`README.md`:** a short description, and the status.
- **`CLAUDE.md`:** "Current state" and "Commands".

Accept:
- `Pkg.test()` is green on 1.10 and on the current release, at one and
  four threads.
- A clean `git archive HEAD` instantiates and passes.

## Step 1 — The tableaus (TreeGRRMHD 4a) *(Done, 2026-09-24.)*

`CODE.md`: "Tableaus", "Tableaus are values", "Testing" (Tableaus).

Changes:
- **`src/tableau.jl`:**
  - `IMEXTableau{R}` with its name, `Ã`, `b̃`, `A` and `b`, plus `c̃`
    and `c`, computed exactly;
  - the constructor checks, each an `ArgumentError` that says why: the
    parts are square and of equal size; `Ã` is strictly lower
    triangular; `A` is lower triangular; admissibility (`a_kk = 0`
    implies column `k` of `A` and `b_k` are zero);
  - the explicit-used and implicit-used patterns;
  - the conversion of the coefficients to `T` that step 2's plan uses,
    including `a_kj/a_jj` and `b_j/a_jj`.
- **`src/tableaus.jl`:** `IMEXSSP222`, `IMEXSSP2322`, `IMEXSSP2332`,
  `IMEXSSP3332`, `IMEXSSP3433`, `ARS222` and `ARS443`, in closed form
  (`IMEXSSP2332`, SSP2(3,3,2), added in step 1 at Erik's decision), each
  with a
  comment citing the table in Pareschi & Russo (2005) or Ascher, Ruuth &
  Spiteri (1997).
  - SSP3(4,3,3) uses `α = (9 − √57)/6`, `β = α/4` and
    `η = (1 − 2α)/4`.
  - Cross-check each against OrdinaryDiffEqSDIRK's `imex_tableaus.jl` and
    ClimaTimeSteppers' `imex_tableaus.jl`, by reading them only. Record
    any disagreement in `CODE.md`.
- **`test/tableau_properties.jl`:** test-only helpers.
  - The additive order conditions up to order 3. At order 2, all four
    combinations of `b`, `b̃` with `c`, `c̃`. At order 3, all
    `b`/`b̃` × `c`/`c̃` × `c`/`c̃` for `bᵀ(xy) = 1/3`, and all
    `b`/`b̃` × `A`/`Ã` × `c`/`c̃` for `bᵀXy = 1/6`.
  - The classical order-4 conditions of each part, used to show the
    order stops at the stated one.
  - `R(z) = 1 + z bᵀ(I − zA)⁻¹𝟙`, `R(∞)`, and A-stability through the
    E-polynomial `E(y) = |Q(iy)|² − |P(iy)|² ≥ 0`. The poles are at
    `1/a_kk > 0`, which is in the right half-plane.
  - The SSP coefficient of the explicit part, by bisection on
    Kraaijevanger's conditions for `K = [Ã 0; b̃ᵀ 0]`, to 1e−10.
  - Stiff accuracy: `b` equals the last row of `A`, and `b̃` equals the
    last row of `Ã`.
- **`test/tableau_tests.jl`:**
  - each tableau meets its stated order: exactly (`== 0`) for the
    rational ones, and within 1e−70 in `BigFloat` for the others;
  - each fails at the next order: some order-4 condition misses by more
    than 1e−3;
  - the constructor refuses a non-triangular tableau and an ESDIRK-type
    first column, each with its reason;
  - the converted coefficients are the correctly rounded 256-bit values,
    in `Float32`, `Float64` and `BigFloat`;
  - the 14 printed digits of SSP3(4,3,3) are the closed form rounded,
    with order residuals below 5e−15.

Record in `CODE.md`, "(measured in step 1)": a table per tableau with
the order, stiff accuracy, `R(∞)`, A-stability (and so L-stability), the
SSP coefficient of the explicit part, the explicit-used and
implicit-used patterns, and the scratch count from "The stage plan and
storage".

Accept:
- The suite is green.
- The table is in `CODE.md`.
- SSP3(4,3,3)'s L-stability is stated as computed, not quoted.

## Step 2 — The integrator, broadcast path (TreeGRRMHD 4b) *(Done, 2026-09-24.)*

`CODE.md`: "One step", and all of "Package design" except the by-owner
path in "Stage arithmetic".

Changes:
- **`src/plan.jl`:** the stage plan.
  - Per stage: the `(coefficient, array)` tuples of `u★`, with only the
    nonzero terms; whether the stage solves; whether it is
    explicit-used and implicit-used; and its two times.
  - The update, in the same form.
  - The trivial first stage aliased to `integ.u`.
  - `u★` formed in its `d_k` array.
- **`src/lincomb.jl`:** one internal function for a fused linear
  combination `dst = x₀ + Σ c_j x_j`, as a single broadcast.
  - Its signature takes a partition argument that is `nothing` for now,
    so that step 5 only adds methods.
  - Terms are summed in plan order.
- **`src/integrator.jl`:**
  - `IMEXProblem(f_exp!, solve_imp!, u0, tspan, p = nothing)`;
  - `init` with `dt`, `stage_limiter`, `step_limiter` and `alias_u0`.
    `partition` is accepted only as `nothing` until step 5; any other
    value is an `ArgumentError` saying it is not implemented yet;
  - `step!`, `solve!` and `solve`;
  - the step count and the time arithmetic of "Time and the step count";
  - the public fields;
  - scratch written once in `init`.
- **Docstrings** for every export. **`README.md`** gets a worked example:
  scalar relaxation `u′ = f(u, t) − (u − ū)/ε`, whose stage solve has a
  closed form.
- **`test/interface_tests.jl`:**
  - the step count: `Δt ≤ dt`, a whole number of steps to `t1`, and no
    extra step from round-off (`(0, 1)` with `dt = 1/10`);
  - the last `t` equals `t1` exactly, and `step!` after it throws;
  - the time type and `T` are separate, and a `Float32` state with
    `Float64` time works;
  - a complex state works;
  - `p` defaults to `nothing`;
  - `alias_u0` behaves as documented;
  - `solve` returns the integrator;
  - `step!` passes `@inferred`.
- **`test/mechanics_tests.jl`**, all of "Testing" (Mechanics) except the
  partition items:
  - Call counts per step for each tableau, checked against the plan and
    against hand-counted constants: SSP3(4,3,3) makes three `f_exp!`
    calls and four `solve_imp!` calls.
  - Each call's time is the right abscissa: `c̃` for `f_exp!` and the
    stage limiter, and `c` for `solve_imp!`.
  - `solve_imp!` receives `U` equal to `u★`, as a distinct array, and
    `u★` is unchanged afterwards.
  - The stage limiter is called exactly before each `f_exp!` call, on
    the same array, never on `integ.u`.
  - At ARS's first stage, `f_exp!` receives `integ.u` itself.
  - Scratch filled with NaN leaves no NaN in the result.
  - An exception from `solve_imp!` leaves `integ.u` and `integ.t`
    unchanged.
  - With `g ≡ 0`, the result equals the explicit RK method to round-off.
  - Untouched components match the explicit-only run bitwise.
  - The step limiter is called once per step, on `integ.u`, at `tⁿ⁺¹`.
  - `step!` is allocation-free after warm-up.
- **`test/smoke_order_tests.jl`:** `u′ = −u + cos t`, with the implicit
  part `−u`. The observed order of SSP2(2,2,2) and SSP3(4,3,3) is within
  0.2 of 2 and 3. This keeps step 2 from shipping an integrator that
  passes only mocks. Step 3 does the full validation.

Accept:
- The suite is green.
- `step!` allocates 0 bytes per step for a `Vector{Float64}` state, for
  every tableau (measured).
- `CLAUDE.md` says TreeGRRMHD's step 5 may start. It adds the package
  by URL.

## Step 3 — Validation (TreeGRRMHD 4c) *(Done, 2026-09-24.)*

`CODE.md`: "Testing" (Order, Stiff limit, Asymptotic preservation, SSP,
Oracle), "Why not an existing package" (the oracle restrictions).

Changes:
- **`test/order_tests.jl`**, for every tableau: the observed order is
  within 0.1 of the stated one on two problems.
  - The split linear ODE `u′ = iu − u`, with a complex state; `iu` is
    explicit.
  - `u′ = −u + cos t`.
  - Each uses a sequence of `Δt` in the asymptotic range, and the fit
    uses the finest three.
- **`test/stiff_tests.jl`:** the Kaps problem,
  `y₁′ = −(2 + 1/ε) y₁ + y₂²/ε` and `y₂′ = y₁ − y₂ − y₂²`, with exact
  solution `y₁ = e^{−2t}`, `y₂ = e^{−t}`.
  - The implicit part is `((y₂² − y₁)/ε, 0)`. Its stage solve is
    `U₂ = u★₂`, then `U₁` in closed form.
  - For `ε ∈ {1, 1e−3, 1e−6, 1e−9}`, record the observed order per `ε`
    and per tableau, including any order reduction.
  - Assert the recorded numbers to ±0.15, so that a regression is
    caught.
- **`test/ap_tests.jl`:** at `ε = 1e−12`, the residual `|y₁ − y₂²|`
  after one step, per tableau.
  - Assert `O(ε)` where the theory predicts it.
  - Record it, and amend `CODE.md`, where a tableau does not land on the
    manifold.
- **`test/ssp_tests.jl`:** first-order upwind advection of a square wave
  with stiff relaxation toward a fixed profile. Find the largest
  `C = Δt/Δt_FE` at which the total variation stays non-increasing over
  a fixed run, by bisection. Record `C` per tableau against step 1's SSP
  coefficient.
- **`test/oracle_tests.jl`:** `import OrdinaryDiffEqSDIRK as ODE`, and
  add it to `[extras]`.
  - For each of the six tableaus upstream has (all seven but SSP2(3,3,2),
    which has no oracle), ten steps of a real linear problem
    with an explicit part that does not depend on `t` agree with
    upstream to 1e−12.
  - The problem is `SplitODEProblem(g, f, …)`, with upstream's default
    nonlinear solver: for a linear `g` it is exact to round-off.
  - Also one comparison with a `t`-dependent explicit part, marked
    `@test_broken`, with the #4620 link. When upstream is fixed, it turns
    into a pass and is noticed.

Record in `CODE.md`, "(measured in step 3)":
- the observed orders;
- the Kaps table;
- the AP residuals;
- the measured `C`;
- SSP3(4,3,3)'s stiff-limit order, in "Tableaus".

Accept:
- The suite is green, and each whole file runs in under 60 s.
- The tables are in `CODE.md`.
- The report gives the numbers TreeGRRMHD's "The partition" should be
  amended with. Erik makes that edit, not this step.

## Step 4 — Metal smoke test, then v0.1.0 *(Done, 2026-09-24; the tag is Erik's.)*

`CODE.md`: "Testing" (Mechanics, the device smoke run).

Changes:
- **`test/metal_tests.jl`**, run only under
  `IMEXRUNGEKUTTA_TEST_METAL=1`:
  - a `Float32` relaxation problem on an `MtlArray` state, with scalar
    indexing disallowed;
  - ten steps each of SSP2(2,2,2) and SSP3(4,3,3);
  - agreement with the same run on the CPU in `Float32`, to a few ulps
    per step. It is not bitwise, because the device may contract to FMA.
- **Metal must not be installed on every test run.** Decide how the
  test gets it: a separate `test/metal/Project.toml` that develops the
  package, or a conditional `Pkg.add` in the gated file. Record the
  decision, "(proposed in step 4)", and put the command in `CLAUDE.md`.
- **Prepare the release:** the README status, and the version already
  at `0.1.0`. Ask Erik to tag it. The tag and any registration are his.

Accept:
- The gated test passes on an Apple-silicon Mac, or the step reports it
  as blocked.
- The ungated suite does not load Metal.

Outcome: the environment is `test/metal/Project.toml`, developing the
package from `../..` (proposed in step 4). The device agrees with the CPU
bitwise, on Julia 1.13 and 1.10; the numbers are in `CODE.md`, "On a
device".

## Step 5 — Stage arithmetic by owner *(Done, 2026-09-24; the Symmetry run is pending.)*

`CODE.md`: "Stage arithmetic", "The stage plan and storage" (first
touch).

The **(open)** question of where TreeAMR's partition comes from does not
block this step. The tests use `:even` and hand-built partitions with
several ranges per thread.

Changes:
- **`src/lincomb.jl`:** the owner path for a CPU `Array`.
  - Thread `c` loops over its ranges, as a sticky task placed on
    default-pool thread `c`. That is TreeAMR's `threaded_chunks` on its
    branch `claude/festive-bun-656842`: `task.sticky = true`, then
    `jl_set_task_tid` with the interactive-pool offset. Read it; do not
    depend on TreeAMR.
  - Errors from worker tasks are unwrapped to the innermost exception.
  - Forming `u★` writes the `d_k` array and `U` in one pass.
- **`src/integrator.jl`:**
  - the `partition` keyword: `nothing`, `:even` or explicit ranges;
  - its validation (disjoint, covering, the length is `nthreads()`),
    each failure an `ArgumentError` naming the missing or doubled index;
  - refusing a partition for a non-`Array` state;
  - writing the scratch by owner in `init`, for first touch.
- **Allocations.** Spawning tasks allocates `O(nthreads)` small objects
  per combination.
  - Measure it, and replace "allocation-free" for this path with a bound
    that does not depend on the state size, amended in `CODE.md`.
  - Or, if persistent sticky workers give zero allocations at no cost
    in speed, use them. In TreeAMR's affinity measurements, persistent
    workers were no faster than fresh tasks.
- **Tests**, from "Testing" (Mechanics):
  - broadcast, `:even` and an explicit multi-range partition give
    bitwise identical results, at one thread and at four;
  - each range runs on its own thread, recorded by an internal hook in
    the tests;
  - a gap or an overlap is refused.
- **`bench/stage_arithmetic.jl`:** one combination, and one SSP3(4,3,3)
  step with trivial callbacks. It runs over a thread sweep, broadcast
  against `:even`, on a state of about 10⁸ bytes. Also a SLURM script
  for Symmetry, after TreeAMR's `bench/symmetry_affinity.sh`.

Record in `CODE.md`, "(measured in step 5)":
- the Mac numbers;
- the Symmetry numbers, if Erik runs the job, or "(pending a Symmetry
  run)".

Accept:
- The suite is green at one and four threads.
- The allocation claim in `CODE.md` matches the measurement.
- The benchmark runs.

Outcome: fresh sticky tasks, not persistent workers (proposed in step
5). A persistent-worker prototype reaches 0 bytes at the same speed for
one combination, and is kept in the benchmark; `CODE.md`, "By owner, as
built", says what it leaves out. The owner path allocates 64 bytes plus
403–433 per thread per combination on 1.13 (559–589 on 1.10), whatever
the state size, and nothing at one thread. The Mac numbers are in
`CODE.md`; the Symmetry run, `bench/symmetry_stage_arithmetic.sh`, is
Erik's.

## Step 6 — Review pass

Read every file against `CODE.md`.
- Every **(proposed in step N)** is either confirmed with Erik or
  listed.
- Every docstring points at the right `CODE.md` section.
- The README example runs as written, and is a doctest or a test.
- `CLAUDE.md` describes what exists.

Then mark the last step *(Done.)* and delete this file, recording in
`CODE.md`'s status line that the plan is complete.

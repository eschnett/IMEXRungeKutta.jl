# IMEXRungeKutta.jl

[![CI](https://github.com/eschnett/IMEXRungeKutta.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/eschnett/IMEXRungeKutta.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/eschnett/IMEXRungeKutta.jl/graph/badge.svg?token=RFHAIF5SWS)](https://codecov.io/gh/eschnett/IMEXRungeKutta.jl)

- [Installation](#installation)
- [Example](#example)
- [Status](#status)

Fixed-step Runge–Kutta integration for method-of-lines systems

    u′ = f(u, t) + g(u, t)

where `f` is non-stiff and treated explicitly and `g` is stiff and
treated implicitly. The core is additive implicit–explicit (IMEX)
Runge–Kutta, **with the implicit stage equation solved by the caller**.
Per implicit stage the integrator makes exactly one call

    solve_imp!(U, u★, γΔt, p, t)

which must leave `U = u★ + γΔt g(U, t)`, by whatever means suits the
problem: a closed form, a fixed point with a physics fallback, Newton on a
few unknowns per cell. The integrator recovers the implicit increment as
`U − u★`. It never evaluates `g`, never forms a Jacobian, and has no
nonlinear-solver loop of its own.

The thirteen tableaus come in three kinds, all held in closed form in
extended precision:
- **IMEX:** SSP2(2,2,2), SSP2(3,2,2), SSP2(3,3,2), SSP3(3,3,2) and
  SSP3(4,3,3) of Pareschi & Russo (2005), and ARS(2,2,2) and ARS(4,4,3)
  of Ascher, Ruuth & Spiteri (1997);
- **explicit**, with no stage solver: explicit Euler (for debugging),
  classical RK4, Shu & Osher's SSPRK(3,3), Butcher's sixth-order and
  Cooper & Verner's eighth-order methods;
- **implicit**, with no explicit part: backward Euler.

A caller may also give a tableau of its own.

The target is hyperbolic systems with stiff relaxation local to a grid
cell: resistive MHD, radiation or neutrino transport, reaction networks.
The package was written for TreeGRRMHD.jl; TreeHydro.jl and
TreeGeneralizedHarmonic.jl use it too. Nothing here depends on them.

The interface is CommonSolve's `init`, `step!`, `solve!` and `solve`, so
the package loads beside SciMLBase or OrdinaryDiffEq without a name
clash. It has stage and step limiter hooks, works for any array type that
broadcasts, and supports Julia 1.10 and later.

`CODE.md` is the design document: the requirements, the method, the
package design, why an existing package does not fit, and the tests.
`HISTORY.md` records the decisions that shaped it, and the releases.

## Installation

The package is not registered. Add it by its URL, at the release tag:

    using Pkg
    Pkg.add(url = "https://github.com/eschnett/IMEXRungeKutta.jl", rev = "v1.4.0")

## Example

Scalar relaxation in each of three cells, `u′ = cos t − (u − ū)/ε`, with
the forcing `cos t` explicit and the stiff relaxation toward `ū`
implicit. The stage equation `U = u★ − γΔt (U − ū)/ε` has a closed form,
so the stage solver is a single broadcast:

```julia
using IMEXRungeKutta

# The explicit part, f(u, t) = cos t.
f_exp!(du, u, p, t) = (du .= cos(t); nothing)

# The implicit stage: U = u★ + γΔt g(U, t) with g(u) = −(u − ū)/ε.
function solve_imp!(U, u★, γΔt, p, t)
    λ = γΔt / p.ε
    @. U = (u★ + λ * p.ū) / (1 + λ)
    return nothing
end

p = (ε = 1e-6, ū = [1.0, 2.0, 3.0])
prob = IMEXProblem(f_exp!, solve_imp!, copy(p.ū), (0.0, 1.0), p)
integ = solve(prob, ARS443(); dt = 0.01)

integ.t                         # 1.0, exactly, after integ.nstep == 100 steps
(integ.u .- p.ū) ./ p.ε         # ≈ cos(1) = 0.5403 in each cell
```

`solve` returns the integrator. For a chunked driver, `init` it once per
chunk and call `step!(integ)` or `solve!(integ)`; `integ.u` may be changed
in place between steps. `init` also takes `stage_limiter` and
`step_limiter`, with OrdinaryDiffEq's signature `(u, integ, p, t)`. Unlike
in OrdinaryDiffEq's SSPRK methods, a stage limiter's change reaches the
result only through the explicit evaluations. Pass a correction that must
hold in the state, such as an atmosphere reset, as `step_limiter` too
(`CODE.md`, "One step").

ARS(4,4,3) is stiffly accurate, so as `ε → 0` each step ends on the
quasi-steady state `u ≈ ū + ε cos t`. SSP3(4,3,3)
is not: in that limit its result is off it by `O(Δt)`,
here by about `−0.28 Δt cos t` (see `CODE.md`, "Tableaus").

A purely explicit tableau, `Euler()`, `RK4()`, `SSPRK33()`, the
sixth-order `Butcher62()` or the eighth-order `CooperVerner8()`, makes no
stage solve, so the stage solver may be `nothing`:

```julia
# u′ = −u with classical RK4: four evaluations per step, and no solver.
f_decay!(du, u, p, t) = (du .= .-u; nothing)
explicit = solve(IMEXProblem(f_decay!, nothing, [1.0], (0.0, 1.0)), RK4(); dt = 0.1)
explicit.u[1] - exp(-1.0)       # ≈ 3.3e-7, RK4's error at Δt = 0.1
```

Its first stage reads `uⁿ` itself, so only the step limiter limits it:
pass a correction that must reach every right-hand-side input as both
`stage_limiter` and `step_limiter` (`CODE.md`, "Explicit tableaus").

The purely implicit `ImplicitEuler()`, backward Euler, makes no explicit
evaluation, so the explicit part may be `nothing`:

```julia
# u′ = −u with backward Euler: one stage solve per step, U = u★/(1 + Δt).
solve_decay!(U, u★, γΔt, p, t) = (U .= u★ ./ (1 + γΔt); nothing)
implicit = solve(IMEXProblem(nothing, solve_decay!, [1.0], (0.0, 1.0)), ImplicitEuler();
                 dt = 0.1)
implicit.u[1] - exp(-1.0)       # ≈ 0.018, backward Euler's error at Δt = 0.1
```

## Status

**Version 1.4.0.** `IMEXProblem`, `init`, `step!`, `solve!` and `solve`,
with the stage and step limiters, for the thirteen named tableaus
(`IMEXSSP222`, `IMEXSSP2322`, `IMEXSSP2332`, `IMEXSSP3332`, `IMEXSSP3433`,
`ARS222`, `ARS443`; `Euler`, `RK4`, `SSPRK33`, `Butcher62`,
`CooperVerner8`; `ImplicitEuler`) and a caller's own `IMEXTableau`.
- The stage arithmetic is one fused broadcast per combination, for any
  array type. `step!` is type-stable, and on this path allocation-free
  for a CPU `Array`.
- For a CPU `Array` with threads, `init(...; partition = :even)`, or a
  partition of the caller's own with one collection of index ranges per
  thread, runs each combination on every thread at once, each element on
  the thread that owns it, with bitwise the same result. It allocates a
  few hundred bytes per thread per combination, whatever the state size,
  and nothing at one thread.
- A chunked driver, with one integrator per chunk, passes the previous
  chunk's integrator as `init(...; reuse = integ)` while the grid is
  unchanged, to take over its scratch arrays instead of allocating and
  first-touching new ones (`CODE.md`, "Scratch reuse").
- MultiFloats' software double-floats, `Float32x2` and `Float64x2`, work
  as the state's real type and as the time type, and `Float32x2` on Metal
  too (`CODE.md`, "Software floats").

What is tested (`CODE.md`, "Testing" and "Validation"): the tableaus'
order, stiff accuracy, L-stability and SSP coefficient, in extended
precision; the observed orders, the order in the stiff limit (SSP3(4,3,3)
drops from 3 to 2 there), where a step lands as `ε → 0`, and total
variation under upwind advection; agreement with OrdinaryDiffEq to
round-off; the double-floats against 256-bit `BigFloat`; a PDE, the
Jin–Xin relaxation of 2D Burgers' equation, in
`examples/jin_xin_2d.jl` (run it with
`julia --project=. examples/jin_xin_2d.jl`); and, on request, a `Float32`
and a `Float32x2` run on an Apple GPU that agrees with the CPU
(`test/metal_tests.jl` says how).

Known limits:
- SSP2(3,3,2) is in neither OrdinaryDiffEq nor ClimaTimeSteppers, so it
  has no oracle. Its coefficients are checked against Pareschi & Russo
  (Table 4 of arXiv:1009.2757), and it passes the order conditions,
  `R(∞) = 0`, and the observed-order, stiff-limit and total-variation
  tests.
- The broadcast stage arithmetic is serial on the host; `partition` is the
  threaded path, for a CPU `Array` only, and the caller builds the
  partition (`CODE.md`, "Stage arithmetic").

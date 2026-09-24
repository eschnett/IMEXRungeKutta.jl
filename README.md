# IMEXRungeKutta.jl

Fixed-step additive implicit–explicit Runge–Kutta (IMEX RK) integration
for method-of-lines systems

    u′ = f(u, t) + g(u, t)

where `f` is non-stiff and treated explicitly and `g` is stiff and
treated implicitly — **with the implicit stage equation solved by the
caller**. Per implicit stage the integrator makes exactly one call

    solve_imp!(U, u★, γΔt, p, t)

which must leave `U = u★ + γΔt g(U, t)`, by whatever means suits the
problem: a closed form, a fixed point with a physics fallback, Newton on a
few unknowns per cell. The integrator recovers the implicit increment as
`U − u★`. It never evaluates `g`, never forms a Jacobian, and has no
nonlinear-solver loop of its own.

The target is hyperbolic systems with stiff relaxation local to a grid
cell: resistive MHD, radiation or neutrino transport, reaction networks.
The first intended user is TreeGRRMHD.jl; nothing here depends on it.

The tableaus are SSP2(2,2,2), SSP2(3,2,2), SSP2(3,3,2), SSP3(3,3,2) and
SSP3(4,3,3) of Pareschi & Russo (2005), and ARS(2,2,2) and ARS(4,4,3) of
Ascher, Ruuth & Spiteri (1997), all held in closed form in extended
precision. The interface is CommonSolve's `init`, `step!`, `solve!` and
`solve`, so the package loads beside SciMLBase or OrdinaryDiffEq without a
name clash. It
has stage and step limiter hooks, works for any array type that
broadcasts, and supports Julia 1.10 and later.

`CODE.md` is the design document: the requirements, the method, the
package design, why an existing package does not fit, and the test plan.

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
`step_limiter`, with OrdinaryDiffEq's signature `(u, integ, p, t)`.

ARS(4,4,3) is stiffly accurate, so as `ε → 0` each step ends on the
quasi-steady state `u ≈ ū + ε cos t`. SSP3(4,3,3), the intended
production scheme, is not: in that limit its result is off it by `O(Δt)`,
here by about `−0.28 Δt cos t` (see `CODE.md`, "Tableaus").

## Status

**The integrator exists, on the broadcast path.** `IMEXProblem`, `init`,
`step!`, `solve!` and `solve` are in place, with the stage and step
limiters, for all seven named tableaus (`IMEXSSP222`, `IMEXSSP2322`,
`IMEXSSP2332`, `IMEXSSP3332`, `IMEXSSP3433`, `ARS222`, `ARS443`) and a
caller's own `IMEXTableau`. `step!` is type-stable and allocation-free.
The tableaus' order, stiff accuracy, L-stability and SSP coefficient are
computed, tested, and recorded in `CODE.md`. `PLAN.md` breaks the rest
into steps: the validation next, then a 0.1.0 release, then the by-owner
stage arithmetic for threaded CPU arrays.

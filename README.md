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

The tableaus are SSP2(2,2,2), SSP2(3,3,2), SSP3(3,3,2) and SSP3(4,3,3) of
Pareschi & Russo (2005), and ARS(2,2,2) and ARS(4,4,3) of Ascher, Ruuth &
Spiteri (1997), all held in closed form in extended precision. The
interface is CommonSolve's `init`, `step!`, `solve!` and `solve`, so the
package loads beside SciMLBase or OrdinaryDiffEq without a name clash. It
has stage and step limiter hooks, works for any array type that
broadcasts, and supports Julia 1.10 and later.

`CODE.md` is the design document: the requirements, the method, the
package design, why an existing package does not fit, and the test plan.

## Status

**The tableaus exist; the integrator does not yet.** `IMEXTableau` and
the six named tableaus (`IMEXSSP222`, `IMEXSSP2322`, `IMEXSSP3332`,
`IMEXSSP3433`, `ARS222`, `ARS443`) are in place. Their order, stiff
accuracy, L-stability and SSP coefficient are computed and tested, and
recorded in `CODE.md`. `PLAN.md` breaks the rest into steps: the
integrator next, then its validation, and a 0.1.0 release. There is no
worked example until the integrator exists.

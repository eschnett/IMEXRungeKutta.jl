"""
    IMEXRungeKutta

Fixed-step additive implicit–explicit Runge–Kutta integration for
`u′ = f(u, t) + g(u, t)`, where `f` is non-stiff and treated explicitly
and `g` is stiff and treated implicitly, in which **the caller solves the
implicit stage equation**. Per implicit stage the integrator makes one
call `solve_imp!(U, u★, γΔt, p, t)`, which must leave
`U = u★ + γΔt g(U, t)`, and recovers the implicit increment as `U − u★`.
It never evaluates `g` and never forms a Jacobian. The target is
hyperbolic systems with stiff relaxation local to a grid cell, where the
best stage solver is problem-specific.

The tableaus are the IMEX-SSP schemes of Pareschi & Russo (2005) and the
ARS schemes of Ascher, Ruuth & Spiteri (1997). The interface is
CommonSolve's `init`, `solve!`, `step!` and `solve`, re-exported here, so
that this package loads beside SciMLBase or OrdinaryDiffEq without a name
clash.

See `CODE.md` in the package root for the design, and why it is so.
"""
module IMEXRungeKutta

# The same bindings as SciMLBase's and OrdinaryDiffEq's: this package adds
# methods to them and re-exports them ("Dependencies and names" in
# `CODE.md`).
using CommonSolve: CommonSolve, init, solve, solve!, step!

export init, solve, solve!, step!
export IMEXProblem
export IMEXTableau
export IMEXSSP222, IMEXSSP2322, IMEXSSP2332, IMEXSSP3332, IMEXSSP3433
export ARS222, ARS443

# The source files of "File layout" in `CODE.md`.
include("tableau.jl")
include("tableaus.jl")
include("lincomb.jl")
include("plan.jl")
include("integrator.jl")

end # module IMEXRungeKutta

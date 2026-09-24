# The Jin–Xin relaxation of 2D Burgers' equation: a small PDE that
# exercises the whole interface ("Testing", A PDE, in `CODE.md`).
#
#     u_t + v_x + w_y = 0
#     v_t + a² u_x    = −(v − u²/2)/ε
#     w_t + a² u_y    = −(w − u²/2)/ε
#
# on the periodic unit square, N × N cells. The linear hyperbolic part is
# explicit, first-order upwind (Lax–Friedrichs with the exact speed `a`,
# which is upwind for a linear system); the relaxation is implicit and
# local to a cell, and its stage solve has a closed form. As ε → 0 the
# system relaxes to Burgers' equation, `u_t + (u²/2)_x + (u²/2)_y = 0`,
# provided `a ≥ max |u|` (the subcharacteristic condition). Jin & Xin,
# Comm. Pure Appl. Math. 48 (1995) 235–276; Pareschi & Russo (2005) use
# relaxation systems of this kind as their test problems.
#
# The state is a 3 × N × N `Array`, component first: the integrator sees
# any array that broadcasts, and this one is not a vector.
#
#     julia --project=. examples/jin_xin_2d.jl
#
# from the package root prints one line per tableau and ε. The test
# `test/jin_xin_tests.jl` includes this file and asserts what it prints.

using IMEXRungeKutta

"""
    JinXin(N, a, ε)

The parameters `p`: `N × N` cells of width `Δx = 1/N`, the relaxation
speed `a` and the relaxation time `ε`. `nlimit` counts the stage-limiter
calls, to show the hook firing.
"""
struct JinXin{T}
    N::Int
    Δx::T
    a::T
    ε::T
    nlimit::Base.RefValue{Int}
end
JinXin(N, a, ε) = JinXin(N, 1 / N, float(a), float(ε), Ref(0))

jinxin_flux(u) = u^2 / 2

# The Lax–Friedrichs flux, at speed `a`, of the linear pair `(u, q)` whose
# flux is `(q, a² u)`: `(v)` in x, `(w)` in y. For a linear system with
# speeds ±a this is the upwind flux.
jinxin_Fu(a, uL, qL, uR, qR) = (qL + qR) / 2 - a / 2 * (uR - uL)
jinxin_Fq(a, uL, qL, uR, qR) = a^2 * (uL + uR) / 2 - a / 2 * (qR - qL)

"""
    jinxin_rhs!(dU, U, p::JinXin, t)

The explicit part, `f_exp!`: the divergence of the upwind fluxes. It is
conservative in `u`, so `sum(u)` changes only by round-off. It indexes the
state, which a caller's right-hand side may do; the integrator does not.
"""
function jinxin_rhs!(dU, U, p::JinXin, t)
    N, a, h = p.N, p.a, inv(p.Δx)
    @inbounds for j in 1:N, i in 1:N
        ip, im = mod1(i + 1, N), mod1(i - 1, N)
        jp, jm = mod1(j + 1, N), mod1(j - 1, N)
        u, v, w = U[1, i, j], U[2, i, j], U[3, i, j]
        uE, vE, uW, vW = U[1, ip, j], U[2, ip, j], U[1, im, j], U[2, im, j]
        uN, wN, uS, wS = U[1, i, jp], U[3, i, jp], U[1, i, jm], U[3, i, jm]
        dU[1, i, j] = -h * (jinxin_Fu(a, u, v, uE, vE) - jinxin_Fu(a, uW, vW, u, v)) -
                      h * (jinxin_Fu(a, u, w, uN, wN) - jinxin_Fu(a, uS, wS, u, w))
        dU[2, i, j] = -h * (jinxin_Fq(a, u, v, uE, vE) - jinxin_Fq(a, uW, vW, u, v))
        dU[3, i, j] = -h * (jinxin_Fq(a, u, w, uN, wN) - jinxin_Fq(a, uS, wS, u, w))
    end
    return nothing
end

"""
    jinxin_solve!(U, u★, γΔt, p::JinXin, t)

The stage solve, `solve_imp!`: `U = u★ + γΔt g(U)` with
`g = (0, −(v − u²/2)/ε, −(w − u²/2)/ε)`. `U` enters as a copy of `u★`, so
`u` needs no line; `v` and `w` relax towards `u²/2` in closed form.
"""
function jinxin_solve!(U, u★, γΔt, p::JinXin, t)
    ε = p.ε
    @views begin
        u = U[1, :, :]
        @. U[2, :, :] = (ε * u★[2, :, :] + γΔt * jinxin_flux(u)) / (ε + γΔt)
        @. U[3, :, :] = (ε * u★[3, :, :] + γΔt * jinxin_flux(u)) / (ε + γΔt)
    end
    return nothing
end

# A stage limiter that only counts its calls.
jinxin_count!(u, integ, p, t) = (p.nlimit[] += 1; nothing)

"""
    jinxin_initial(N)

`u = 1 + sin(2πx) sin(2πy)/2` on the cell centres, and `v = w = u²/2`,
on the equilibrium.
"""
function jinxin_initial(N)
    Δx = 1 / N
    U = zeros(3, N, N)
    for j in 1:N, i in 1:N
        x, y = (i - 1 / 2) * Δx, (j - 1 / 2) * Δx
        u = 1 + sinpi(2x) * sinpi(2y) / 2
        U[1, i, j] = u
        U[2, i, j] = jinxin_flux(u)
        U[3, i, j] = jinxin_flux(u)
    end
    return U
end

"""
    jinxin_run(tab, ε; N = 20, a = 2, t1 = 1/4, cfl = 0.8, partition = nothing)

Integrate to `t1` with `Δt = cfl Δx/(2a)` (two-dimensional first-order
upwind is stable for `Δt ≤ Δx/(2a)`). Returns the final state and what the
example prints: the steps, the change in `sum(u)`, the distance
`max |v − u²/2|, |w − u²/2|` from the equilibrium, the range of `u`, and
the stage-limiter calls.
"""
function jinxin_run(tab, ε; N = 20, a = 2, t1 = 1 / 4, cfl = 0.8, partition = nothing)
    p = JinXin(N, a, ε)
    dt = cfl * p.Δx / (2 * p.a)
    U0 = jinxin_initial(N)
    prob = IMEXProblem(jinxin_rhs!, jinxin_solve!, U0, (0.0, t1), p)
    integ = init(prob, tab; dt, stage_limiter = jinxin_count!, partition)
    solve!(integ)
    U = integ.u
    u = U[1, :, :]
    q = jinxin_flux.(u)
    return (; name = tab.name, ε, steps = integ.nsteps, t = integ.t, U,
            mass = sum(u) - sum(U0[1, :, :]),
            residual = max(maximum(abs.(U[2, :, :] .- q)), maximum(abs.(U[3, :, :] .- q))),
            umin = minimum(u), umax = maximum(u), nlimit = p.nlimit[])
end

function jinxin_main()
    println("Jin–Xin relaxation of 2D Burgers, 20 × 20 cells, a = 2, t = 1/4")
    println(rpad("tableau", 14), rpad("ε", 8), rpad("steps", 7), rpad("Δ sum(u)", 11),
            rpad("max|v − u²/2|", 15), rpad("u range", 18), "limiter calls")
    for tab in (IMEXSSP222(), IMEXSSP2332(), IMEXSSP3433(), ARS443()), ε in (1e-2, 1e-8)
        r = jinxin_run(tab, ε)
        println(rpad(r.name, 14), rpad(string(ε), 8), rpad(string(r.steps), 7),
                rpad(string(round(r.mass; sigdigits = 2)), 11),
                rpad(string(round(r.residual; sigdigits = 3)), 15),
                rpad(string(round(r.umin; digits = 4), " … ", round(r.umax; digits = 4)), 18),
                r.nlimit)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    jinxin_main()
end

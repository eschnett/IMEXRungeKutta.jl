# The device smoke run ("Testing" (Mechanics) in `CODE.md`): a short
# `Float32` run on an `MtlArray` state with scalar indexing disallowed,
# and a `Float32x2` one (MultiFloats' double-`Float32`; added 2026-09-28).
#
# It is not part of `Pkg.test()`, and Metal is not in the package's test
# environment: it has its own, `test/metal/Project.toml`, which develops
# the package from `../..` (proposed in step 4, decided 2026-09-24;
# "Commands" in `CLAUDE.md`):
#
#     julia --project=test/metal -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#     IMEXRUNGEKUTTA_TEST_METAL=1 julia --project=test/metal test/metal_tests.jl
#
# Without `IMEXRUNGEKUTTA_TEST_METAL=1` this file does nothing. With it, a
# Metal that is not functional is a failure, not a skip: the run was asked
# for.
#
# What it covers: the broadcast path end to end on a device, with every
# callback a broadcast too, so that anything that indexes the state, or
# puts a `Float64` into a kernel, fails here. The time stays `Float64` (or
# `Float32`) on the host; only `T`-typed values reach the device.

if get(ENV, "IMEXRUNGEKUTTA_TEST_METAL", "") != "1"
    @info "metal_tests.jl: skipped; set IMEXRUNGEKUTTA_TEST_METAL=1 to run it"
    exit(0)
end

using Test
using Metal: Metal, MtlArray
using MultiFloats: MultiFloats, Float32x2
using IMEXRungeKutta
using IMEXRungeKutta: IMEXProblem

Metal.allowscalar(false)

# Relaxation in each of `N` cells toward `ū` at a rate `1/ε` that runs
# from 10³ to 1 across the cells, so that some cells are stiff at
# `Δt = 0.1` and some are not, with an explicit part that depends on both
# `u` and `t`: `u′ = cos t − κu − (u − ū)/ε`. Every callback is one
# broadcast, and converts the host time to `T` before the kernel sees it.
# MultiFloats has no `cos`, so a Float32x2 time takes it in `BigFloat`, on
# the host.
host_cos(t) = cos(t)
host_cos(t::MultiFloats.MultiFloat) = cos(BigFloat(t))
function f_exp_metal!(du, u, p, t)
    T = eltype(du)
    c = T(host_cos(t))
    κ = p.κ
    @. du = c - κ * u
    return nothing
end

function solve_imp_metal!(U, u★, γΔt, p, t)
    @. U = (u★ + (γΔt / p.ε) * p.ū) / (1 + γΔt / p.ε)
    return nothing
end

# A positivity-type floor, far below the solution, so that it changes
# nothing but still runs as a kernel at every stage and step.
function floor_limiter!(u, integ, p, t)
    fl = p.floor
    @. u = max(u, fl)
    return nothing
end

# The problem's arrays, built on the host in `Float32` and moved with
# `move` (`identity` for the CPU run, `MtlArray` for the device one).
function metal_problem(move, N, tspan)
    x = Float32.((0:(N - 1)) ./ N)
    ū = @. 1 + x
    ε = @. Float32(10)^(-3 + 3x)
    u0 = @. ū + sinpi(2x)
    p = (ū = move(ū), ε = move(ε), κ = 1.0f0 / 2, floor = -10.0f0)
    return IMEXProblem(f_exp_metal!, solve_imp_metal!, move(u0), tspan, p)
end

# The same problem in `T`, a MultiFloat, or `Float64` for the reference:
# the data are formed in `Float64` and rounded to Float32x2, and the
# reference takes those rounded values exactly, so that the two runs
# start from the same state. MultiFloats converts to `Float64` only
# through `BigFloat`.
function metal_problem_mf(move, N, tspan, ::Type{T}) where {T}
    x = (0:(N - 1)) ./ N
    mf(a) = Float32x2.(a)
    to(a) = T === Float64 ? Float64.(BigFloat.(mf(a))) : mf(a)
    ū = to(@. 1 + x)
    ε = to(@. 10.0^(-3 + 3x))
    u0 = to(@. 1 + x + sinpi(2x))
    p = (ū = move(ū), ε = move(ε), κ = T(1 // 2), floor = T(-10))
    return IMEXProblem(f_exp_metal!, solve_imp_metal!, move(u0), tspan, p)
end

tname(Tt) = Tt === Float32x2 ? "Float32x2" : string(Tt)

metal_init(prob, tab) = init(prob, tab; dt = 1 // 10, stage_limiter = floor_limiter!,
                             step_limiter = floor_limiter!)

# The difference of the device state from the CPU state, in units of
# `eps(Float32)` times the largest `|u|`.
function ulps(u_dev, u_cpu)
    return maximum(abs, Array(u_dev) .- u_cpu) / (eps(Float32) * maximum(abs, u_cpu))
end

const METAL_TABLEAUS = (IMEXSSP222, IMEXSSP3433)
const METAL_N = 4096

# `step!` is allocation-free on the CPU; on a device each broadcast is a
# kernel launch, which allocates a little on the host. The claim is that
# the integrator adds nothing of its own, and nothing state-sized.
#
# Metal specializes a broadcast kernel on the array's shape once it has
# seen that shape more than ten times (`BROADCAST_SPECIALIZATION_THRESHOLD`
# in Metal's `broadcast.jl`), so the first steps at a new state size
# compile, and allocate hundreds of MB doing it. Three steps of warm-up get
# past that; the measurement is the least over the next five.
function metal_step_allocations(integ)
    for _ in 1:3
        step!(integ)
    end
    return minimum(_ -> (@allocated step!(integ)), 1:5)
end

# The kernel launches of one step: the package's own combinations
# (`u★`, the copy into `U`, the increment, the update) and the callbacks',
# each of which is one broadcast here.
function metal_launches(integ)
    n = 1 # the update
    for st in integ.plan.stages
        if IMEXRungeKutta.solves(st)
            n += !isempty(st.terms) + 1 + IMEXRungeKutta.implicit_used(st)
        elseif IMEXRungeKutta.explicit_used(st) && !isempty(st.terms)
            n += 1
        end
    end
    calls = IMEXRungeKutta.plan_calls(integ.plan)
    return n + calls.f_exp + calls.solve_imp + calls.stage_limiter + 1
end

# One launch of a two-operand broadcast, for scale, after the same warm-up.
function one_launch_allocations(a, b)
    for _ in 1:20
        a .= b
    end
    return minimum(_ -> (@allocated a .= b), 1:5)
end

@testset "IMEXRungeKutta on Metal" begin
    # Scalar indexing left enabled would let an indexing loop in the package
    # pass here, slowly, and fail on a device where it is not merely slow.
    @testset "Metal is functional and refuses scalar indexing" begin
        @test Metal.functional()
        a = MtlArray(ones(Float32, 4))
        @test_throws ErrorException a[1]
    end

    # A package that indexed the state, formed a combination as a loop, or
    # let a `Float64` coefficient or time into a kernel would fail to run
    # here, or would run and disagree with the CPU.
    @testset "Ten steps on Metal agree with the CPU in Float32: $(make().name), time $Tt" for
            make in METAL_TABLEAUS, Tt in (Float64, Float32)
        tab = make()
        tspan = (zero(Tt), one(Tt))
        dev = metal_init(metal_problem(MtlArray, METAL_N, tspan), tab)
        cpu = metal_init(metal_problem(identity, METAL_N, tspan), tab)
        @test dev.u isa MtlArray{Float32}
        @test dev.nsteps == cpu.nsteps == 10
        @test typeof(dev.t) === Tt
        d = Float64[]
        for n in 1:10
            step!(dev)
            step!(cpu)
            push!(d, ulps(dev.u, cpu.u))
        end
        @test dev.t == cpu.t == one(Tt)
        @test all(isfinite, Array(dev.u))
        # The run is a real one: the state moved by O(1) from `u0`.
        @test maximum(abs, cpu.u .- Array(metal_problem(identity, METAL_N, tspan).u0)) > 0.1
        # Not bitwise in general: the device may contract `a + c*x` to an FMA,
        # and its division need not be correctly rounded. Either changes a
        # rounding, not the arithmetic, so the two runs may differ by at most
        # the sum of their rounding errors, which do not grow: the stiff cells
        # contract, the others grow by 1 + O(Δt). The CPU run in Float32 is
        # 11.5 (SSP2(2,2,2)) and 15.4 (SSP3(4,3,3)) of these units from the
        # same run in Float64 after ten steps (measured in step 4), about 1.5
        # per step. So 4 per step; a wrong coefficient, or the other tableau
        # (41349), is three orders of magnitude above it.
        @test all(n -> d[n] ≤ 4n, 1:10)
        @info "Metal − CPU, in eps(Float32)·max|u|, per step" tab.name Tt d = repr(d)
    end

    # Metal has no `Float64`, so Float32x2 is how a device run gets about
    # 46 bits. A kernel that MultiFloats' double-float arithmetic did not
    # compile to, or a coefficient, `Δt` or time rounded to `Float32` on
    # the way, would fail here or leave the device run a `Float32` error
    # (1e−7) off the `Float64` one, where 1e−14 is due ("On a device" in
    # `CODE.md`; added 2026-09-28).
    @testset "Metal carries Float32x2 precision: $(make().name), time $(tname(Tt))" for
            make in METAL_TABLEAUS, Tt in (Float64, Float32x2)
        tab = make()
        tspan = (zero(Tt), one(Tt))
        dev = metal_init(metal_problem_mf(MtlArray, METAL_N, tspan, Float32x2), tab)
        cpu = metal_init(metal_problem_mf(identity, METAL_N, tspan, Float32x2), tab)
        ref = metal_init(metal_problem_mf(identity, METAL_N, (0.0, 1.0), Float64), tab)
        @test dev.u isa MtlArray{Float32x2}
        @test typeof(dev.t) === Tt
        @test dev.nsteps == cpu.nsteps == ref.nsteps == 10
        wide(u) = Float64.(BigFloat.(Array(u)))
        d_cpu = Float64[]
        d_ref = Float64[]
        for n in 1:10
            step!(dev)
            step!(cpu)
            step!(ref)
            scale = maximum(abs, ref.u)
            push!(d_cpu, maximum(abs, wide(dev.u) .- wide(cpu.u)) /
                         (Float64(BigFloat(eps(Float32x2))) * scale))
            push!(d_ref, maximum(abs, wide(dev.u) .- ref.u) / scale)
        end
        @test dev.t == cpu.t == one(Tt)
        # The device and the CPU run the same double-float arithmetic; any
        # difference is a rounding in a kernel, as in Float32 above.
        @test all(n -> d_cpu[n] ≤ 4n, 1:10)
        # Against `Float64`: about eps(Float32x2) per step, not eps(Float32).
        @test all(n -> d_ref[n] ≤ 1e-12, 1:10)
        @info("Metal Float32x2 − CPU Float32x2, in eps(Float32x2)·max|u|, and − CPU \
               Float64, relative, per step", tab.name, Tt, d_cpu = repr(d_cpu),
              d_ref = repr(d_ref))
    end

    # Reuse on a device checks the scratch's type by inference
    # (`Base.promote_op(similar, …)`), which must accept the device's own
    # arrays and refuse host ones; and a reused plan must run the same
    # kernels as a fresh one ("Scratch reuse" in `CODE.md`).
    @testset "Two chunks reusing the scratch on Metal equal fresh ones: $(make().name)" for
            make in METAL_TABLEAUS
        tab = make()
        function chunks(reuse)
            prob = metal_problem(MtlArray, METAL_N, (0.0, 0.5))
            a = init(prob, tab; dt = 1 // 10, alias_u0 = true,
                     stage_limiter = floor_limiter!, step_limiter = floor_limiter!)
            solve!(a)
            prob2 = IMEXProblem(prob.f_exp!, prob.solve_imp!, a.u, (0.5, 1.0), prob.p)
            b = init(prob2, tab; dt = 1 // 14, alias_u0 = true,
                     reuse = reuse ? a : nothing, stage_limiter = floor_limiter!,
                     step_limiter = floor_limiter!)
            @test all(map(===, b.plan.scratch, a.plan.scratch)) == reuse
            return Array(solve!(b).u)
        end
        @test chunks(true) == chunks(false)
        host = metal_init(metal_problem(identity, METAL_N, (0.0, 1.0)), tab)
        @test_throws ArgumentError init(metal_problem(MtlArray, METAL_N, (0.0, 1.0)), tab;
                                        dt = 1 // 10, reuse = host)
    end

    # A step that copied the state to the host, or allocated scratch per
    # step, would allocate in proportion to the state; Metal's launches
    # allocate a fixed amount each.
    @testset "A Metal step allocates on the host only for its launches: $(make().name)" for
            make in METAL_TABLEAUS
        tab = make()
        tspan = (0.0, 1.0)
        small = metal_init(metal_problem(MtlArray, METAL_N, tspan), tab)
        large = metal_init(metal_problem(MtlArray, 256 * METAL_N, tspan), tab)
        @test (@inferred step!(small)) === nothing
        bytes_small = metal_step_allocations(small)
        bytes_large = metal_step_allocations(large)
        launches = metal_launches(small)
        one = one_launch_allocations(similar(small.u), small.u)
        @info "Metal host allocations per step" tab.name launches bytes_small bytes_large one
        # Nothing state-sized: a host copy of the large state alone would be
        # 4 MiB, some 200 times what a step allocates.
        @test bytes_large ≤ 5 * bytes_small ÷ 4
    end

    # A Float32x2 step that converted on the host per step (through
    # `BigFloat`, the only way MultiFloats converts to other types), or
    # copied the state back, would allocate more than the Float32 step, or
    # in proportion to the state. Metal cannot be allocation-free: each
    # launch allocates on the host, as above.
    @testset "A Float32x2 step on Metal allocates as a Float32 one does: $(make().name)" for
            make in METAL_TABLEAUS
        tab = make()
        tspan = (0.0, 1.0)
        f32 = metal_init(metal_problem(MtlArray, METAL_N, tspan), tab)
        small = metal_init(metal_problem_mf(MtlArray, METAL_N, tspan, Float32x2), tab)
        large = metal_init(metal_problem_mf(MtlArray, 256 * METAL_N, tspan, Float32x2), tab)
        @test (@inferred step!(small)) === nothing
        bytes_f32 = metal_step_allocations(f32)
        bytes_small = metal_step_allocations(small)
        bytes_large = metal_step_allocations(large)
        @info("Metal host allocations per step, Float32x2", tab.name, bytes_f32,
              bytes_small, bytes_large)
        @test bytes_large ≤ 5 * bytes_small ÷ 4
        @test bytes_small ≤ 5 * bytes_f32 ÷ 4
    end
end

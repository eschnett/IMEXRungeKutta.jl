using IMEXRungeKutta: IMEXRungeKutta, IMEXProblem
using IMEXRungeKutta: IMEXSSP222, IMEXSSP3433, RK4
using IMEXRungeKutta: scratch_count

# Scratch reuse: `init(…; reuse = integ′)` ("Scratch reuse" in `CODE.md`).
# A chunked driver builds one integrator per chunk, with the chunk's `Δt`
# and `tspan`, and on an unchanged grid takes the previous one's scratch
# instead of allocating and first-touching its own. This file follows
# `owner_tests.jl`, whose problem, states and partitions it uses.

# A chunked run: `u0` over the chunks `(tᵢ, tᵢ₊₁)` of `ts`, each with its
# own `dt`, one integrator per chunk stepping the one state in place
# (`alias_u0`), as TreeGeneralizedHarmonic's driver does. With `reuse`,
# each chunk's integrator takes the scratch of the one before; the first
# chunk's scratch is filled with NaN after its first step, so a reused
# array whose old value were read would show.
function chunked_run(tab, u0, partition, ts, dts; reuse::Bool)
    u = copy(u0)
    integ = nothing
    for i in 1:(length(ts) - 1)
        prob = IMEXProblem(f_owner!, solve_owner!, u, (ts[i], ts[i + 1]),
                           owner_problem(u).p)
        integ = init(prob, tab; dt = dts[i], partition, alias_u0 = true,
                     reuse = reuse ? integ : nothing, stage_limiter = clamp_limiter!,
                     step_limiter = clamp_limiter!)
        step!(integ)
        i == 1 && foreach(x -> fill!(x, real(eltype(u))(NaN)), integ.plan.scratch)
        solve!(integ)
    end
    return integ
end

const CHUNK_TIMES = (0.0, 0.25, 0.6, 1.0)
const CHUNK_DTS = (0.05, 0.07, 0.1)

# A reused array holding a value the new plan read, a term weighted with
# the old chunk's `Δt`, or a mistimed stage would make a chunked run
# differ from the same run with fresh scratch per chunk.
@testset "Chunks that reuse the scratch give the same bits as fresh ones" begin
    for T in (Float64, Float32), tab in all_tableaus()
        u0 = random_state(T, NSTATE)
        for partition in (nothing, :even, multiset_partition())
            fresh = chunked_run(tab, u0, partition, CHUNK_TIMES, CHUNK_DTS; reuse = false)
            reused = chunked_run(tab, u0, partition, CHUNK_TIMES, CHUNK_DTS; reuse = true)
            @test !any(isnan, reused.u)
            @test bits(reused.u) == bits(fresh.u)
            @test reused.t == fresh.t == 1.0
        end
    end
end

# A reuse that allocated anyway would keep the cost it exists to remove,
# and one that wrote the scratch would first-touch it again for nothing.
@testset "init with reuse takes the old arrays as they are, and writes none" begin
    for tab in all_tableaus(), partition in (nothing, :even)
        u0 = random_state(Float64, NSTATE)
        old = init(owner_problem(u0), tab; dt = 0.1, partition)
        step!(old)
        foreach(x -> fill!(x, NaN), old.plan.scratch)
        new = init(owner_problem(u0), tab; dt = 0.03, partition, reuse = old)
        @test length(new.plan.scratch) == scratch_count(tab)
        @test all(map(===, new.plan.scratch, old.plan.scratch))
        @test all(x -> all(isnan, x), new.plan.scratch)
        @test new.u !== old.u
    end
    # Another tableau with as many scratch arrays: RK4 and SSP2(2,2,2)
    # have five each, in other roles.
    u0 = random_state(Float64, NSTATE)
    old = init(owner_problem(u0), RK4(); dt = 0.1)
    new = init(owner_problem(u0), IMEXSSP222(); dt = 0.1, reuse = old)
    @test all(map(===, new.plan.scratch, old.plan.scratch))
    ref = solve(owner_problem(u0), IMEXSSP222(); dt = 0.1)
    @test bits(solve!(new).u) == bits(ref.u)
    # A view of the state: `similar` of it is a `Vector`, as the scratch is.
    new = init(owner_problem(view(copy(u0), :)), RK4(); dt = 0.1, reuse = old)
    @test all(map(===, new.plan.scratch, old.plan.scratch))
end

# The two integrators share the scratch afterwards; since no scratch value
# carries over between steps, each stays correct when they step in turn.
@testset "Two integrators sharing scratch step in turn as if apart" begin
    for tab in all_tableaus()
        u0 = random_state(Float64, NSTATE)
        a = init(owner_problem(u0), tab; dt = 0.1)
        b = init(owner_problem(u0), tab; dt = 0.05, reuse = a)
        ra = init(owner_problem(u0), tab; dt = 0.1)
        rb = init(owner_problem(u0), tab; dt = 0.05)
        for _ in 1:3
            step!(a)
            step!(b)
            step!(ra)
            step!(rb)
        end
        @test bits(a.u) == bits(ra.u)
        @test bits(b.u) == bits(rb.u)
    end
end

# A misfit reused silently would read out of bounds, mix array types, or
# leave pages on the wrong threads; it must be refused, saying why, and
# never turn into a silent allocation.
@testset "Scratch that does not fit is refused" begin
    u0 = random_state(Float64, NSTATE)
    old = init(owner_problem(u0), RK4(); dt = 0.1)
    rk4_init(u; kwargs...) = init(owner_problem(u), RK4(); dt = 0.1, kwargs...)
    # Not an integrator.
    @test_throws ArgumentError rk4_init(u0; reuse = old.plan.scratch)
    @test_throws ArgumentError rk4_init(u0; reuse = :old)
    # Another scratch count: SSP3(4,3,3) needs eight.
    @test_throws ArgumentError init(owner_problem(u0), IMEXSSP3433(); dt = 0.1,
                                    reuse = old)
    # Another length, element type or array type.
    @test_throws ArgumentError rk4_init(random_state(Float64, NSTATE + 1); reuse = old)
    @test_throws ArgumentError rk4_init(random_state(Float32, NSTATE); reuse = old)
    @test_throws ArgumentError rk4_init(random_state(ComplexF64, NSTATE); reuse = old)
    @test_throws ArgumentError rk4_init(reshape(copy(u0), 1, :); reuse = old)
    # Another partition: by owner against the broadcast, and two by owner
    # with other ranges.
    @test_throws ArgumentError rk4_init(u0; partition = :even, reuse = old)
    owned = rk4_init(u0; partition = :even)
    @test_throws ArgumentError rk4_init(u0; reuse = owned)
    @test_throws ArgumentError rk4_init(u0; partition = irregular_partition(NSTATE),
                                      reuse = owned)
    @test rk4_init(u0; partition = :even, reuse = owned).plan.scratch === owned.plan.scratch
    # The state is one of the scratch arrays.
    @test_throws ArgumentError rk4_init(old.plan.scratch[1]; alias_u0 = true, reuse = old)
end

# The helper is top-level and takes concrete arguments ("Commands" in
# `CLAUDE.md`).
function init_allocations(prob, tab, reuse)
    init(prob, tab; dt = 0.1, alias_u0 = true, reuse)
    return @allocated init(prob, tab; dt = 0.1, alias_u0 = true, reuse)
end
# The cost reuse removes is the state-sized allocations; any that remained
# would scale with the state.
@testset "init with reuse allocates less than one state" begin
    if CHECK_BOUNDS_FORCED
        @info "Skipping the allocation tests under --check-bounds=yes"
    else
        u0 = random_state(Float64, 10^6)
        for tab in all_tableaus()
            prob = owner_problem(u0)
            old = init(prob, tab; dt = 0.1, alias_u0 = true)
            fresh = init_allocations(prob, tab, nothing)
            reused = init_allocations(prob, tab, old)
            @test fresh ≥ scratch_count(tab) * sizeof(u0)
            @test reused < sizeof(u0) ÷ 10
            # And `step!` on the reused integrator stays as it was.
            integ = init(prob, tab; dt = 0.1, alias_u0 = true, reuse = old)
            @test step_allocations(integ) == 0
            @inferred step!(integ)
        end
    end
end

using Test
using IMEXRungeKutta

# One suite, run whole, one file per group of "Testing" in `CODE.md`.
# `CLAUDE.md` ("Commands") has the runs every step makes: one thread, four
# threads, and `--check-bounds=yes`, on Julia 1.10 and on the current
# release.
#
# The bounds-checking mode is printed because it is not always the one
# asked for: on Julia 1.10, `Pkg.test()` forces `--check-bounds=yes` on
# the test process whatever the parent was started with (measured in
# step 0), and allocation tests are skipped under it.
const CHECK_BOUNDS_FORCED = Base.JLOptions().check_bounds == 1
@info "Running the tests on $(Threads.nthreads()) thread(s)" CHECK_BOUNDS_FORCED

# Test-only helpers: the order conditions, stability and SSP properties;
# the mock callbacks, which log their calls; and the fitted order and the
# Kaps problem, for the validation files.
include("tableau_properties.jl")
include("mocks.jl")
include("problems.jl")

@testset "IMEXRungeKutta.jl" begin
    include("scaffold_tests.jl")
    @testset "Tableaus" begin
        include("tableau_tests.jl")
    end
    @testset "Interface" begin
        include("interface_tests.jl")
    end
    @testset "Mechanics" begin
        include("mechanics_tests.jl")
    end
    @testset "Stage arithmetic by owner" begin
        include("owner_tests.jl")
    end
    @testset "Smoke order" begin
        include("smoke_order_tests.jl")
    end
    @testset "README" begin
        include("readme_tests.jl")
    end
    @testset "Order" begin
        include("order_tests.jl")
    end
    @testset "Stiff limit" begin
        include("stiff_tests.jl")
    end
    @testset "Asymptotic preservation" begin
        include("ap_tests.jl")
    end
    @testset "SSP" begin
        include("ssp_tests.jl")
    end
    @testset "Oracle" begin
        include("oracle_tests.jl")
    end
    # Last, after every file has run: the device smoke run is
    # `metal_tests.jl`, in its own environment, and a file here that
    # loaded Metal would make every `Pkg.test()` need it.
    @testset "The ordinary suite loaded no Metal" begin
        @test !any(id -> id.name == "Metal", keys(Base.loaded_modules))
    end
end

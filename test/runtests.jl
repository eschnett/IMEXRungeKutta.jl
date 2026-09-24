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

# Test-only helpers: the order conditions, stability and SSP properties.
include("tableau_properties.jl")

@testset "IMEXRungeKutta.jl" begin
    include("scaffold_tests.jl")
    @testset "Tableaus" begin
        include("tableau_tests.jl")
    end
end

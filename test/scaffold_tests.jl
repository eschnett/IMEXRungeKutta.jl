using CommonSolve: CommonSolve
using TOML: TOML

# The raw text of a module's own docstring. `@doc` cannot be used for this:
# from Julia 1.11 on it returns `nothing` unless REPL is loaded. The docs
# table it reads is the same on 1.10 and later.
function module_docstring(m::Module)
    multidoc = Base.Docs.meta(m)[Base.Docs.Binding(m, nameof(m))]
    return join(only(values(multidoc.docs)).text)
end

# A module shell that does not load, or a leftover of the Pkg template,
# would make every later file fail for a reason that is not its own.
@testset "The package loads as a module documented against CODE.md" begin
    @test IMEXRungeKutta isa Module
    @test nameof(IMEXRungeKutta) === :IMEXRungeKutta
    @test !isdefined(IMEXRungeKutta, :hello)
    @test !isdefined(IMEXRungeKutta, :domath)
    @test occursin("CODE.md", module_docstring(IMEXRungeKutta))
end

# Methods added to functions of our own named `init`, `solve`, `solve!`
# and `step!` would clash with SciMLBase's and OrdinaryDiffEq's, which are
# CommonSolve's, whenever both are loaded ("Dependencies and names" in
# `CODE.md`).
@testset "init, solve, solve! and step! are CommonSolve's bindings" begin
    @test IMEXRungeKutta.init === CommonSolve.init
    @test IMEXRungeKutta.solve === CommonSolve.solve
    @test IMEXRungeKutta.solve! === CommonSolve.solve!
    @test IMEXRungeKutta.step! === CommonSolve.step!
    for name in (:init, :solve, :solve!, :step!)
        @test name in names(IMEXRungeKutta)
    end
end

# Two packages exporting different functions under one name leave that
# name unusable in a module that loads both. Here the second exporter is
# CommonSolve itself, which is what SciMLBase re-exports.
@testset "Loading CommonSolve beside the package leaves the names usable" begin
    m = Module(:ClashCheck)
    Core.eval(m, :(using CommonSolve, IMEXRungeKutta))
    for name in (:init, :solve, :solve!, :step!)
        @test Core.eval(m, name) === getfield(CommonSolve, name)
    end
end

# A run-time dependency added without amending "Requirements" in `CODE.md`,
# or a compat floor above Julia 1.10, would reach every downstream silently.
# A test-only dependency back in the root `Project.toml`'s `[extras]`, or
# its bound in the root `[compat]`, would split the test environment over
# two files again ("File layout" in `CODE.md`).
@testset "The run-time dependencies are CommonSolve alone, on Julia 1.10" begin
    project = TOML.parsefile(joinpath(pkgdir(IMEXRungeKutta), "Project.toml"))
    @test sort(collect(keys(project["deps"]))) == ["CommonSolve"]
    @test sort(collect(keys(project["compat"]))) == ["CommonSolve", "julia"]
    @test project["compat"]["julia"] == "1.10"
    for section in ("sources", "extras", "targets", "weakdeps")
        @test !haskey(project, section)
    end
end

# The test environment is `test/Project.toml` (decided by Erik in step 6,
# "File layout" in `CODE.md`). A dependency missing from it fails a file
# far from here; the oracle without its bound could resolve to a release
# whose tableaus or #4620 behaviour the recorded numbers do not describe.
# `Pkg.test()` adds this package to the environment itself, on 1.10 and
# on 1.13, so the file does not list it. CommonSolve is listed because the
# tests load it by name, which the package's own dependency does not
# allow; its bound is the root `[compat]`'s, through this package.
@testset "The test-only dependencies are test/Project.toml's, the oracle bounded" begin
    test_deps = ["CommonSolve", "LinearAlgebra", "OrdinaryDiffEqSDIRK", "TOML", "Test"]
    root = pkgdir(IMEXRungeKutta)
    test_project = TOML.parsefile(joinpath(root, "test", "Project.toml"))
    @test sort(collect(keys(test_project["deps"]))) == test_deps
    @test test_project["compat"] == Dict("OrdinaryDiffEqSDIRK" => "2.9.6")
    @test !haskey(test_project, "sources")
    # Under `Pkg.test()` the active project is a copy of that file with
    # this package added, and its `[compat]` kept. The load path's
    # `@v#.#` marks a plain `julia --project=.` run instead.
    if "@v#.#" ∉ LOAD_PATH
        active = TOML.parsefile(Base.active_project())
        @test sort(collect(keys(active["deps"]))) ==
              sort([test_deps; "IMEXRungeKutta"])
        @test active["compat"]["OrdinaryDiffEqSDIRK"] == "2.9.6"
    end
end

# Metal in the package's test environment would download and precompile a
# GPU stack on every `Pkg.test()`, on Linux too, for a test that runs only
# on request ("Commands" in `CLAUDE.md`). The resolved environment is
# checked, not just `Project.toml`, since a test dependency could bring it
# in. Under `Pkg.test()` the load path is that environment alone, so
# `find_package` then answers for it; under a plain `julia --project=.` it
# would also search the global environment, which may well have Metal.
@testset "Metal is not in the ordinary test environment" begin
    root = pkgdir(IMEXRungeKutta)
    for file in (joinpath(root, "Project.toml"), joinpath(root, "test", "Project.toml"))
        project = TOML.parsefile(file)
        for section in ("deps", "weakdeps", "extras")
            @test !haskey(get(project, section, Dict()), "Metal")
        end
    end
    manifest = Base.project_file_manifest_path(Base.active_project())
    @test manifest !== nothing
    manifest === nothing || @test !haskey(TOML.parsefile(manifest)["deps"], "Metal")
    if "@v#.#" ∉ LOAD_PATH
        @test Base.find_package("Metal") === nothing
    end
end

# The Metal environment is run by hand, so nothing else would notice it
# naming a different package, or no longer pointing at this checkout.
@testset "test/metal/Project.toml develops this package and adds Metal" begin
    root = pkgdir(IMEXRungeKutta)
    project = TOML.parsefile(joinpath(root, "Project.toml"))
    metal = TOML.parsefile(joinpath(root, "test", "metal", "Project.toml"))
    @test metal["deps"]["IMEXRungeKutta"] == project["uuid"]
    @test sort(collect(keys(metal["deps"]))) == ["IMEXRungeKutta", "Metal", "Test"]
    source = joinpath(root, "test", "metal", metal["sources"]["IMEXRungeKutta"]["path"])
    @test realpath(source) == realpath(root)
    @test isfile(joinpath(root, "test", "metal_tests.jl"))
end

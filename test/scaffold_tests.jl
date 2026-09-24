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
@testset "The run-time dependencies are CommonSolve alone, on Julia 1.10" begin
    project = TOML.parsefile(joinpath(pkgdir(IMEXRungeKutta), "Project.toml"))
    @test sort(collect(keys(project["deps"]))) == ["CommonSolve"]
    @test project["compat"]["julia"] == "1.10"
    @test !haskey(project, "sources")
end

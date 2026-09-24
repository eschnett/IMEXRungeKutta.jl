using LinearAlgebra: LinearAlgebra

# The worked example in `README.md` must run as written: a README whose
# example has gone stale is worse than none. Its `julia` code blocks are
# extracted and evaluated, in order, in a fresh module, and the result is
# checked against the relaxation's quasi-steady state.

function readme_code()
    text = read(joinpath(pkgdir(IMEXRungeKutta), "README.md"), String)
    blocks = [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, text)]
    return blocks
end

@testset "The README's worked example runs as written" begin
    blocks = readme_code()
    @test !isempty(blocks)
    m = Module(:ReadmeExample)
    for block in blocks
        Base.include_string(m, block, "README.md")
    end
    integ = Core.eval(m, :integ)
    p = Core.eval(m, :p)
    @test integ isa IMEXRungeKutta.IMEXIntegrator
    @test integ.t === 1.0 && integ.nstep == integ.nsteps
    # Relaxed to ū, displaced by the forcing: u ≈ ū + ε cos(t), to O(ε²)
    # and the scheme's error.
    @test maximum(abs.(integ.u .- (p.ū .+ p.ε * cos(1.0)))) < 1e-2 * p.ε
    # The README's remark on SSP3(4,3,3): not stiffly accurate, so in the
    # stiff limit each step ends off the quasi-steady state by
    # `Δt (b̃ᵀ − bᵀA⁻¹Ã) f`, which for this `f` is `Δt (1 − bᵀA⁻¹c̃) cos t`
    # to leading order, `−0.2844 Δt cos t`.
    prob = Core.eval(m, :prob)
    ssp = solve(prob, IMEXSSP3433(); dt = 0.01)
    tab = IMEXSSP3433()
    A, b, c̃ = Float64.(tab.A), Float64.(tab.b), Float64.(tab.c̃)
    predicted = 1 - b' * (A \ c̃)
    @test abs(predicted + 0.2844) < 1e-4
    displacement = (ssp.u .- p.ū) ./ (ssp.dt * cos(1.0))
    @test maximum(abs.(displacement .- predicted)) < 0.01
end

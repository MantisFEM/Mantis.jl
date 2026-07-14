module MantisTests

using Pkg
using Test

@testset verbose=true "GeneralHelpers" begin include("GeneralHelpers/runtests.jl") end
@testset verbose=true "Mesh" begin include("Mesh/runtests.jl") end
@testset verbose=true "Points" begin include("Points/runtests.jl") end
@testset verbose=true "Geometry" begin include("Geometry/runtests.jl") end
@testset verbose=true "FunctionSpaces" begin include("FunctionSpaces/runtests.jl") end
@testset verbose=true "Quadrature" begin include("Quadrature/runtests.jl") end
@testset verbose=true "Forms" begin include("Forms/runtests.jl") end
@testset verbose=true "Assembly" begin include("Assemblers/runtests.jl") end
@testset verbose=true "Plot" begin include("Plot/runtests.jl") end

# Do not run JET tests on pre-release versions, as JET is too unstable and the CI will
# appear as failing. Also only run it from v1.12 onwards, also for compatibility reasons.
@static if VERSION >= v"1.12"
    if isempty(VERSION.prerelease)
        Pkg.activate("Inference")
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @testset verbose=true "Inference Tests" include("Inference/runtests.jl")
    end
end

end; nothing

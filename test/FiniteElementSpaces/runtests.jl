module FiniteElementSpacesTests

using Test

@testset "BezierExtraction" begin include("BezierExtractionTest.jl") end
@testset "UnivariateSplineSpaces" begin include("UnivariateSplineTests.jl") end

end
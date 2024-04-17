module FiniteElementSpacesTests

using Test

@testset "BezierExtraction" begin include("BezierExtractionTest.jl") end
@testset "UnivariateSplineSpaces" begin include("UnivariateSplineTests.jl") end
@testset "TensorProductTests" begin include("TensorProductTests.jl") end
# @testset "KnotInsertion" begin include("KnotInsertionTest.jl") end

end
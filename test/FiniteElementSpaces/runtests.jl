module FiniteElementSpacesTests

using Test

@testset "BezierExtraction" begin include("BezierExtractionTest.jl") end
@testset "KnotInsertion" begin include("KnotInsertionTest.jl") end
@testset "UnivariateSplineSpaces" begin include("UnivariateSplineTests.jl") end
@testset "TensorProductTests" begin include("TensorProductTests.jl") end
@testset "TensorProductTwoScaleRelations" begin include("TensorProductTwoScaleRelationsTest.jl") end
@testset "HierarchicalFiniteElementSpaces" begin include("HierarchicalFiniteElementSpaces/runtests.jl") end

end
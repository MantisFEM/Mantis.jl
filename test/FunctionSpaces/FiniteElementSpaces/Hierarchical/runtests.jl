module HierarchicalFiniteElementSpacesTests

using Test

# @testset "HierarchicalBSplines" begin
#     include("HierarchicalBSplineTests.jl")
# end
# @testset "TensorProductHBSplines" begin
#     include("TensorProductHBSplineTests.jl")
# end
# @testset "TensorProductTHBSplines" begin
#     include("TensorProductTHBSplineTests.jl")
# end
# @testset "HierarchicalMultiComponent" begin
#     include("HierarchicalMultiComponentTests.jl")
# end

@testset "Refinement" verbose = true begin
    include("RefinementTests.jl")
end

@testset "Scalings" verbose = true begin
    include("ScalingsTests.jl")
end

@testset "Basis" verbose = true begin
    include("BasisTests.jl")
end

@testset "Extraction" verbose = true begin
    include("ExtractionTests.jl")
end

end

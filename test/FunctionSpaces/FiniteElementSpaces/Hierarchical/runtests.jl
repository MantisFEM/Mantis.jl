module HierarchicalFiniteElementSpacesTests

using Test

@testset "HierarchicalBSplines" begin
    include("HierarchicalBSplineTests.jl")
end
@testset "TensorProductHBSplines" begin
    include("TensorProductHBSplineTests.jl")
end
@testset "TensorProductTHBSplines" begin
    include("TensorProductTHBSplineTests.jl")
end
@testset "HierarchicalMultiComponent" begin
    include("HierarchicalMultiComponentTests.jl")
end

end

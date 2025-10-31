module TwoScaleRelationsTests

using Test

@testset "TensorProductTwoScaleRelations" begin
    include("TensorProductTwoScaleRelationsTests.jl")
end
@testset "UnstructuredTwoScaleRelations" begin
    include("UnstructuredTwoScaleRelationsTests.jl")
end

end

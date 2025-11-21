module FiniteElementSpacesTests

using Test

@testset verbose = true "UnivariateSplines" begin
    include("UnivariateSplines/runtests.jl")
end

@testset verbose = true "TensorProductSpace" begin
    include("TensorProductSpaces/runtests.jl")
end

# @testset verbose=true "HierarchicalSpaces" begin include("Hierarchical/runtests.jl") end

@testset verbose = true "OtherSpaces" begin
    include("OtherSpaces/runtests.jl")
end

# @testset verbose=true "TwoScaleRelations" begin include("TwoScaleRelations/runtests.jl") end

@testset verbose = true "UnstructuredSpaces" begin
    include("UnstructuredSpaces/runtests.jl")
end

end

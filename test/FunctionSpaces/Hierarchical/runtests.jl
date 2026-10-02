module HierarchicalFiniteElementSpacesTests

using Test

@testset "Refinement" verbose = true begin
    include("RefinementTests.jl")
end

@testset "Scalings" verbose = true begin
    include("ScalingsTests.jl")
end

# Hierarchical Space

@testset "Basis" verbose = true begin
    include("BasisTests.jl")
end

@testset "Extraction" verbose = true begin
    include("ExtractionTests.jl")
end

@testset "HierarchicalSpace" verbose = true begin
    include("HierarchicalSpace.jl")
end

end

module HierarchicalTests

using Test
    
@testset "Scalings" verbose = true begin
	include("ScalingsTests.jl")
end

@testset "Refinement" verbose = true begin
	include("RefinementTests.jl")
end

@testset "HierarchicalGeometry" verbose = true begin
	include("HierarchicalGeometryTests.jl")
end

end

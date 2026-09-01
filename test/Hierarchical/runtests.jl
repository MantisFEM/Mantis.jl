module HierarchicalTests

using Test

@testset "Scaling" verbose = true begin
	include("ScalingTests.jl")
end

@testset "ActiveInfo" verbose = true begin
	include("ActiveInfoTests.jl")
end

@testset "Hierarchy" verbose = true begin
	include("HierarchyTests.jl")
end

end

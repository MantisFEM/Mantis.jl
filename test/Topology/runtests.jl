module TopologyTests

using Test

@testset "Topology" verbose=true begin
    @testset "Patches" verbose=true begin
        include("PatchesTests.jl")
    end
    @testset "MeshTopology" verbose=true begin
        include("MeshTopologyTests.jl")
    end
    @testset "SkeletonTopology" verbose=true begin
        include("SkeletonTopologyTests.jl")
    end
end

end

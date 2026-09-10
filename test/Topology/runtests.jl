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
    @testset "TensorProductTopology" verbose=true begin
        include("TensorProductTopologyTests.jl")
    end
end

end

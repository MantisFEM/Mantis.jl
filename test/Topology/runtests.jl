module TopologyTests

using Test

@testset "Topology" verbose=true begin
    @testset "Patches" verbose=true begin
        include("PatchesTests.jl")
    end
    @testset "MeshTopology" verbose=true begin
        include("MeshTopologyTests.jl")
    end
    #include("TopologyTests.jl")
end

end

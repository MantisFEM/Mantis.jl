module MeshTests

using Test

@testset "PatchInterval" begin include("PatchIntervalTests.jl") end
@testset "MeshTopology" begin include("MeshTopologyTests.jl") end

end
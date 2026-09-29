module TopologyInferenceTests

using Mantis
using Test
using JET

# 1D
# Single-patch
@test_opt Topology.MeshTopology([(1, 2)], Topology.LINE, Val(1))
topology_1patch_1D = Topology.MeshTopology([(1, 2)], Topology.LINE, Val(1))
# Multi-patch with flip
# @test_opt Topology.MeshTopology([(1, 2), (3, 2), (3, 4)], Topology.LINE, Val(3))
topology_3patch_1D = Topology.MeshTopology([(1, 2), (3, 2), (3, 4)], Topology.LINE, Val(3))
# Larger multi-patch, periodic
# @test_opt Topology.MeshTopology(
#     ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)),
#     Topology.LINE,
# )
topology_10patch_1D = Topology.MeshTopology(
    ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)),
    Topology.LINE,
)

# 2D
# Single-patch
# @test_opt Topology.MeshTopology([(1, 2, 3, 4)], Topology.QUAD, Val(1))
topology_1patch_2D = Topology.MeshTopology([(1, 2, 3, 4)], Topology.QUAD, Val(1))
# Multi-patch with same origin
# @test_opt Topology.MeshTopology(((1, 2, 3, 4), (1, 4, 5, 6), (1, 6, 7, 8)), Topology.QUAD)
topology_3patch_2D = Topology.MeshTopology(
    ((1, 2, 3, 4), (1, 4, 5, 6), (1, 6, 7, 8)), Topology.QUAD
)
# Larger multi-patch with hole
# @test_opt Topology.MeshTopology(
#     [
#         (1, 2, 3, 4), # 1
#         (2, 5, 6, 3), # 2
#         (5, 7, 8, 6), # 3
#         (6, 8, 9, 10), # 4
#         (10, 9, 11, 12), # 5
#         (13, 10, 12, 14), # 6
#         (15, 13, 14, 16), # 7
#         (4, 3, 13, 15), # 8
#     ],
#     Topology.QUAD,
#     Val(8),
# )
topology_8patch_2D = Topology.MeshTopology(
    [
        (1, 2, 3, 4), # 1
        (2, 5, 6, 3), # 2
        (5, 7, 8, 6), # 3
        (6, 8, 9, 10), # 4
        (10, 9, 11, 12), # 5
        (13, 10, 12, 14), # 6
        (15, 13, 14, 16), # 7
        (4, 3, 13, 15), # 8
    ],
    Topology.QUAD,
    Val(8),
)

# 3D
# Single-patch
# @test_opt Topology.MeshTopology(((1, 2, 3, 4, 5, 6, 7, 8),), Topology.HEX)
topology_1patch_3D = Topology.MeshTopology(((1, 2, 3, 4, 5, 6, 7, 8),), Topology.HEX)

const topologies = (
    topology_1patch_1D,
    topology_3patch_1D,
    topology_10patch_1D,
    topology_1patch_2D,
    topology_3patch_2D,
    topology_8patch_2D,
)

for topology in topologies
    @test_opt Topology.get_manifold_dim(topology)
    @test_opt Topology.get_incidence_relations_dim(topology)
    @test_opt Topology.get_num_patches(topology)
    @test_opt Topology.get_patch_type(topology)
    @test_opt Topology.get_topological_patch(topology)
    @test_opt size(topology)
    @test_opt size(topology, 2)
    @test_opt Topology.get_local_size(topology)
    @test_opt Topology.get_local_size(topology, 2)
    @test_opt lastindex(topology)
    @test_opt Topology.get_global_id(topology, 1, 0, 2, 1)
    @test_opt Topology.get_global_id(topology, 1, 1, 0)
    @test_opt Topology.get_local_id(topology, 1, 1, 0)
    @test_opt Topology.compute_neighbours(topology, 1, 2, 0, true)
    @test_opt Topology.get_interfaces_on_boundary(topology)

    if Topology.get_manifold_dim(topology) >= 2
        @test_opt Topology.compute_neighbours(topology, 1, 2, 1, true)
    end
    if Topology.get_manifold_dim(topology) >= 3
        @test_opt Topology.compute_neighbours(topology, 1, 2, 2, true)
    end
end

end

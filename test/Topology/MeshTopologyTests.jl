module MeshTopologyTestModule

using Test
using Mantis
import MeshCore

include("../TestHelpers.jl")

############################################################################################
#                                         Helpers                                          #
############################################################################################
"""
    line_grid(num_patches)

Return the patches of a 1D mesh of `num_patches` line segments, with vertices numbered
consecutively along the line.
"""
line_grid(num_patches) = [(i, i + 1) for i in 1:num_patches]

"""
    quad_grid(nx, ny)

Return the patches of a 2D `nx × ny` mesh of quadrilaterals, with vertices numbered
lexicographically and each patch listed in counter-clockwise order.
"""
function quad_grid(nx, ny)
    vertex(i, j) = (j - 1) * (nx + 1) + i
    return [
        (vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)) for
        j in 1:ny for i in 1:nx
    ]
end

"""
    hex_grid(nx, ny, nz)

Return the patches of a 3D `nx × ny × nz` mesh of hexahedra, with vertices numbered
lexicographically and each patch listed in the local vertex order of [`Topology.Hex`](@ref).
"""
function hex_grid(nx, ny, nz)
    vertex(i, j, k) = (k - 1) * (nx + 1) * (ny + 1) + (j - 1) * (nx + 1) + i
    return [
        (
            vertex(i, j, k),
            vertex(i + 1, j, k),
            vertex(i + 1, j + 1, k),
            vertex(i, j + 1, k),
            vertex(i, j, k + 1),
            vertex(i + 1, j, k + 1),
            vertex(i + 1, j + 1, k + 1),
            vertex(i, j + 1, k + 1),
        ) for k in 1:nz for j in 1:ny for i in 1:nx
    ]
end

############################################################################################
#                                   Construction errors                                    #
############################################################################################
@testset "construction errors" begin
    # Incorrect type without vertices. The BoundsError is thrown in MeshCore.
    @test_throws MethodError Topology.MeshTopology([()], Topology.LINE, Val(0))
    # Correct type but no actual vertices. The BoundsError is thrown in MeshCore.
    @test_throws BoundsError Topology.MeshTopology(NTuple{2, Int}[], Topology.LINE, Val(0))
    # Invalid (0) vertex id.
    @test_throws ArgumentError Topology.MeshTopology(((0, 1),), Topology.LINE)
    # Invalid negative vertex id.
    @test_throws ArgumentError Topology.MeshTopology([(-1, 1)], Topology.LINE, Val(1))
    # Vertex 3 does not belong to any patch.
    @test_throws ArgumentError Topology.MeshTopology(((1, 2), (4, 5)), Topology.LINE)
    # Vertex 6 does not belong to any patch.
    @test_throws ArgumentError Topology.MeshTopology(
        [(1, 2, 3, 4), (2, 5, 7, 3)], Topology.QUAD, Val(2)
    )
    # Vertex 10 does not belong to any patch.
    @test_throws ArgumentError Topology.MeshTopology(
        [(1, 2, 3, 4, 5, 6, 7, 8), (2, 9, 11, 3, 5, 12, 13, 7)], Topology.HEX, Val(2)
    )
    # The number of vertices per patch has to match the topological patch.
    @test_throws MethodError Topology.MeshTopology(((1, 2, 3, 4),), Topology.LINE)
    @test_throws MethodError Topology.MeshTopology([(1, 2)], Topology.QUAD, Val(1))
    @test_throws MethodError Topology.MeshTopology(((1, 2, 3, 4),), Topology.HEX)
    # The provided number of patches must match the actual number of patches.
    @test_throws ArgumentError Topology.MeshTopology([(1, 2)], Topology.LINE, Val(3))
    @test_throws ArgumentError Topology.MeshTopology(
        [(1, 2), (2, 3), (3, 4), (4, 5)], Topology.LINE, Val(2)
    )
end

############################################################################################
#                                            1D                                            #
############################################################################################
@testset "1D" verbose=true begin
    @testset "single patch: 1 --> 2" verbose=true begin
        topology = Topology.MeshTopology([(1, 2)], Topology.LINE, Val(1))

        @test Topology.get_manifold_dim(topology) == 1
        @test Topology.get_incidence_relations_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 1
        @test Topology.get_patch_type(topology) == typeof(Topology.LINE)
        @test Topology.get_topological_patch(topology) == Topology.LINE
        @test size(topology) == (2, 1)
        @test @expected Dict(1 => 2, 2 => 1) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (2, 1)
        @test @expected Dict(1 => 2, 2 => 1) k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 2

        expected_ir = Dict(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [[1], [1]],
            (2, 1) => [[1, 2]],
            (2, 2) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_global_ids = Dict(
            (1, 1, 1, 0) => 1, (1, 1, 2, 0) => 2, (1, 0, 1, 1) => 1, (2, 0, 1, 1) => 1
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict((1, 1, 0) => 1, (1, 2, 0) => 2)
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict((1, 1, 0) => 1, (1, 2, 0) => 2)
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict(
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1; 2; 0; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        # A single patch has only boundary vertices.
        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [(0, 1), (0, 2)]
        @test isempty(interfaces)
        @test isempty(Topology.get_interfaces_on_boundary(topology))
    end

    @testset "two patches: 1 --> 2 --> 3" verbose=true begin
        topology = Topology.MeshTopology([(1, 2), (2, 3)], Topology.LINE, Val(2))

        @test Topology.get_manifold_dim(topology) == 1
        @test Topology.get_incidence_relations_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.LINE)
        @test Topology.get_topological_patch(topology) == Topology.LINE
        @test size(topology) == (3, 2)
        @test @expected Dict(1 => 3, 2 => 2) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (2, 1)
        @test @expected Dict(1 => 2, 2 => 1) k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 2

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [[1], [1, 2], [2]],
            (2, 1) => [[1, 2], [2, 3]],
            (2, 2) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            (1, 1, 1, 0) => 1,
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 2,
            (2, 1, 2, 0) => 3,
            (1, 0, 1, 1) => 1,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 2,
            (3, 0, 1, 1) => 2,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1, (1, 2, 0) => 2, (2, 1, 0) => 2, (2, 2, 0) => 3
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1, (1, 2, 0) => 2, (2, 2, 0) => 1, (2, 3, 0) => 2
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => [2; 1; 0; 1;;],
            (2, 1, 0, false) => [1; 2; 0; 1;;],
            (2, 2, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1 2; 2 1; 0 0; 1 1],
            (2, 1, 0, true) => [1 2; 2 1; 0 0; 1 1],
            (2, 2, 0, true) => [2; 2; 0; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [(0, 1), (0, 3)]
        @test interfaces == [(0, 2)]
        @test isempty(Topology.get_interfaces_on_boundary(topology))
    end

    @testset "flipped patch: 1 --> 2 <-- 3 --> 4" verbose=true begin
        topology = Topology.MeshTopology(((1, 2), (3, 2), (3, 4)), Topology.LINE)

        @test Topology.get_manifold_dim(topology) == 1
        @test Topology.get_incidence_relations_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 3
        @test Topology.get_patch_type(topology) == typeof(Topology.LINE)
        @test Topology.get_topological_patch(topology) == Topology.LINE
        @test size(topology) == (4, 3)
        @test @expected Dict(1 => 4, 2 => 3) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (2, 1)
        @test @expected Dict(1 => 2, 2 => 1) k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 2

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [[1], [1, 2], [2, 3], [3]],
            (2, 1) => [[1, 2], [3, 2], [3, 4]],
            (2, 2) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]
        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            (1, 1, 1, 0) => 1,
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 3,
            (2, 1, 2, 0) => 2,
            (3, 1, 1, 0) => 3,
            (3, 1, 2, 0) => 4,
            (1, 0, 1, 1) => 1,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 2,
            (3, 0, 1, 1) => 2,
            (3, 0, 2, 1) => 3,
            (4, 0, 1, 1) => 3,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 1, 0) => 3,
            (2, 2, 0) => 2,
            (3, 1, 0) => 3,
            (3, 2, 0) => 4,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 3, 0) => 1,
            (2, 2, 0) => 2,
            (3, 3, 0) => 1,
            (3, 4, 0) => 2,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => [2; 2; 0; 1;;],
            (2, 1, 0, false) => [3; 1; 0; 1;;],
            (2, 2, 0, false) => [1; 2; 0; 1;;],
            (3, 1, 0, false) => [2; 1; 0; 1;;],
            (3, 2, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1 2; 2 2; 0 0; 1 1],
            (2, 1, 0, true) => [2 3; 1 1; 0 0; 1 1],
            (2, 2, 0, true) => [1 2; 2 2; 0 0; 1 1],
            (3, 1, 0, true) => [2 3; 1 1; 0 0; 1 1],
            (3, 2, 0, true) => [3; 2; 0; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [(0, 1), (0, 4)]
        @test interfaces == [(0, 2), (0, 3)]
        @test isempty(Topology.get_interfaces_on_boundary(topology))
    end

    @testset "periodic: 1 --> 2 --> 3 --> 1" verbose=true begin
        topology = Topology.MeshTopology(((1, 2), (2, 3), (3, 1)), Topology.LINE)

        @test Topology.get_manifold_dim(topology) == 1
        @test Topology.get_incidence_relations_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 3
        @test Topology.get_patch_type(topology) == typeof(Topology.LINE)
        @test Topology.get_topological_patch(topology) == Topology.LINE
        @test size(topology) == (3, 3)
        @test @expected Dict(1 => 3, 2 => 3) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (2, 1)
        @test @expected Dict(1 => 2, 2 => 1) k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 2

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [[1, 3], [1, 2], [2, 3]],
            (2, 1) => [[1, 2], [2, 3], [3, 1]],
            (2, 2) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]
        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            (1, 1, 1, 0) => 1,
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 2,
            (2, 1, 2, 0) => 3,
            (3, 1, 1, 0) => 3,
            (3, 1, 2, 0) => 1,
            (1, 0, 1, 1) => 1,
            (1, 0, 2, 1) => 3,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 2,
            (3, 0, 1, 1) => 2,
            (3, 0, 2, 1) => 3,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 1, 0) => 2,
            (2, 2, 0) => 3,
            (3, 1, 0) => 3,
            (3, 2, 0) => 1,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 2, 0) => 1,
            (2, 3, 0) => 2,
            (3, 3, 0) => 1,
            (3, 1, 0) => 2,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)
        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            (1, 1, 0, false) => [3; 2; 0; 1;;],
            (1, 2, 0, false) => [2; 1; 0; 1;;],
            (2, 1, 0, false) => [1; 2; 0; 1;;],
            (2, 2, 0, false) => [3; 1; 0; 1;;],
            (3, 1, 0, false) => [2; 2; 0; 1;;],
            (3, 2, 0, false) => [1; 1; 0; 1;;],
            (1, 1, 0, true) => [1 3; 1 2; 0 0; 1 1],
            (1, 2, 0, true) => [1 2; 2 1; 0 0; 1 1],
            (2, 1, 0, true) => [1 2; 2 1; 0 0; 1 1],
            (2, 2, 0, true) => [2 3; 2 1; 0 0; 1 1],
            (3, 1, 0, true) => [2 3; 2 1; 0 0; 1 1],
            (3, 2, 0, true) => [1 3; 1 2; 0 0; 1 1],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test isempty(boundaries)
        @test interfaces == [(0, 1), (0, 2), (0, 3)]
        @test isempty(Topology.get_interfaces_on_boundary(topology))
    end
end

############################################################################################
#                                            2D                                            #
############################################################################################
@testset "2D" verbose=true begin
    @testset "single patch" verbose=true begin
        # 4 -- 3
        # |    |
        # 1 -- 2
        topology = Topology.MeshTopology([(1, 2, 3, 4)], Topology.QUAD, Val(1))

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 1
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (4, 4, 1)
        @test @expected Dict(1 => 4, 2 => 4, 3 => 1) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (4, 4, 1)
        expected_local_size_dim = Dict(1 => 4, 2 => 4, 3 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 3

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [[1, 2], [1, 3], [3, 4], [2, 4]],
            (1, 3) => [[1], [1], [1], [1]],
            (2, 1) => [[1, 2], [1, 4], [2, 3], [4, 3]],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [[1], [1], [1], [1]],
            (3, 1) => [[1, 2, 3, 4]],
            (3, 2) => [[2, 3, 1, 4]],
            (3, 3) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]
        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            (1, 0, 1, 1) => 1, # topology[1, 2]
            (1, 0, 2, 1) => 2,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 3,
            (3, 0, 1, 1) => 3,
            (3, 0, 2, 1) => 4,
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 4,
            (1, 0, 1, 2) => 1, # topology[1, 3]
            (2, 0, 1, 2) => 1,
            (3, 0, 1, 2) => 1,
            (4, 0, 1, 2) => 1,
            (1, 1, 1, 0) => 1, # topology[2, 1]
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 1,
            (2, 1, 2, 0) => 4,
            (3, 1, 1, 0) => 2,
            (3, 1, 2, 0) => 3,
            (4, 1, 1, 0) => 4,
            (4, 1, 2, 0) => 3,
            (1, 1, 1, 2) => 1, # topology[2, 3]
            (2, 1, 1, 2) => 1,
            (3, 1, 1, 2) => 1,
            (4, 1, 1, 2) => 1,
            (1, 2, 1, 0) => 1, # topology[3, 1]
            (1, 2, 2, 0) => 2,
            (1, 2, 3, 0) => 3,
            (1, 2, 4, 0) => 4,
            (1, 2, 1, 1) => 2, # topology[3, 2]
            (1, 2, 2, 1) => 3,
            (1, 2, 3, 1) => 1,
            (1, 2, 4, 1) => 4,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (1, 1, 1) => 2,
            (1, 2, 1) => 3,
            (1, 3, 1) => 1,
            (1, 4, 1) => 4,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (1, 1, 1) => 3,
            (1, 2, 1) => 1,
            (1, 3, 1) => 2,
            (1, 4, 1) => 4,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)
        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1; 2; 0; 1;;],
            (1, 3, 0, true) => [1; 3; 0; 1;;],
            (1, 4, 0, true) => [1; 4; 0; 1;;],
            # Edge neighbours
            (1, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 1, true) => [1; 1; 0; 1;;],
            (1, 2, 1, true) => [1; 2; 0; 1;;],
            (1, 3, 1, true) => [1; 3; 0; 1;;],
            (1, 4, 1, true) => [1; 4; 0; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [(1, 1), (1, 2), (1, 3), (1, 4), (0, 1), (0, 2), (0, 3), (0, 4)]
        @test isempty(interfaces)
        @test isempty(Topology.get_interfaces_on_boundary(topology))
    end

    @testset "two patches: same orientation" verbose=true begin
        # 4 -- 3 -- 6
        # |    |    |
        # 1 -- 2 -- 5
        # Origin patch 1 is vertex 1, origin patch 2 is vertex 2
        topology = Topology.MeshTopology(
            [(1, 2, 3, 4), (2, 5, 6, 3)], Topology.QUAD, Val(2)
        )

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (6, 7, 2)
        @test @expected Dict(1 => 6, 2 => 7, 3 => 2) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (4, 4, 1)
        expected_local_size_dim = Dict(1 => 4, 2 => 4, 3 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 3

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [[1, 2], [1, 3, 4], [3, 5, 6], [2, 5], [4, 7], [6, 7]],
            (1, 3) => [[1], [1, 2], [1, 2], [1], [2], [2]],
            (2, 1) => [[1, 2], [1, 4], [2, 3], [2, 5], [4, 3], [3, 6], [5, 6]],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [[1], [1], [1, 2], [2], [1], [2], [2]],
            (3, 1) => [[1, 2, 3, 4], [2, 5, 6, 3]],
            (3, 2) => [[2, 3, 1, 5], [3, 7, 4, 6]],
            (3, 3) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            (1, 0, 1, 1) => 1, # topology[1, 2]
            (1, 0, 2, 1) => 2,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 3,
            (2, 0, 3, 1) => 4,
            (3, 0, 1, 1) => 3,
            (3, 0, 2, 1) => 5,
            (3, 0, 3, 1) => 6,
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 5,
            (5, 0, 1, 1) => 4,
            (5, 0, 2, 1) => 7,
            (6, 0, 1, 1) => 6,
            (6, 0, 2, 1) => 7,
            (1, 0, 1, 2) => 1, # topology[1, 3]
            (2, 0, 1, 2) => 1,
            (2, 0, 2, 2) => 2,
            (3, 0, 1, 2) => 1,
            (3, 0, 2, 2) => 2,
            (4, 0, 1, 2) => 1,
            (5, 0, 1, 2) => 2,
            (6, 0, 1, 2) => 2,
            (1, 1, 1, 0) => 1, # topology[2, 1]
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 1,
            (2, 1, 2, 0) => 4,
            (3, 1, 1, 0) => 2,
            (3, 1, 2, 0) => 3,
            (4, 1, 1, 0) => 2,
            (4, 1, 2, 0) => 5,
            (5, 1, 1, 0) => 4,
            (5, 1, 2, 0) => 3,
            (6, 1, 1, 0) => 3,
            (6, 1, 2, 0) => 6,
            (7, 1, 1, 0) => 5,
            (7, 1, 2, 0) => 6,
            (1, 1, 1, 2) => 1, # topology[2, 3]
            (2, 1, 1, 2) => 1,
            (3, 1, 1, 2) => 1,
            (3, 1, 2, 2) => 2,
            (4, 1, 1, 2) => 2,
            (5, 1, 1, 2) => 1,
            (6, 1, 1, 2) => 2,
            (7, 1, 1, 2) => 2,
            (1, 2, 1, 0) => 1, # topology[3, 1]
            (1, 2, 2, 0) => 2,
            (1, 2, 3, 0) => 3,
            (1, 2, 4, 0) => 4,
            (2, 2, 1, 0) => 2,
            (2, 2, 2, 0) => 5,
            (2, 2, 3, 0) => 6,
            (2, 2, 4, 0) => 3,
            (1, 2, 1, 1) => 2, # topology[3, 2]
            (1, 2, 2, 1) => 3,
            (1, 2, 3, 1) => 1,
            (1, 2, 4, 1) => 5,
            (2, 2, 1, 1) => 3,
            (2, 2, 2, 1) => 7,
            (2, 2, 3, 1) => 4,
            (2, 2, 4, 1) => 6,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)

        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (2, 1, 0) => 2,
            (2, 2, 0) => 5,
            (2, 3, 0) => 6,
            (2, 4, 0) => 3,
            (1, 1, 1) => 2,
            (1, 2, 1) => 3,
            (1, 3, 1) => 1,
            (1, 4, 1) => 5,
            (2, 1, 1) => 3,
            (2, 2, 1) => 7,
            (2, 3, 1) => 4,
            (2, 4, 1) => 6,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (2, 2, 0) => 1,
            (2, 5, 0) => 2,
            (2, 6, 0) => 3,
            (2, 3, 0) => 4,
            (1, 1, 1) => 3,
            (1, 2, 1) => 1,
            (1, 3, 1) => 2,
            (1, 5, 1) => 4,
            (2, 3, 1) => 1,
            (2, 4, 1) => 3,
            (2, 6, 1) => 4,
            (2, 7, 1) => 2,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => [2; 1; 0; 1;;],
            (1, 3, 0, false) => [2; 4; 0; 1;;],
            (1, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 0, false) => [1; 2; 0; 1;;],
            (2, 2, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 0, false) => [1; 3; 0; 1;;],
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1 2; 2 1; 0 0; 1 1],
            (1, 3, 0, true) => [1 2; 3 4; 0 0; 1 1],
            (1, 4, 0, true) => [1; 4; 0; 1;;],
            (2, 1, 0, true) => [1 2; 2 1; 0 0; 1 1],
            (2, 2, 0, true) => [2; 2; 0; 1;;],
            (2, 3, 0, true) => [2; 3; 0; 1;;],
            (2, 4, 0, true) => [1 2; 3 4; 0 0; 1 1],
            # Edge neighbours
            (1, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 1, false) => [2; 1; 0; 1;;],
            (1, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 1, false) => [1; 2; 0; 1;;],
            (2, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 1, true) => [1; 1; 0; 1;;],
            (1, 2, 1, true) => [1 2; 2 1; 0 0; 1 1],
            (1, 3, 1, true) => [1; 3; 0; 1;;],
            (1, 4, 1, true) => [1; 4; 0; 1;;],
            (2, 1, 1, true) => [1 2; 2 1; 0 0; 1 1],
            (2, 2, 1, true) => [2; 2; 0; 1;;],
            (2, 3, 1, true) => [2; 3; 0; 1;;],
            (2, 4, 1, true) => [2; 4; 0; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (1, 1), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (0, 1), (0, 4), (0, 5), (0, 6)
        ]
        @test interfaces == [(1, 3), (0, 2), (0, 3)]
        @test Topology.get_interfaces_on_boundary(topology) == [(0, 2), (0, 3)]
    end

    # Same mesh as the previous testset, but the second patch is rotated by 90 degrees
    # clockwise. As a result, this is the first test with a negative id in the incidence
    # relations.
    @testset "two patches: right patch 90° cw rotated" begin
        # 4 -- 3 -- 6
        # |    |    |
        # 1 -- 2 -- 5
        # Origin patch 1 is vertex 1, origin patch 2 is vertex 3
        topology = Topology.MeshTopology(((1, 2, 3, 4), (3, 2, 5, 6)), Topology.QUAD)

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (6, 7, 2)
        @test @expected Dict(1 => 6, 2 => 7, 3 => 2) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (4, 4, 1)
        expected_local_size_dim = Dict(1 => 4, 2 => 4, 3 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 3

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [[1, 2], [1, 3, 4], [3, 5, 6], [2, 5], [4, 7], [6, 7]],
            (1, 3) => [[1], [1, 2], [1, 2], [1], [2], [2]],
            (2, 1) => [[1, 2], [1, 4], [2, 3], [2, 5], [4, 3], [3, 6], [6, 5]],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [[1], [1], [1, 2], [2], [1], [2], [2]],
            (3, 1) => [[1, 2, 3, 4], [3, 2, 5, 6]],
            (3, 2) => [[2, 3, 1, 5], [6, 4, -3, 7]],
            (3, 3) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            (1, 0, 1, 1) => 1, # topology[1, 2]
            (1, 0, 2, 1) => 2,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 3,
            (2, 0, 3, 1) => 4,
            (3, 0, 1, 1) => 3,
            (3, 0, 2, 1) => 5,
            (3, 0, 3, 1) => 6,
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 5,
            (5, 0, 1, 1) => 4,
            (5, 0, 2, 1) => 7,
            (6, 0, 1, 1) => 6,
            (6, 0, 2, 1) => 7,
            (1, 0, 1, 2) => 1, # topology[1, 3]
            (2, 0, 1, 2) => 1,
            (2, 0, 2, 2) => 2,
            (3, 0, 1, 2) => 1,
            (3, 0, 2, 2) => 2,
            (4, 0, 1, 2) => 1,
            (5, 0, 1, 2) => 2,
            (6, 0, 1, 2) => 2,
            (1, 1, 1, 0) => 1, # topology[2, 1]
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 1,
            (2, 1, 2, 0) => 4,
            (3, 1, 1, 0) => 2,
            (3, 1, 2, 0) => 3,
            (4, 1, 1, 0) => 2,
            (4, 1, 2, 0) => 5,
            (5, 1, 1, 0) => 4,
            (5, 1, 2, 0) => 3,
            (6, 1, 1, 0) => 3,
            (6, 1, 2, 0) => 6,
            (7, 1, 1, 0) => 6,
            (7, 1, 2, 0) => 5,
            (1, 1, 1, 2) => 1, # topology[2, 3]
            (2, 1, 1, 2) => 1,
            (3, 1, 1, 2) => 1,
            (3, 1, 2, 2) => 2,
            (4, 1, 1, 2) => 2,
            (5, 1, 1, 2) => 1,
            (6, 1, 1, 2) => 2,
            (7, 1, 1, 2) => 2,
            (1, 2, 1, 0) => 1, # topology[3, 1]
            (1, 2, 2, 0) => 2,
            (1, 2, 3, 0) => 3,
            (1, 2, 4, 0) => 4,
            (2, 2, 1, 0) => 3,
            (2, 2, 2, 0) => 2,
            (2, 2, 3, 0) => 5,
            (2, 2, 4, 0) => 6,
            (1, 2, 1, 1) => 2, # topology[3, 2]
            (1, 2, 2, 1) => 3,
            (1, 2, 3, 1) => 1,
            (1, 2, 4, 1) => 5,
            (2, 2, 1, 1) => 6,
            (2, 2, 2, 1) => 4,
            (2, 2, 3, 1) => -3,
            (2, 2, 4, 1) => 7,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)

        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (2, 1, 0) => 3,
            (2, 2, 0) => 2,
            (2, 3, 0) => 5,
            (2, 4, 0) => 6,
            (1, 1, 1) => 2,
            (1, 2, 1) => 3,
            (1, 3, 1) => 1,
            (1, 4, 1) => 5,
            (2, 1, 1) => 6,
            (2, 2, 1) => 4,
            (2, 3, 1) => -3,
            (2, 4, 1) => 7,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (2, 2, 0) => 2,
            (2, 5, 0) => 3,
            (2, 6, 0) => 4,
            (2, 3, 0) => 1,
            (1, 1, 1) => 3,
            (1, 2, 1) => 1,
            (1, 3, 1) => 2,
            (1, 5, 1) => 4,
            (2, 3, 1) => -3,
            (2, -3, 1) => -3, # TODO: Do we allow negative global ids????
            (2, 4, 1) => 2,
            (2, 6, 1) => 1,
            (2, 7, 1) => 4,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => [2; 2; 0; 1;;],
            (1, 3, 0, false) => [2; 1; 0; 1;;],
            (1, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 0, false) => [1; 3; 0; 1;;],
            (2, 2, 0, false) => [1; 2; 0; 1;;],
            (2, 3, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1 2; 2 2; 0 0; 1 1],
            (1, 3, 0, true) => [1 2; 3 1; 0 0; 1 1],
            (1, 4, 0, true) => [1; 4; 0; 1;;],
            (2, 1, 0, true) => [1 2; 3 1; 0 0; 1 1],
            (2, 2, 0, true) => [1 2; 2 2; 0 0; 1 1],
            (2, 3, 0, true) => [2; 3; 0; 1;;],
            (2, 4, 0, true) => [2; 4; 0; 1;;],
            # Edge neighbours
            (1, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 1, false) => [2; 3; 0; -1;;],
            (1, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 1, false) => [1; 2; 0; -1;;],
            (2, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 1, true) => [1; 1; 0; 1;;],
            (1, 2, 1, true) => [1 2; 2 3; 0 0; 1 -1],
            (1, 3, 1, true) => [1; 3; 0; 1;;],
            (1, 4, 1, true) => [1; 4; 0; 1;;],
            (2, 1, 1, true) => [2; 1; 0; 1;;],
            (2, 2, 1, true) => [2; 2; 0; 1;;],
            (2, 3, 1, true) => [1 2; 2 3; 0 0; 1 -1],
            (2, 4, 1, true) => [2; 4; 0; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (1, 1), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (0, 1), (0, 4), (0, 5), (0, 6)
        ]
        @test interfaces == [(1, 3), (0, 2), (0, 3)]
        @test Topology.get_interfaces_on_boundary(topology) == [(0, 2), (0, 3)]
    end

    @testset "three patches: L-shape with common origin" begin
        # 5 -- 4 -- 3
        # |    |    |
        # 6 -- 1 -- 2
        # |    |
        # 7 -- 8
        # Origin of all patches is vertex 1
        topology = Topology.MeshTopology(
            ((1, 2, 3, 4), (1, 4, 5, 6), (1, 6, 7, 8)), Topology.QUAD
        )

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 3
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (8, 10, 3)
        @test @expected Dict(1 => 8, 2 => 10, 3 => 3) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (4, 4, 1)
        expected_local_size_dim = Dict(1 => 4, 2 => 4, 3 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 3

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2, 3, 4],
                [1, 5],
                [5, 6],
                [2, 6, 7],
                [7, 8],
                [3, 8, 9],
                [9, 10],
                [4, 10],
            ],
            (1, 3) => [[1, 2, 3], [1], [1], [1, 2], [2], [2, 3], [3], [3]],
            (2, 1) => [
                [1, 2],
                [1, 4],
                [1, 6],
                [1, 8],
                [2, 3],
                [4, 3],
                [4, 5],
                [6, 5],
                [6, 7],
                [8, 7],
            ],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [[1], [1, 2], [2, 3], [3], [1], [1], [2], [2], [3], [3]],
            (3, 1) => [[1, 2, 3, 4], [1, 4, 5, 6], [1, 6, 7, 8]],
            (3, 2) => [[2, 5, 1, 6], [3, 7, 2, 8], [4, 9, 3, 10]],
            (3, 3) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]
        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            # topology[1, 2]
            (1, 0, 1, 1) => 1,
            (1, 0, 2, 1) => 2,
            (1, 0, 3, 1) => 3,
            (1, 0, 4, 1) => 4,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 5,
            (3, 0, 1, 1) => 5,
            (3, 0, 2, 1) => 6,
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 6,
            (4, 0, 3, 1) => 7,
            (5, 0, 1, 1) => 7,
            (5, 0, 2, 1) => 8,
            (6, 0, 1, 1) => 3,
            (6, 0, 2, 1) => 8,
            (6, 0, 3, 1) => 9,
            (7, 0, 1, 1) => 9,
            (7, 0, 2, 1) => 10,
            (8, 0, 1, 1) => 4,
            (8, 0, 2, 1) => 10,
            # topology[1, 3]
            (1, 0, 1, 2) => 1,
            (1, 0, 2, 2) => 2,
            (1, 0, 3, 2) => 3,
            (2, 0, 1, 2) => 1,
            (3, 0, 1, 2) => 1,
            (4, 0, 1, 2) => 1,
            (4, 0, 2, 2) => 2,
            (5, 0, 1, 2) => 2,
            (6, 0, 1, 2) => 2,
            (6, 0, 2, 2) => 3,
            (7, 0, 1, 2) => 3,
            (8, 0, 1, 2) => 3,
            # topology[2, 1]
            (1, 1, 1, 0) => 1,
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 1,
            (2, 1, 2, 0) => 4,
            (3, 1, 1, 0) => 1,
            (3, 1, 2, 0) => 6,
            (4, 1, 1, 0) => 1,
            (4, 1, 2, 0) => 8,
            (5, 1, 1, 0) => 2,
            (5, 1, 2, 0) => 3,
            (6, 1, 1, 0) => 4,
            (6, 1, 2, 0) => 3,
            (7, 1, 1, 0) => 4,
            (7, 1, 2, 0) => 5,
            (8, 1, 1, 0) => 6,
            (8, 1, 2, 0) => 5,
            (9, 1, 1, 0) => 6,
            (9, 1, 2, 0) => 7,
            (10, 1, 1, 0) => 8,
            (10, 1, 2, 0) => 7,
            # topology[2, 3]
            (1, 1, 1, 2) => 1,
            (2, 1, 1, 2) => 1,
            (2, 1, 2, 2) => 2,
            (3, 1, 1, 2) => 2,
            (3, 1, 2, 2) => 3,
            (4, 1, 1, 2) => 3,
            (5, 1, 1, 2) => 1,
            (6, 1, 1, 2) => 1,
            (7, 1, 1, 2) => 2,
            (8, 1, 1, 2) => 2,
            (9, 1, 1, 2) => 3,
            (10, 1, 1, 2) => 3,
            # topology[3, 1]
            (1, 2, 1, 0) => 1,
            (1, 2, 2, 0) => 2,
            (1, 2, 3, 0) => 3,
            (1, 2, 4, 0) => 4,
            (2, 2, 1, 0) => 1,
            (2, 2, 2, 0) => 4,
            (2, 2, 3, 0) => 5,
            (2, 2, 4, 0) => 6,
            (3, 2, 1, 0) => 1,
            (3, 2, 2, 0) => 6,
            (3, 2, 3, 0) => 7,
            (3, 2, 4, 0) => 8,
            # topology[3, 2]
            (1, 2, 1, 1) => 2,
            (1, 2, 2, 1) => 5,
            (1, 2, 3, 1) => 1,
            (1, 2, 4, 1) => 6,
            (2, 2, 1, 1) => 3,
            (2, 2, 2, 1) => 7,
            (2, 2, 3, 1) => 2,
            (2, 2, 4, 1) => 8,
            (3, 2, 1, 1) => 4,
            (3, 2, 2, 1) => 9,
            (3, 2, 3, 1) => 3,
            (3, 2, 4, 1) => 10,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (2, 1, 0) => 1,
            (2, 2, 0) => 4,
            (2, 3, 0) => 5,
            (2, 4, 0) => 6,
            (3, 1, 0) => 1,
            (3, 2, 0) => 6,
            (3, 3, 0) => 7,
            (3, 4, 0) => 8,
            (1, 1, 1) => 2,
            (1, 2, 1) => 5,
            (1, 3, 1) => 1,
            (1, 4, 1) => 6,
            (2, 1, 1) => 3,
            (2, 2, 1) => 7,
            (2, 3, 1) => 2,
            (2, 4, 1) => 8,
            (3, 1, 1) => 4,
            (3, 2, 1) => 9,
            (3, 3, 1) => 3,
            (3, 4, 1) => 10,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)
        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (2, 1, 0) => 1,
            (2, 4, 0) => 2,
            (2, 5, 0) => 3,
            (2, 6, 0) => 4,
            (3, 1, 0) => 1,
            (3, 6, 0) => 2,
            (3, 7, 0) => 3,
            (3, 8, 0) => 4,
            (1, 2, 1) => 1,
            (1, 5, 1) => 2,
            (1, 1, 1) => 3,
            (1, 6, 1) => 4,
            (2, 3, 1) => 1,
            (2, 7, 1) => 2,
            (2, 2, 1) => 3,
            (2, 8, 1) => 4,
            (3, 4, 1) => 1,
            (3, 9, 1) => 2,
            (3, 3, 1) => 3,
            (3, 10, 1) => 4,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)
        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => [2 3; 1 1; 0 0; 1 1],
            (1, 2, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 0, false) => [2; 2; 0; 1;;],
            (2, 1, 0, false) => [1 3; 1 1; 0 0; 1 1],
            (2, 2, 0, false) => [1; 4; 0; 1;;],
            (2, 3, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 0, false) => [3; 2; 0; 1;;],
            (3, 1, 0, false) => [1 2; 1 1; 0 0; 1 1],
            (3, 2, 0, false) => [2; 4; 0; 1;;],
            (3, 3, 0, false) => Matrix{Int}(undef, 4, 0),
            (3, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1 2 3; 1 1 1; 0 0 0; 1 1 1],
            (1, 2, 0, true) => [1; 2; 0; 1;;],
            (1, 3, 0, true) => [1; 3; 0; 1;;],
            (1, 4, 0, true) => [1 2; 4 2; 0 0; 1 1],
            (2, 1, 0, true) => [1 2 3; 1 1 1; 0 0 0; 1 1 1],
            (2, 2, 0, true) => [1 2; 4 2; 0 0; 1 1],
            (2, 3, 0, true) => [2; 3; 0; 1;;],
            (2, 4, 0, true) => [2 3; 4 2; 0 0; 1 1],
            (3, 1, 0, true) => [1 2 3; 1 1 1; 0 0 0; 1 1 1],
            (3, 2, 0, true) => [2 3; 4 2; 0 0; 1 1],
            (3, 3, 0, true) => [3; 3; 0; 1;;],
            (3, 4, 0, true) => [3; 4; 0; 1;;],
            # Edge neighbours
            (1, 1, 1, false) => [2; 3; 0; 1;;],
            (1, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 1, false) => [3; 3; 0; 1;;],
            (2, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 1, false) => [1; 1; 0; 1;;],
            (2, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (3, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (3, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (3, 3, 1, false) => [2; 1; 0; 1;;],
            (3, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 1, true) => [1 2; 1 3; 0 0; 1 1],
            (1, 2, 1, true) => [1; 2; 0; 1;;],
            (1, 3, 1, true) => [1; 3; 0; 1;;],
            (1, 4, 1, true) => [1; 4; 0; 1;;],
            (2, 1, 1, true) => [2 3; 1 3; 0 0; 1 1],
            (2, 2, 1, true) => [2; 2; 0; 1;;],
            (2, 3, 1, true) => [1 2; 1 3; 0 0; 1 1],
            (2, 4, 1, true) => [2; 4; 0; 1;;],
            (3, 1, 1, true) => [3; 1; 0; 1;;],
            (3, 2, 1, true) => [3; 2; 0; 1;;],
            (3, 3, 1, true) => [2 3; 1 3; 0 0; 1 1],
            (3, 4, 1, true) => [3; 4; 0; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (1, 1),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (0, 2),
            (0, 3),
            (0, 5),
            (0, 7),
            (0, 8),
        ]
        @test interfaces == [(1, 2), (1, 3), (0, 1), (0, 4), (0, 6)]
        @test Topology.get_interfaces_on_boundary(topology) == [(0, 1), (0, 4), (0, 6)]
    end

    @testset "8 patches: square with hole" begin
        topology = Topology.MeshTopology(
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

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 8
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (16, 24, 8)
        @test @expected Dict(1 => 16, 2 => 24, 3 => 8) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (4, 4, 1)
        expected_local_size_dim = Dict(1 => 4, 2 => 4, 3 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 3

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2],
                [1, 3, 4],
                [3, 5, 6, 7],
                [2, 5, 8],
                [4, 9, 10],
                [6, 9, 11, 12],
                [10, 13],
                [11, 13, 14],
                [14, 15, 16],
                [12, 15, 17, 18],
                [16, 19],
                [17, 19, 20],
                [7, 18, 21, 22],
                [20, 21, 23],
                [8, 22, 24],
                [23, 24],
            ],
            (1, 3) => [
                [1],
                [1, 2],
                [1, 2, 8],
                [1, 8],
                [2, 3],
                [2, 3, 4],
                [3],
                [3, 4],
                [4, 5],
                [4, 5, 6],
                [5],
                [5, 6],
                [6, 7, 8],
                [6, 7],
                [7, 8],
                [7],
            ],
            (2, 1) => [
                [1, 2],
                [1, 4],
                [2, 3],
                [2, 5],
                [4, 3],
                [3, 6],
                [3, 13],
                [4, 15],
                [5, 6],
                [5, 7],
                [6, 8],
                [6, 10],
                [7, 8],
                [8, 9],
                [10, 9],
                [9, 11],
                [10, 12],
                [13, 10],
                [12, 11],
                [14, 12],
                [13, 14],
                [15, 13],
                [16, 14],
                [15, 16],
            ],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [
                [1],
                [1],
                [1, 2],
                [2],
                [1, 8],
                [2],
                [8],
                [8],
                [2, 3],
                [3],
                [3, 4],
                [4],
                [3],
                [4],
                [4, 5],
                [5],
                [5, 6],
                [6],
                [5],
                [6],
                [6, 7],
                [7, 8],
                [7],
                [7],
            ],
            (3, 1) => [
                [1, 2, 3, 4],
                [2, 5, 6, 3],
                [5, 7, 8, 6],
                [6, 8, 9, 10],
                [10, 9, 11, 12],
                [13, 10, 12, 14],
                [15, 13, 14, 16],
                [4, 3, 13, 15],
            ],
            (3, 2) => [
                [2, 3, 1, 5],
                [3, 9, 4, 6],
                [9, 13, 10, 11],
                [12, 14, 11, 15],
                [17, 16, 15, 19],
                [21, 17, 18, 20],
                [24, 21, 22, 23],
                [8, 7, 5, 22],
            ],
            (3, 3) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]
        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            # topology[1, 2]
            (1, 0, 1, 1) => 1,
            (1, 0, 2, 1) => 2,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 3,
            (2, 0, 3, 1) => 4,
            (3, 0, 1, 1) => 3,
            (3, 0, 2, 1) => 5,
            (3, 0, 3, 1) => 6,
            (3, 0, 4, 1) => 7,
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 5,
            (4, 0, 3, 1) => 8,
            (5, 0, 1, 1) => 4,
            (5, 0, 2, 1) => 9,
            (5, 0, 3, 1) => 10,
            (6, 0, 1, 1) => 6,
            (6, 0, 2, 1) => 9,
            (6, 0, 3, 1) => 11,
            (6, 0, 4, 1) => 12,
            (7, 0, 1, 1) => 10,
            (7, 0, 2, 1) => 13,
            (8, 0, 1, 1) => 11,
            (8, 0, 2, 1) => 13,
            (8, 0, 3, 1) => 14,
            (9, 0, 1, 1) => 14,
            (9, 0, 2, 1) => 15,
            (9, 0, 3, 1) => 16,
            (10, 0, 1, 1) => 12,
            (10, 0, 2, 1) => 15,
            (10, 0, 3, 1) => 17,
            (10, 0, 4, 1) => 18,
            (11, 0, 1, 1) => 16,
            (11, 0, 2, 1) => 19,
            (12, 0, 1, 1) => 17,
            (12, 0, 2, 1) => 19,
            (12, 0, 3, 1) => 20,
            (13, 0, 1, 1) => 7,
            (13, 0, 2, 1) => 18,
            (13, 0, 3, 1) => 21,
            (13, 0, 4, 1) => 22,
            (14, 0, 1, 1) => 20,
            (14, 0, 2, 1) => 21,
            (14, 0, 3, 1) => 23,
            (15, 0, 1, 1) => 8,
            (15, 0, 2, 1) => 22,
            (15, 0, 3, 1) => 24,
            (16, 0, 1, 1) => 23,
            (16, 0, 2, 1) => 24,
            # topology[1, 3]
            (1, 0, 1, 2) => 1,
            (2, 0, 1, 2) => 1,
            (2, 0, 2, 2) => 2,
            (3, 0, 1, 2) => 1,
            (3, 0, 2, 2) => 2,
            (3, 0, 3, 2) => 8,
            (4, 0, 1, 2) => 1,
            (4, 0, 2, 2) => 8,
            (5, 0, 1, 2) => 2,
            (5, 0, 2, 2) => 3,
            (6, 0, 1, 2) => 2,
            (6, 0, 2, 2) => 3,
            (6, 0, 3, 2) => 4,
            (7, 0, 1, 2) => 3,
            (8, 0, 1, 2) => 3,
            (8, 0, 2, 2) => 4,
            (9, 0, 1, 2) => 4,
            (9, 0, 2, 2) => 5,
            (10, 0, 1, 2) => 4,
            (10, 0, 2, 2) => 5,
            (10, 0, 3, 2) => 6,
            (11, 0, 1, 2) => 5,
            (12, 0, 1, 2) => 5,
            (12, 0, 2, 2) => 6,
            (13, 0, 1, 2) => 6,
            (13, 0, 2, 2) => 7,
            (13, 0, 3, 2) => 8,
            (14, 0, 1, 2) => 6,
            (14, 0, 2, 2) => 7,
            (15, 0, 1, 2) => 7,
            (15, 0, 2, 2) => 8,
            (16, 0, 1, 2) => 7,
            # topology[2, 1]
            (1, 1, 1, 0) => 1,
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 1,
            (2, 1, 2, 0) => 4,
            (3, 1, 1, 0) => 2,
            (3, 1, 2, 0) => 3,
            (4, 1, 1, 0) => 2,
            (4, 1, 2, 0) => 5,
            (5, 1, 1, 0) => 4,
            (5, 1, 2, 0) => 3,
            (6, 1, 1, 0) => 3,
            (6, 1, 2, 0) => 6,
            (7, 1, 1, 0) => 3,
            (7, 1, 2, 0) => 13,
            (8, 1, 1, 0) => 4,
            (8, 1, 2, 0) => 15,
            (9, 1, 1, 0) => 5,
            (9, 1, 2, 0) => 6,
            (10, 1, 1, 0) => 5,
            (10, 1, 2, 0) => 7,
            (11, 1, 1, 0) => 6,
            (11, 1, 2, 0) => 8,
            (12, 1, 1, 0) => 6,
            (12, 1, 2, 0) => 10,
            (13, 1, 1, 0) => 7,
            (13, 1, 2, 0) => 8,
            (14, 1, 1, 0) => 8,
            (14, 1, 2, 0) => 9,
            (15, 1, 1, 0) => 10,
            (15, 1, 2, 0) => 9,
            (16, 1, 1, 0) => 9,
            (16, 1, 2, 0) => 11,
            (17, 1, 1, 0) => 10,
            (17, 1, 2, 0) => 12,
            (18, 1, 1, 0) => 13,
            (18, 1, 2, 0) => 10,
            (19, 1, 1, 0) => 12,
            (19, 1, 2, 0) => 11,
            (20, 1, 1, 0) => 14,
            (20, 1, 2, 0) => 12,
            (21, 1, 1, 0) => 13,
            (21, 1, 2, 0) => 14,
            (22, 1, 1, 0) => 15,
            (22, 1, 2, 0) => 13,
            (23, 1, 1, 0) => 16,
            (23, 1, 2, 0) => 14,
            (24, 1, 1, 0) => 15,
            (24, 1, 2, 0) => 16,
            # topology[2, 3]
            (1, 1, 1, 2) => 1,
            (2, 1, 1, 2) => 1,
            (3, 1, 1, 2) => 1,
            (3, 1, 2, 2) => 2,
            (4, 1, 1, 2) => 2,
            (5, 1, 1, 2) => 1,
            (5, 1, 2, 2) => 8,
            (6, 1, 1, 2) => 2,
            (7, 1, 1, 2) => 8,
            (8, 1, 1, 2) => 8,
            (9, 1, 1, 2) => 2,
            (9, 1, 2, 2) => 3,
            (10, 1, 1, 2) => 3,
            (11, 1, 1, 2) => 3,
            (11, 1, 2, 2) => 4,
            (12, 1, 1, 2) => 4,
            (13, 1, 1, 2) => 3,
            (14, 1, 1, 2) => 4,
            (15, 1, 1, 2) => 4,
            (15, 1, 2, 2) => 5,
            (16, 1, 1, 2) => 5,
            (17, 1, 1, 2) => 5,
            (17, 1, 2, 2) => 6,
            (18, 1, 1, 2) => 6,
            (19, 1, 1, 2) => 5,
            (20, 1, 1, 2) => 6,
            (21, 1, 1, 2) => 6,
            (21, 1, 2, 2) => 7,
            (22, 1, 1, 2) => 7,
            (22, 1, 2, 2) => 8,
            (23, 1, 1, 2) => 7,
            (24, 1, 1, 2) => 7,
            # topology[3, 1]
            (1, 2, 1, 0) => 1,
            (1, 2, 2, 0) => 2,
            (1, 2, 3, 0) => 3,
            (1, 2, 4, 0) => 4,
            (2, 2, 1, 0) => 2,
            (2, 2, 2, 0) => 5,
            (2, 2, 3, 0) => 6,
            (2, 2, 4, 0) => 3,
            (3, 2, 1, 0) => 5,
            (3, 2, 2, 0) => 7,
            (3, 2, 3, 0) => 8,
            (3, 2, 4, 0) => 6,
            (4, 2, 1, 0) => 6,
            (4, 2, 2, 0) => 8,
            (4, 2, 3, 0) => 9,
            (4, 2, 4, 0) => 10,
            (5, 2, 1, 0) => 10,
            (5, 2, 2, 0) => 9,
            (5, 2, 3, 0) => 11,
            (5, 2, 4, 0) => 12,
            (6, 2, 1, 0) => 13,
            (6, 2, 2, 0) => 10,
            (6, 2, 3, 0) => 12,
            (6, 2, 4, 0) => 14,
            (7, 2, 1, 0) => 15,
            (7, 2, 2, 0) => 13,
            (7, 2, 3, 0) => 14,
            (7, 2, 4, 0) => 16,
            (8, 2, 1, 0) => 4,
            (8, 2, 2, 0) => 3,
            (8, 2, 3, 0) => 13,
            (8, 2, 4, 0) => 15,
            # topology[3, 2]
            (1, 2, 1, 1) => 2,
            (1, 2, 2, 1) => 3,
            (1, 2, 3, 1) => 1,
            (1, 2, 4, 1) => 5,
            (2, 2, 1, 1) => 3,
            (2, 2, 2, 1) => 9,
            (2, 2, 3, 1) => 4,
            (2, 2, 4, 1) => 6,
            (3, 2, 1, 1) => 9,
            (3, 2, 2, 1) => 13,
            (3, 2, 3, 1) => 10,
            (3, 2, 4, 1) => 11,
            (4, 2, 1, 1) => 12,
            (4, 2, 2, 1) => 14,
            (4, 2, 3, 1) => 11,
            (4, 2, 4, 1) => 15,
            (5, 2, 1, 1) => 17,
            (5, 2, 2, 1) => 16,
            (5, 2, 3, 1) => 15,
            (5, 2, 4, 1) => 19,
            (6, 2, 1, 1) => 21,
            (6, 2, 2, 1) => 17,
            (6, 2, 3, 1) => 18,
            (6, 2, 4, 1) => 20,
            (7, 2, 1, 1) => 24,
            (7, 2, 2, 1) => 21,
            (7, 2, 3, 1) => 22,
            (7, 2, 4, 1) => 23,
            (8, 2, 1, 1) => 8,
            (8, 2, 2, 1) => 7,
            (8, 2, 3, 1) => 5,
            (8, 2, 4, 1) => 22,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (2, 1, 0) => 2,
            (2, 2, 0) => 5,
            (2, 3, 0) => 6,
            (2, 4, 0) => 3,
            (3, 1, 0) => 5,
            (3, 2, 0) => 7,
            (3, 3, 0) => 8,
            (3, 4, 0) => 6,
            (4, 1, 0) => 6,
            (4, 2, 0) => 8,
            (4, 3, 0) => 9,
            (4, 4, 0) => 10,
            (5, 1, 0) => 10,
            (5, 2, 0) => 9,
            (5, 3, 0) => 11,
            (5, 4, 0) => 12,
            (6, 1, 0) => 13,
            (6, 2, 0) => 10,
            (6, 3, 0) => 12,
            (6, 4, 0) => 14,
            (7, 1, 0) => 15,
            (7, 2, 0) => 13,
            (7, 3, 0) => 14,
            (7, 4, 0) => 16,
            (8, 1, 0) => 4,
            (8, 2, 0) => 3,
            (8, 3, 0) => 13,
            (8, 4, 0) => 15,
            (1, 1, 1) => 2,
            (1, 2, 1) => 3,
            (1, 3, 1) => 1,
            (1, 4, 1) => 5,
            (2, 1, 1) => 3,
            (2, 2, 1) => 9,
            (2, 3, 1) => 4,
            (2, 4, 1) => 6,
            (3, 1, 1) => 9,
            (3, 2, 1) => 13,
            (3, 3, 1) => 10,
            (3, 4, 1) => 11,
            (4, 1, 1) => 12,
            (4, 2, 1) => 14,
            (4, 3, 1) => 11,
            (4, 4, 1) => 15,
            (5, 1, 1) => 17,
            (5, 2, 1) => 16,
            (5, 3, 1) => 15,
            (5, 4, 1) => 19,
            (6, 1, 1) => 21,
            (6, 2, 1) => 17,
            (6, 3, 1) => 18,
            (6, 4, 1) => 20,
            (7, 1, 1) => 24,
            (7, 2, 1) => 21,
            (7, 3, 1) => 22,
            (7, 4, 1) => 23,
            (8, 1, 1) => 8,
            (8, 2, 1) => 7,
            (8, 3, 1) => 5,
            (8, 4, 1) => 22,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)
        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (2, 2, 0) => 1,
            (2, 5, 0) => 2,
            (2, 6, 0) => 3,
            (2, 3, 0) => 4,
            (3, 5, 0) => 1,
            (3, 7, 0) => 2,
            (3, 8, 0) => 3,
            (3, 6, 0) => 4,
            (4, 6, 0) => 1,
            (4, 8, 0) => 2,
            (4, 9, 0) => 3,
            (4, 10, 0) => 4,
            (5, 10, 0) => 1,
            (5, 9, 0) => 2,
            (5, 11, 0) => 3,
            (5, 12, 0) => 4,
            (6, 13, 0) => 1,
            (6, 10, 0) => 2,
            (6, 12, 0) => 3,
            (6, 14, 0) => 4,
            (7, 15, 0) => 1,
            (7, 13, 0) => 2,
            (7, 14, 0) => 3,
            (7, 16, 0) => 4,
            (8, 4, 0) => 1,
            (8, 3, 0) => 2,
            (8, 13, 0) => 3,
            (8, 15, 0) => 4,
            (1, 2, 1) => 1,
            (1, 3, 1) => 2,
            (1, 1, 1) => 3,
            (1, 5, 1) => 4,
            (2, 3, 1) => 1,
            (2, 9, 1) => 2,
            (2, 4, 1) => 3,
            (2, 6, 1) => 4,
            (3, 9, 1) => 1,
            (3, 13, 1) => 2,
            (3, 10, 1) => 3,
            (3, 11, 1) => 4,
            (4, 12, 1) => 1,
            (4, 14, 1) => 2,
            (4, 11, 1) => 3,
            (4, 15, 1) => 4,
            (5, 17, 1) => 1,
            (5, 16, 1) => 2,
            (5, 15, 1) => 3,
            (5, 19, 1) => 4,
            (6, 21, 1) => 1,
            (6, 17, 1) => 2,
            (6, 18, 1) => 3,
            (6, 20, 1) => 4,
            (7, 24, 1) => 1,
            (7, 21, 1) => 2,
            (7, 22, 1) => 3,
            (7, 23, 1) => 4,
            (8, 8, 1) => 1,
            (8, 7, 1) => 2,
            (8, 5, 1) => 3,
            (8, 22, 1) => 4,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)
        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => [2; 1; 0; 1;;],
            (1, 3, 0, false) => [2 8; 4 2; 0 0; 1 1],
            (1, 4, 0, false) => [8; 1; 0; 1;;],
            (2, 1, 0, false) => [1; 2; 0; 1;;],
            (2, 2, 0, false) => [3; 1; 0; 1;;],
            (2, 3, 0, false) => [3 4; 4 1; 0 0; 1 1],
            (2, 4, 0, false) => [1 8; 3 2; 0 0; 1 1],
            (3, 1, 0, false) => [2; 2; 0; 1;;],
            (3, 2, 0, false) => Matrix{Int}(undef, 4, 0),
            (3, 3, 0, false) => [4; 2; 0; 1;;],
            (3, 4, 0, false) => [2 4; 3 1; 0 0; 1 1],
            (4, 1, 0, false) => [2 3; 3 4; 0 0; 1 1],
            (4, 2, 0, false) => [3; 3; 0; 1;;],
            (4, 3, 0, false) => [5; 2; 0; 1;;],
            (4, 4, 0, false) => [5 6; 1 2; 0 0; 1 1],
            (5, 1, 0, false) => [4 6; 4 2; 0 0; 1 1],
            (5, 2, 0, false) => [4; 3; 0; 1;;],
            (5, 3, 0, false) => Matrix{Int}(undef, 4, 0),
            (5, 4, 0, false) => [6; 3; 0; 1;;],
            (6, 1, 0, false) => [7 8; 2 3; 0 0; 1 1],
            (6, 2, 0, false) => [4 5; 4 1; 0 0; 1 1],
            (6, 3, 0, false) => [5; 4; 0; 1;;],
            (6, 4, 0, false) => [7; 3; 0; 1;;],
            (7, 1, 0, false) => [8; 4; 0; 1;;],
            (7, 2, 0, false) => [6 8; 1 3; 0 0; 1 1],
            (7, 3, 0, false) => [6; 4; 0; 1;;],
            (7, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (8, 1, 0, false) => [1; 4; 0; 1;;],
            (8, 2, 0, false) => [1 2; 3 4; 0 0; 1 1],
            (8, 3, 0, false) => [6 7; 1 2; 0 0; 1 1],
            (8, 4, 0, false) => [7; 1; 0; 1;;],
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1 2; 2 1; 0 0; 1 1],
            (1, 3, 0, true) => [1 2 8; 3 4 2; 0 0 0; 1 1 1],
            (1, 4, 0, true) => [1 8; 4 1; 0 0; 1 1],
            (2, 1, 0, true) => [1 2; 2 1; 0 0; 1 1],
            (2, 2, 0, true) => [2 3; 2 1; 0 0; 1 1],
            (2, 3, 0, true) => [2 3 4; 3 4 1; 0 0 0; 1 1 1],
            (2, 4, 0, true) => [1 2 8; 3 4 2; 0 0 0; 1 1 1],
            (3, 1, 0, true) => [2 3; 2 1; 0 0; 1 1],
            (3, 2, 0, true) => [3; 2; 0; 1;;],
            (3, 3, 0, true) => [3 4; 3 2; 0 0; 1 1],
            (3, 4, 0, true) => [2 3 4; 3 4 1; 0 0 0; 1 1 1],
            (4, 1, 0, true) => [2 3 4; 3 4 1; 0 0 0; 1 1 1],
            (4, 2, 0, true) => [3 4; 3 2; 0 0; 1 1],
            (4, 3, 0, true) => [4 5; 3 2; 0 0; 1 1],
            (4, 4, 0, true) => [4 5 6; 4 1 2; 0 0 0; 1 1 1],
            (5, 1, 0, true) => [4 5 6; 4 1 2; 0 0 0; 1 1 1],
            (5, 2, 0, true) => [4 5; 3 2; 0 0; 1 1],
            (5, 3, 0, true) => [5; 3; 0; 1;;],
            (5, 4, 0, true) => [5 6; 4 3; 0 0; 1 1],
            (6, 1, 0, true) => [6 7 8; 1 2 3; 0 0 0; 1 1 1],
            (6, 2, 0, true) => [4 5 6; 4 1 2; 0 0 0; 1 1 1],
            (6, 3, 0, true) => [5 6; 4 3; 0 0; 1 1],
            (6, 4, 0, true) => [6 7; 4 3; 0 0; 1 1],
            (7, 1, 0, true) => [7 8; 1 4; 0 0; 1 1],
            (7, 2, 0, true) => [6 7 8; 1 2 3; 0 0 0; 1 1 1],
            (7, 3, 0, true) => [6 7; 4 3; 0 0; 1 1],
            (7, 4, 0, true) => [7; 4; 0; 1;;],
            (8, 1, 0, true) => [1 8; 4 1; 0 0; 1 1],
            (8, 2, 0, true) => [1 2 8; 3 4 2; 0 0 0; 1 1 1],
            (8, 3, 0, true) => [6 7 8; 1 2 3; 0 0 0; 1 1 1],
            (8, 4, 0, true) => [7 8; 1 4; 0 0; 1 1],
            # Edge neighbours
            (1, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 1, false) => [2; 1; 0; 1;;],
            (1, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 1, false) => [8; 3; 0; 1;;],
            (2, 1, 1, false) => [1; 2; 0; 1;;],
            (2, 2, 1, false) => [3; 1; 0; 1;;],
            (2, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (3, 1, 1, false) => [2; 2; 0; 1;;],
            (3, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (3, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (3, 4, 1, false) => [4; 3; 0; 1;;],
            (4, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (4, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (4, 3, 1, false) => [3; 4; 0; 1;;],
            (4, 4, 1, false) => [5; 3; 0; 1;;],
            (5, 1, 1, false) => [6; 2; 0; 1;;],
            (5, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (5, 3, 1, false) => [4; 4; 0; 1;;],
            (5, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (6, 1, 1, false) => [7; 2; 0; 1;;],
            (6, 2, 1, false) => [5; 1; 0; 1;;],
            (6, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (6, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (7, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (7, 2, 1, false) => [6; 1; 0; 1;;],
            (7, 3, 1, false) => [8; 4; 0; 1;;],
            (7, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (8, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (8, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (8, 3, 1, false) => [1; 4; 0; 1;;],
            (8, 4, 1, false) => [7; 3; 0; 1;;],
            (1, 1, 1, true) => [1; 1; 0; 1;;],
            (1, 2, 1, true) => [1 2; 2 1; 0 0; 1 1],
            (1, 3, 1, true) => [1; 3; 0; 1;;],
            (1, 4, 1, true) => [1 8; 4 3; 0 0; 1 1],
            (2, 1, 1, true) => [1 2; 2 1; 0 0; 1 1],
            (2, 2, 1, true) => [2 3; 2 1; 0 0; 1 1],
            (2, 3, 1, true) => [2; 3; 0; 1;;],
            (2, 4, 1, true) => [2; 4; 0; 1;;],
            (3, 1, 1, true) => [2 3; 2 1; 0 0; 1 1],
            (3, 2, 1, true) => [3; 2; 0; 1;;],
            (3, 3, 1, true) => [3; 3; 0; 1;;],
            (3, 4, 1, true) => [3 4; 4 3; 0 0; 1 1],
            (4, 1, 1, true) => [4; 1; 0; 1;;],
            (4, 2, 1, true) => [4; 2; 0; 1;;],
            (4, 3, 1, true) => [3 4; 4 3; 0 0; 1 1],
            (4, 4, 1, true) => [4 5; 4 3; 0 0; 1 1],
            (5, 1, 1, true) => [5 6; 1 2; 0 0; 1 1],
            (5, 2, 1, true) => [5; 2; 0; 1;;],
            (5, 3, 1, true) => [4 5; 4 3; 0 0; 1 1],
            (5, 4, 1, true) => [5; 4; 0; 1;;],
            (6, 1, 1, true) => [6 7; 1 2; 0 0; 1 1],
            (6, 2, 1, true) => [5 6; 1 2; 0 0; 1 1],
            (6, 3, 1, true) => [6; 3; 0; 1;;],
            (6, 4, 1, true) => [6; 4; 0; 1;;],
            (7, 1, 1, true) => [7; 1; 0; 1;;],
            (7, 2, 1, true) => [6 7; 1 2; 0 0; 1 1],
            (7, 3, 1, true) => [7 8; 3 4; 0 0; 1 1],
            (7, 4, 1, true) => [7; 4; 0; 1;;],
            (8, 1, 1, true) => [8; 1; 0; 1;;],
            (8, 2, 1, true) => [8; 2; 0; 1;;],
            (8, 3, 1, true) => [1 8; 4 3; 0 0; 1 1],
            (8, 4, 1, true) => [7 8; 3 4; 0 0; 1 1],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        # Remember that the edges next to the hole are also boundaries.
        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 10),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 16),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 23),
            (1, 24),
            (0, 1),
            (0, 7),
            (0, 11),
            (0, 16),
        ]
        @test interfaces == [
            (1, 3),
            (1, 5),
            (1, 9),
            (1, 11),
            (1, 15),
            (1, 17),
            (1, 21),
            (1, 22),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
        ]
        @test Topology.get_interfaces_on_boundary(topology) == [
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
        ]
    end
end

############################################################################################
#                                            3D                                            #
############################################################################################
@testset "3D" verbose = true begin
    @testset "single patch" verbose=true begin
        topology = Topology.MeshTopology(((1, 2, 3, 4, 5, 6, 7, 8),), Topology.HEX)

        @test Topology.get_manifold_dim(topology) == 3
        @test Topology.get_incidence_relations_dim(topology) == 4
        @test Topology.get_num_patches(topology) == 1
        @test Topology.get_patch_type(topology) == typeof(Topology.HEX)
        @test Topology.get_topological_patch(topology) == Topology.HEX
        @test size(topology) == (8, 12, 6, 1)
        @test @expected Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (8, 12, 6, 1)
        expected_local_size_dim = Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 4

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2, 3],
                [1, 4, 5],
                [4, 6, 7],
                [2, 6, 8],
                [3, 9, 10],
                [5, 9, 11],
                [7, 11, 12],
                [8, 10, 12],
            ],
            (1, 3) => [
                [1, 2, 3],
                [1, 2, 4],
                [1, 4, 5],
                [1, 3, 5],
                [2, 3, 6],
                [2, 4, 6],
                [4, 5, 6],
                [3, 5, 6],
            ],
            (1, 4) => [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            (2, 1) => [
                [2, 1],
                [4, 1],
                [5, 1],
                [3, 2],
                [6, 2],
                [3, 4],
                [7, 3],
                [8, 4],
                [6, 5],
                [8, 5],
                [7, 6],
                [7, 8],
            ],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [
                [1, 2],
                [1, 3],
                [2, 3],
                [1, 4],
                [2, 4],
                [1, 5],
                [4, 5],
                [3, 5],
                [2, 6],
                [3, 6],
                [4, 6],
                [5, 6],
            ],
            (2, 4) => [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            (3, 1) => [
                [3, 2, 1, 4],
                [6, 5, 1, 2],
                [8, 4, 1, 5],
                [7, 6, 2, 3],
                [7, 3, 4, 8],
                [7, 8, 5, 6],
            ],
            (3, 2) => [
                [6, 1, 4, 2],
                [5, 3, 9, 1],
                [10, 2, 8, 3],
                [7, 5, 11, 4],
                [12, 6, 7, 8],
                [11, 10, 12, 9],
            ],
            (3, 3) => Vector{Vector{Int}}(),
            (3, 4) => [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            (4, 1) => [
                [1, 2, 3, 4, 5, 6, 7, 8],
            ],
            (4, 2) => [
                [-12, -9, -1, -6, -11, -10, -2, -4, -7, -8, -3, -5],
            ],
            (4, 3) => [
                [3, 4, 2, 5, 1, 6],
            ],
            (4, 4) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            # topology[1, 2]
            (1, 0, 1, 1) => 1,
            (1, 0, 2, 1) => 2,
            (1, 0, 3, 1) => 3,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 4,
            (2, 0, 3, 1) => 5,
            (3, 0, 1, 1) => 4,
            (3, 0, 2, 1) => 6,
            (3, 0, 3, 1) => 7,
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 6,
            (4, 0, 3, 1) => 8,
            (5, 0, 1, 1) => 3,
            (5, 0, 2, 1) => 9,
            (5, 0, 3, 1) => 10,
            (6, 0, 1, 1) => 5,
            (6, 0, 2, 1) => 9,
            (6, 0, 3, 1) => 11,
            (7, 0, 1, 1) => 7,
            (7, 0, 2, 1) => 11,
            (7, 0, 3, 1) => 12,
            (8, 0, 1, 1) => 8,
            (8, 0, 2, 1) => 10,
            (8, 0, 3, 1) => 12,
            # topology[1, 3]
            (1, 0, 1, 2) => 1,
            (1, 0, 2, 2) => 2,
            (1, 0, 3, 2) => 3,
            (2, 0, 1, 2) => 1,
            (2, 0, 2, 2) => 2,
            (2, 0, 3, 2) => 4,
            (3, 0, 1, 2) => 1,
            (3, 0, 2, 2) => 4,
            (3, 0, 3, 2) => 5,
            (4, 0, 1, 2) => 1,
            (4, 0, 2, 2) => 3,
            (4, 0, 3, 2) => 5,
            (5, 0, 1, 2) => 2,
            (5, 0, 2, 2) => 3,
            (5, 0, 3, 2) => 6,
            (6, 0, 1, 2) => 2,
            (6, 0, 2, 2) => 4,
            (6, 0, 3, 2) => 6,
            (7, 0, 1, 2) => 4,
            (7, 0, 2, 2) => 5,
            (7, 0, 3, 2) => 6,
            (8, 0, 1, 2) => 3,
            (8, 0, 2, 2) => 5,
            (8, 0, 3, 2) => 6,
            # topology[1, 4]
            (1, 0, 1, 3) => 1,
            (2, 0, 1, 3) => 1,
            (3, 0, 1, 3) => 1,
            (4, 0, 1, 3) => 1,
            (5, 0, 1, 3) => 1,
            (6, 0, 1, 3) => 1,
            (7, 0, 1, 3) => 1,
            (8, 0, 1, 3) => 1,
            # topology[2, 1]
            (1, 1, 1, 0) => 2,
            (1, 1, 2, 0) => 1,
            (2, 1, 1, 0) => 4,
            (2, 1, 2, 0) => 1,
            (3, 1, 1, 0) => 5,
            (3, 1, 2, 0) => 1,
            (4, 1, 1, 0) => 3,
            (4, 1, 2, 0) => 2,
            (5, 1, 1, 0) => 6,
            (5, 1, 2, 0) => 2,
            (6, 1, 1, 0) => 3,
            (6, 1, 2, 0) => 4,
            (7, 1, 1, 0) => 7,
            (7, 1, 2, 0) => 3,
            (8, 1, 1, 0) => 8,
            (8, 1, 2, 0) => 4,
            (9, 1, 1, 0) => 6,
            (9, 1, 2, 0) => 5,
            (10, 1, 1, 0) => 8,
            (10, 1, 2, 0) => 5,
            (11, 1, 1, 0) => 7,
            (11, 1, 2, 0) => 6,
            (12, 1, 1, 0) => 7,
            (12, 1, 2, 0) => 8,
            # topology[2, 3]
            (1, 1, 1, 2) => 1,
            (1, 1, 2, 2) => 2,
            (2, 1, 1, 2) => 1,
            (2, 1, 2, 2) => 3,
            (3, 1, 1, 2) => 2,
            (3, 1, 2, 2) => 3,
            (4, 1, 1, 2) => 1,
            (4, 1, 2, 2) => 4,
            (5, 1, 1, 2) => 2,
            (5, 1, 2, 2) => 4,
            (6, 1, 1, 2) => 1,
            (6, 1, 2, 2) => 5,
            (7, 1, 1, 2) => 4,
            (7, 1, 2, 2) => 5,
            (8, 1, 1, 2) => 3,
            (8, 1, 2, 2) => 5,
            (9, 1, 1, 2) => 2,
            (9, 1, 2, 2) => 6,
            (10, 1, 1, 2) => 3,
            (10, 1, 2, 2) => 6,
            (11, 1, 1, 2) => 4,
            (11, 1, 2, 2) => 6,
            (12, 1, 1, 2) => 5,
            (12, 1, 2, 2) => 6,
            # topology[2, 4]
            (1, 1, 1, 3) => 1,
            (2, 1, 1, 3) => 1,
            (3, 1, 1, 3) => 1,
            (4, 1, 1, 3) => 1,
            (5, 1, 1, 3) => 1,
            (6, 1, 1, 3) => 1,
            (7, 1, 1, 3) => 1,
            (8, 1, 1, 3) => 1,
            (9, 1, 1, 3) => 1,
            (10, 1, 1, 3) => 1,
            (11, 1, 1, 3) => 1,
            (12, 1, 1, 3) => 1,
            # topology[3, 1]
            (1, 2, 1, 0) => 3,
            (1, 2, 2, 0) => 2,
            (1, 2, 3, 0) => 1,
            (1, 2, 4, 0) => 4,
            (2, 2, 1, 0) => 6,
            (2, 2, 2, 0) => 5,
            (2, 2, 3, 0) => 1,
            (2, 2, 4, 0) => 2,
            (3, 2, 1, 0) => 8,
            (3, 2, 2, 0) => 4,
            (3, 2, 3, 0) => 1,
            (3, 2, 4, 0) => 5,
            (4, 2, 1, 0) => 7,
            (4, 2, 2, 0) => 6,
            (4, 2, 3, 0) => 2,
            (4, 2, 4, 0) => 3,
            (5, 2, 1, 0) => 7,
            (5, 2, 2, 0) => 3,
            (5, 2, 3, 0) => 4,
            (5, 2, 4, 0) => 8,
            (6, 2, 1, 0) => 7,
            (6, 2, 2, 0) => 8,
            (6, 2, 3, 0) => 5,
            (6, 2, 4, 0) => 6,
            # topology[3, 2]
            (1, 2, 1, 1) => 6,
            (1, 2, 2, 1) => 1,
            (1, 2, 3, 1) => 4,
            (1, 2, 4, 1) => 2,
            (2, 2, 1, 1) => 5,
            (2, 2, 2, 1) => 3,
            (2, 2, 3, 1) => 9,
            (2, 2, 4, 1) => 1,
            (3, 2, 1, 1) => 10,
            (3, 2, 2, 1) => 2,
            (3, 2, 3, 1) => 8,
            (3, 2, 4, 1) => 3,
            (4, 2, 1, 1) => 7,
            (4, 2, 2, 1) => 5,
            (4, 2, 3, 1) => 11,
            (4, 2, 4, 1) => 4,
            (5, 2, 1, 1) => 12,
            (5, 2, 2, 1) => 6,
            (5, 2, 3, 1) => 7,
            (5, 2, 4, 1) => 8,
            (6, 2, 1, 1) => 11,
            (6, 2, 2, 1) => 10,
            (6, 2, 3, 1) => 12,
            (6, 2, 4, 1) => 9,
            # topology[3, 4]
            (1, 2, 1, 3) => 1,
            (2, 2, 1, 3) => 1,
            (3, 2, 1, 3) => 1,
            (4, 2, 1, 3) => 1,
            (5, 2, 1, 3) => 1,
            (6, 2, 1, 3) => 1,
            # topology[4, 1]
            (1, 3, 1, 0) => 1,
            (1, 3, 2, 0) => 2,
            (1, 3, 3, 0) => 3,
            (1, 3, 4, 0) => 4,
            (1, 3, 5, 0) => 5,
            (1, 3, 6, 0) => 6,
            (1, 3, 7, 0) => 7,
            (1, 3, 8, 0) => 8,
            # topology[4, 2]
            (1, 3, 1, 1) => -12,
            (1, 3, 2, 1) => -9,
            (1, 3, 3, 1) => -1,
            (1, 3, 4, 1) => -6,
            (1, 3, 5, 1) => -11,
            (1, 3, 6, 1) => -10,
            (1, 3, 7, 1) => -2,
            (1, 3, 8, 1) => -4,
            (1, 3, 9, 1) => -7,
            (1, 3, 10, 1) => -8,
            (1, 3, 11, 1) => -3,
            (1, 3, 12, 1) => -5,
            # topology[4, 3]
            (1, 3, 1, 2) => 3,
            (1, 3, 2, 2) => 4,
            (1, 3, 3, 2) => 2,
            (1, 3, 4, 2) => 5,
            (1, 3, 5, 2) => 1,
            (1, 3, 6, 2) => 6,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (1, 5, 0) => 5,
            (1, 6, 0) => 6,
            (1, 7, 0) => 7,
            (1, 8, 0) => 8,
            (1, 1, 1) => -12,
            (1, 2, 1) => -9,
            (1, 3, 1) => -1,
            (1, 4, 1) => -6,
            (1, 5, 1) => -11,
            (1, 6, 1) => -10,
            (1, 7, 1) => -2,
            (1, 8, 1) => -4,
            (1, 9, 1) => -7,
            (1, 10, 1) => -8,
            (1, 11, 1) => -3,
            (1, 12, 1) => -5,
            (1, 1, 2) => 3,
            (1, 2, 2) => 4,
            (1, 3, 2) => 2,
            (1, 4, 2) => 5,
            (1, 5, 2) => 1,
            (1, 6, 2) => 6,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (1, 5, 0) => 5,
            (1, 6, 0) => 6,
            (1, 7, 0) => 7,
            (1, 8, 0) => 8,
            (1, 12, 1) => -1,
            (1, 9, 1) => -2,
            (1, 1, 1) => -3,
            (1, 6, 1) => -4,
            (1, 11, 1) => -5,
            (1, 10, 1) => -6,
            (1, 2, 1) => -7,
            (1, 4, 1) => -8,
            (1, 7, 1) => -9,
            (1, 8, 1) => -10,
            (1, 3, 1) => -11,
            (1, 5, 1) => -12,
            (1, 3, 2) => 1,
            (1, 4, 2) => 2,
            (1, 2, 2) => 3,
            (1, 5, 2) => 4,
            (1, 1, 2) => 5,
            (1, 6, 2) => 6,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 7, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 8, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1; 2; 0; 1;;],
            (1, 3, 0, true) => [1; 3; 0; 1;;],
            (1, 4, 0, true) => [1; 4; 0; 1;;],
            (1, 5, 0, true) => [1; 5; 0; 1;;],
            (1, 6, 0, true) => [1; 6; 0; 1;;],
            (1, 7, 0, true) => [1; 7; 0; 1;;],
            (1, 8, 0, true) => [1; 8; 0; 1;;],
            # Edge neighbours
            (1, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 7, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 8, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 9, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 10, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 11, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 12, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 1, true) => [1; 1; 0; -1;;],
            (1, 2, 1, true) => [1; 2; 0; -1;;],
            (1, 3, 1, true) => [1; 3; 0; -1;;],
            (1, 4, 1, true) => [1; 4; 0; -1;;],
            (1, 5, 1, true) => [1; 5; 0; -1;;],
            (1, 6, 1, true) => [1; 6; 0; -1;;],
            (1, 7, 1, true) => [1; 7; 0; -1;;],
            (1, 8, 1, true) => [1; 8; 0; -1;;],
            (1, 9, 1, true) => [1; 9; 0; -1;;],
            (1, 10, 1, true) => [1; 10; 0; -1;;],
            (1, 11, 1, true) => [1; 11; 0; -1;;],
            (1, 12, 1, true) => [1; 12; 0; -1;;],
            # Face neighbours
            (1, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 2, true) => [1; 1; 2; 1;;],
            (1, 2, 2, true) => [1; 2; 2; 1;;],
            (1, 3, 2, true) => [1; 3; 2; 1;;],
            (1, 4, 2, true) => [1; 4; 2; 1;;],
            (1, 5, 2, true) => [1; 5; 2; 1;;],
            (1, 6, 2, true) => [1; 6; 2; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
        ]
        @test isempty(interfaces)
        @test isempty(Topology.get_interfaces_on_boundary(topology))
    end

    # The second patch is the first one rotated such that the shared face is local face 2
    # of patch 1 and local face 5 of patch 2.
    @testset "two patches: face rotation 2 (local face 5)" verbose=true begin
        topology = Topology.MeshTopology(
            ((1, 2, 3, 4, 5, 6, 7, 8), (7, 6, 2, 3, 12, 11, 9, 10)), Topology.HEX
        )

        @test Topology.get_manifold_dim(topology) == 3
        @test Topology.get_incidence_relations_dim(topology) == 4
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.HEX)
        @test Topology.get_topological_patch(topology) == Topology.HEX
        @test size(topology) == (12, 20, 11, 2)
        @test @expected Dict(1 => 12, 2 => 20, 3 => 11, 4 => 2) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (8, 12, 6, 1)
        expected_local_size_dim = Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 4

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2, 3],
                [1, 4, 5, 6],
                [4, 7, 8, 9],
                [2, 7, 10],
                [3, 11, 12],
                [5, 11, 13, 14],
                [8, 13, 15, 16],
                [10, 12, 15],
                [6, 17, 18],
                [9, 17, 19],
                [14, 18, 20],
                [16, 19, 20],
            ],
            (1, 3) => [
                [1, 2, 3],
                [1, 2, 4, 5, 6],
                [1, 4, 5, 7, 8],
                [1, 3, 7],
                [2, 3, 9],
                [2, 4, 6, 9, 10],
                [4, 7, 8, 9, 10],
                [3, 7, 9],
                [5, 6, 11],
                [5, 8, 11],
                [6, 10, 11],
                [8, 10, 11],
            ],
            (1, 4) => [
                [1],
                [1, 2],
                [1, 2],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [1],
                [2],
                [2],
                [2],
                [2],
            ],
            (2, 1) => [
                [2, 1],
                [4, 1],
                [5, 1],
                [2, 3],
                [6, 2],
                [9, 2],
                [3, 4],
                [7, 3],
                [10, 3],
                [8, 4],
                [6, 5],
                [8, 5],
                [7, 6],
                [11, 6],
                [7, 8],
                [12, 7],
                [9, 10],
                [9, 11],
                [10, 12],
                [11, 12],
            ],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [
                [1, 2],
                [1, 3],
                [2, 3],
                [1, 4, 5],
                [2, 4, 6],
                [5, 6],
                [1, 7],
                [4, 7, 8],
                [5, 8],
                [3, 7],
                [2, 9],
                [3, 9],
                [4, 9, 10],
                [6, 10],
                [7, 9],
                [8, 10],
                [5, 11],
                [6, 11],
                [8, 11],
                [10, 11],
            ],
            (2, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [2],
                [1],
                [1, 2],
                [2],
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [1],
                [2],
                [2],
                [2],
                [2],
                [2],
            ],
            (3, 1) => [
                [3, 2, 1, 4],
                [6, 5, 1, 2],
                [8, 4, 1, 5],
                [7, 6, 2, 3],
                [9, 2, 3, 10],
                [9, 11, 6, 2],
                [7, 3, 4, 8],
                [10, 3, 7, 12],
                [7, 8, 5, 6],
                [11, 12, 7, 6],
                [9, 10, 12, 11],
            ],
            (3, 2) => [
                [7, 1, -4, 2],
                [5, 3, 11, 1],
                [12, 2, 10, 3],
                [8, 5, 13, -4],
                [17, 4, 6, 9],
                [6, 14, 18, -5],
                [15, 7, 8, 10],
                [19, -8, 9, 16],
                [13, 12, 15, 11],
                [14, 16, 20, -13],
                [18, 19, 17, 20],
            ],
            (3, 3) => Vector{Vector{Int}}(),
            (3, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [2],
                [1],
                [2],
                [1],
                [2],
                [2],
            ],
            (4, 1) => [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [7, 6, 2, 3, 12, 11, 9, 10],
            ],
            (4, 2) => [
                [-15, -11, -1, -7, -13, -12, -2, 4, -8, -10, -3, -5],
                [-17, -20, 13, -4, -18, -19, 8, 5, -6, -9, -16, -14],
            ],
            (4, 3) => [
                [3, 4, 2, 7, 1, 9],
                [8, 6, 10, 5, -4, 11],
            ],
            (4, 4) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            # topology[1, 4]
            (1, 0, 1, 3) => 1,
            (2, 0, 1, 3) => 1,
            (2, 0, 2, 3) => 2,
            (3, 0, 1, 3) => 1,
            (3, 0, 2, 3) => 2,
            (4, 0, 1, 3) => 1,
            (5, 0, 1, 3) => 1,
            (6, 0, 1, 3) => 1,
            (6, 0, 2, 3) => 2,
            (7, 0, 1, 3) => 1,
            (7, 0, 2, 3) => 2,
            (8, 0, 1, 3) => 1,
            (9, 0, 1, 3) => 2,
            (10, 0, 1, 3) => 2,
            (11, 0, 1, 3) => 2,
            (12, 0, 1, 3) => 2,
            # topology[2, 4]
            (1, 1, 1, 3) => 1,
            (2, 1, 1, 3) => 1,
            (3, 1, 1, 3) => 1,
            (4, 1, 1, 3) => 1,
            (4, 1, 2, 3) => 2,
            (5, 1, 1, 3) => 1,
            (5, 1, 2, 3) => 2,
            (6, 1, 1, 3) => 2,
            (7, 1, 1, 3) => 1,
            (8, 1, 1, 3) => 1,
            (8, 1, 2, 3) => 2,
            (9, 1, 1, 3) => 2,
            (10, 1, 1, 3) => 1,
            (11, 1, 1, 3) => 1,
            (12, 1, 1, 3) => 1,
            (13, 1, 1, 3) => 1,
            (13, 1, 2, 3) => 2,
            (14, 1, 1, 3) => 2,
            (15, 1, 1, 3) => 1,
            (16, 1, 1, 3) => 2,
            (17, 1, 1, 3) => 2,
            (18, 1, 1, 3) => 2,
            (19, 1, 1, 3) => 2,
            (20, 1, 1, 3) => 2,
            # topology[3, 4]
            (1, 2, 1, 3) => 1,
            (2, 2, 1, 3) => 1,
            (3, 2, 1, 3) => 1,
            (4, 2, 1, 3) => 1,
            (4, 2, 2, 3) => 2,
            (5, 2, 1, 3) => 2,
            (6, 2, 1, 3) => 2,
            (7, 2, 1, 3) => 1,
            (8, 2, 1, 3) => 2,
            (9, 2, 1, 3) => 1,
            (10, 2, 1, 3) => 2,
            (11, 2, 1, 3) => 2,
            # topology[4, 1]
            (1, 3, 1, 0) => 1,
            (1, 3, 2, 0) => 2,
            (1, 3, 3, 0) => 3,
            (1, 3, 4, 0) => 4,
            (1, 3, 5, 0) => 5,
            (1, 3, 6, 0) => 6,
            (1, 3, 7, 0) => 7,
            (1, 3, 8, 0) => 8,
            (2, 3, 1, 0) => 7,
            (2, 3, 2, 0) => 6,
            (2, 3, 3, 0) => 2,
            (2, 3, 4, 0) => 3,
            (2, 3, 5, 0) => 12,
            (2, 3, 6, 0) => 11,
            (2, 3, 7, 0) => 9,
            (2, 3, 8, 0) => 10,
            # topology[4, 2]
            (1, 3, 1, 1) => -15,
            (1, 3, 2, 1) => -11,
            (1, 3, 3, 1) => -1,
            (1, 3, 4, 1) => -7,
            (1, 3, 5, 1) => -13,
            (1, 3, 6, 1) => -12,
            (1, 3, 7, 1) => -2,
            (1, 3, 8, 1) => 4,
            (1, 3, 9, 1) => -8,
            (1, 3, 10, 1) => -10,
            (1, 3, 11, 1) => -3,
            (1, 3, 12, 1) => -5,
            (2, 3, 1, 1) => -17,
            (2, 3, 2, 1) => -20,
            (2, 3, 3, 1) => 13,
            (2, 3, 4, 1) => -4,
            (2, 3, 5, 1) => -18,
            (2, 3, 6, 1) => -19,
            (2, 3, 7, 1) => 8,
            (2, 3, 8, 1) => 5,
            (2, 3, 9, 1) => -6,
            (2, 3, 10, 1) => -9,
            (2, 3, 11, 1) => -16,
            (2, 3, 12, 1) => -14,
            # topology[4, 3]
            (1, 3, 1, 2) => 3,
            (1, 3, 2, 2) => 4,
            (1, 3, 3, 2) => 2,
            (1, 3, 4, 2) => 7,
            (1, 3, 5, 2) => 1,
            (1, 3, 6, 2) => 9,
            (2, 3, 1, 2) => 8,
            (2, 3, 2, 2) => 6,
            (2, 3, 3, 2) => 10,
            (2, 3, 4, 2) => 5,
            (2, 3, 5, 2) => -4,
            (2, 3, 6, 2) => 11,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (1, 5, 0) => 5,
            (1, 6, 0) => 6,
            (1, 7, 0) => 7,
            (1, 8, 0) => 8,
            (2, 1, 0) => 7,
            (2, 2, 0) => 6,
            (2, 3, 0) => 2,
            (2, 4, 0) => 3,
            (2, 5, 0) => 12,
            (2, 6, 0) => 11,
            (2, 7, 0) => 9,
            (2, 8, 0) => 10,
            (1, 1, 1) => -15,
            (1, 2, 1) => -11,
            (1, 3, 1) => -1,
            (1, 4, 1) => -7,
            (1, 5, 1) => -13,
            (1, 6, 1) => -12,
            (1, 7, 1) => -2,
            (1, 8, 1) => 4,
            (1, 9, 1) => -8,
            (1, 10, 1) => -10,
            (1, 11, 1) => -3,
            (1, 12, 1) => -5,
            (2, 1, 1) => -17,
            (2, 2, 1) => -20,
            (2, 3, 1) => 13,
            (2, 4, 1) => -4,
            (2, 5, 1) => -18,
            (2, 6, 1) => -19,
            (2, 7, 1) => 8,
            (2, 8, 1) => 5,
            (2, 9, 1) => -6,
            (2, 10, 1) => -9,
            (2, 11, 1) => -16,
            (2, 12, 1) => -14,
            (1, 1, 2) => 3,
            (1, 2, 2) => 4,
            (1, 3, 2) => 2,
            (1, 4, 2) => 7,
            (1, 5, 2) => 1,
            (1, 6, 2) => 9,
            (2, 1, 2) => 8,
            (2, 2, 2) => 6,
            (2, 3, 2) => 10,
            (2, 4, 2) => 5,
            (2, 5, 2) => -4,
            (2, 6, 2) => 11,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (1, 5, 0) => 5,
            (1, 6, 0) => 6,
            (1, 7, 0) => 7,
            (1, 8, 0) => 8,
            (2, 7, 0) => 1,
            (2, 6, 0) => 2,
            (2, 2, 0) => 3,
            (2, 3, 0) => 4,
            (2, 12, 0) => 5,
            (2, 11, 0) => 6,
            (2, 9, 0) => 7,
            (2, 10, 0) => 8,
            (1, 15, 1) => -1,
            (1, 11, 1) => -2,
            (1, 1, 1) => -3,
            (1, 7, 1) => -4,
            (1, 13, 1) => -5,
            (1, 12, 1) => -6,
            (1, 2, 1) => -7,
            (1, 4, 1) => 8,
            (1, 8, 1) => -9,
            (1, 10, 1) => -10,
            (1, 3, 1) => -11,
            (1, 5, 1) => -12,
            (2, 17, 1) => -1,
            (2, 20, 1) => -2,
            (2, 13, 1) => 3,
            (2, 4, 1) => -4,
            (2, 18, 1) => -5,
            (2, 19, 1) => -6,
            (2, 8, 1) => 7,
            (2, 5, 1) => 8,
            (2, 6, 1) => -9,
            (2, 9, 1) => -10,
            (2, 16, 1) => -11,
            (2, 14, 1) => -12,
            (1, 3, 2) => 1,
            (1, 4, 2) => 2,
            (1, 2, 2) => 3,
            (1, 7, 2) => 4,
            (1, 1, 2) => 5,
            (1, 9, 2) => 6,
            (2, 8, 2) => 1,
            (2, 6, 2) => 2,
            (2, 10, 2) => 3,
            (2, 5, 2) => 4,
            (2, 4, 2) => -5,
            (2, 11, 2) => 6,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => [2; 3; 0; 1;;],
            (1, 3, 0, false) => [2; 4; 0; 1;;],
            (1, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 0, false) => [2; 2; 0; 1;;],
            (1, 7, 0, false) => [2; 1; 0; 1;;],
            (1, 8, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 0, false) => [1; 7; 0; 1;;],
            (2, 2, 0, false) => [1; 6; 0; 1;;],
            (2, 3, 0, false) => [1; 2; 0; 1;;],
            (2, 4, 0, false) => [1; 3; 0; 1;;],
            (2, 5, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 6, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 7, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 8, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1 2; 2 3; 0 0; 1 1],
            (1, 3, 0, true) => [1 2; 3 4; 0 0; 1 1],
            (1, 4, 0, true) => [1; 4; 0; 1;;],
            (1, 5, 0, true) => [1; 5; 0; 1;;],
            (1, 6, 0, true) => [1 2; 6 2; 0 0; 1 1],
            (1, 7, 0, true) => [1 2; 7 1; 0 0; 1 1],
            (1, 8, 0, true) => [1; 8; 0; 1;;],
            (2, 1, 0, true) => [1 2; 7 1; 0 0; 1 1],
            (2, 2, 0, true) => [1 2; 6 2; 0 0; 1 1],
            (2, 3, 0, true) => [1 2; 2 3; 0 0; 1 1],
            (2, 4, 0, true) => [1 2; 3 4; 0 0; 1 1],
            (2, 5, 0, true) => [2; 5; 0; 1;;],
            (2, 6, 0, true) => [2; 6; 0; 1;;],
            (2, 7, 0, true) => [2; 7; 0; 1;;],
            (2, 8, 0, true) => [2; 8; 0; 1;;],
            # Edge neighbours
            (1, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 1, false) => [2; 3; 0; -1;;],
            (1, 6, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 7, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 8, 1, false) => [2; 4; 0; -1;;],
            (1, 9, 1, false) => [2; 7; 0; -1;;],
            (1, 10, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 11, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 12, 1, false) => [2; 8; 0; -1;;],
            (2, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 1, false) => [1; 5; 0; -1;;],
            (2, 4, 1, false) => [1; 8; 0; -1;;],
            (2, 5, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 6, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 7, 1, false) => [1; 9; 0; -1;;],
            (2, 8, 1, false) => [1; 12; 0; -1;;],
            (2, 9, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 10, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 11, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 12, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 1, true) => [1; 1; 0; -1;;],
            (1, 2, 1, true) => [1; 2; 0; -1;;],
            (1, 3, 1, true) => [1; 3; 0; -1;;],
            (1, 4, 1, true) => [1; 4; 0; -1;;],
            (1, 5, 1, true) => [1 2; 5 3; 0 0; -1 1],
            (1, 6, 1, true) => [1; 6; 0; -1;;],
            (1, 7, 1, true) => [1; 7; 0; -1;;],
            (1, 8, 1, true) => [1 2; 8 4; 0 0; 1 -1],
            (1, 9, 1, true) => [1 2; 9 7; 0 0; -1 1],
            (1, 10, 1, true) => [1; 10; 0; -1;;],
            (1, 11, 1, true) => [1; 11; 0; -1;;],
            (1, 12, 1, true) => [1 2; 12 8; 0 0; -1 1],
            (2, 1, 1, true) => [2; 1; 0; -1;;],
            (2, 2, 1, true) => [2; 2; 0; -1;;],
            (2, 3, 1, true) => [1 2; 5 3; 0 0; -1 1],
            (2, 4, 1, true) => [1 2; 8 4; 0 0; 1 -1],
            (2, 5, 1, true) => [2; 5; 0; -1;;],
            (2, 6, 1, true) => [2; 6; 0; -1;;],
            (2, 7, 1, true) => [1 2; 9 7; 0 0; -1 1],
            (2, 8, 1, true) => [1 2; 12 8; 0 0; -1 1],
            (2, 9, 1, true) => [2; 9; 0; -1;;],
            (2, 10, 1, true) => [2; 10; 0; -1;;],
            (2, 11, 1, true) => [2; 11; 0; -1;;],
            (2, 12, 1, true) => [2; 12; 0; -1;;],
            # Face neighbours
            (1, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 2, false) => [2; 5; 2; -1;;],
            (1, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 2, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 5, 2, false) => [1; 2; 2; -1;;],
            (2, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 2, true) => [1; 1; 2; 1;;],
            (1, 2, 2, true) => [1 2; 2 5; 2 0; 1 -1],
            (1, 3, 2, true) => [1; 3; 2; 1;;],
            (1, 4, 2, true) => [1; 4; 2; 1;;],
            (1, 5, 2, true) => [1; 5; 2; 1;;],
            (1, 6, 2, true) => [1; 6; 2; 1;;],
            (2, 1, 2, true) => [2; 1; 2; 1;;],
            (2, 2, 2, true) => [2; 2; 2; 1;;],
            (2, 3, 2, true) => [2; 3; 2; 1;;],
            (2, 4, 2, true) => [2; 4; 2; 1;;],
            (2, 5, 2, true) => [1 2; 2 5; 2 0; 1 -1],
            (2, 6, 2, true) => [2; 6; 2; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (0, 1),
            (0, 4),
            (0, 5),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
        ]
        @test interfaces == [
            (2, 4),
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
        @test Topology.get_interfaces_on_boundary(topology) == [
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
    end

    # The next testsets glue two hexahedra along local face 2 of the first one. The four
    # connectivities are obtained by rotating the second hexahedron about the shared face, and
    # therefore cover every rotation that can be reported. The orientation is always -1: two
    # hexahedra on opposite sides of a face traverse it in opposite cyclic directions.
    @testset "two patches: face rotation 0 (local face 1)" verbose=true begin
        topology = Topology.MeshTopology(
            ((1, 2, 3, 4, 5, 6, 7, 8), (2, 9, 10, 3, 6, 11, 12, 7)), Topology.HEX
        )

        @test Topology.get_manifold_dim(topology) == 3
        @test Topology.get_incidence_relations_dim(topology) == 4
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.HEX)
        @test Topology.get_topological_patch(topology) == Topology.HEX
        @test size(topology) == (12, 20, 11, 2)
        @test @expected Dict(1 => 12, 2 => 20, 3 => 11, 4 => 2) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (8, 12, 6, 1)
        expected_local_size_dim = Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 4

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2, 3],
                [1, 4, 5, 6],
                [4, 7, 8, 9],
                [2, 7, 10],
                [3, 11, 12],
                [5, 11, 13, 14],
                [8, 13, 15, 16],
                [10, 12, 15],
                [6, 17, 18],
                [9, 17, 19],
                [14, 18, 20],
                [16, 19, 20],
            ],
            (1, 3) => [
                [1, 2, 3],
                [1, 2, 4, 5, 6],
                [1, 4, 5, 7, 8],
                [1, 3, 7],
                [2, 3, 9],
                [2, 4, 6, 9, 10],
                [4, 7, 8, 9, 10],
                [3, 7, 9],
                [5, 6, 11],
                [5, 8, 11],
                [6, 10, 11],
                [8, 10, 11],
            ],
            (1, 4) => [
                [1],
                [1, 2],
                [1, 2],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [1],
                [2],
                [2],
                [2],
                [2],
            ],
            (2, 1) => [
                [2, 1],
                [4, 1],
                [5, 1],
                [3, 2],
                [6, 2],
                [9, 2],
                [3, 4],
                [7, 3],
                [10, 3],
                [8, 4],
                [6, 5],
                [8, 5],
                [7, 6],
                [11, 6],
                [7, 8],
                [12, 7],
                [10, 9],
                [11, 9],
                [12, 10],
                [12, 11],
            ],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [
                [1, 2],
                [1, 3],
                [2, 3],
                [1, 4, 5],
                [2, 4, 6],
                [5, 6],
                [1, 7],
                [4, 7, 8],
                [5, 8],
                [3, 7],
                [2, 9],
                [3, 9],
                [4, 9, 10],
                [6, 10],
                [7, 9],
                [8, 10],
                [5, 11],
                [6, 11],
                [8, 11],
                [10, 11],
            ],
            (2, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [2],
                [1],
                [1, 2],
                [2],
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [1],
                [2],
                [2],
                [2],
                [2],
                [2],
            ],
            (3, 1) => [
                [3, 2, 1, 4],
                [6, 5, 1, 2],
                [8, 4, 1, 5],
                [7, 3, 2, 6],
                [10, 9, 2, 3],
                [11, 6, 2, 9],
                [7, 3, 4, 8],
                [12, 10, 3, 7],
                [7, 8, 5, 6],
                [12, 7, 6, 11],
                [12, 11, 9, 10],
            ],
            (3, 2) => [
                [7, 1, 4, 2],
                [5, 3, 11, 1],
                [12, 2, 10, 3],
                [13, 4, 8, 5],
                [9, 6, 17, 4],
                [18, 5, 14, 6],
                [15, 7, 8, 10],
                [16, 9, 19, 8],
                [13, 12, 15, 11],
                [20, 13, 16, 14],
                [19, 18, 20, 17],
            ],
            (3, 3) => Vector{Vector{Int}}(),
            (3, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [2],
                [1],
                [2],
                [1],
                [2],
                [2],
            ],
            (4, 1) => [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 9, 10, 3, 6, 11, 12, 7],
            ],
            (4, 2) => [
                [-15, -11, -1, -7, -13, -12, -2, -4, -8, -10, -3, -5],
                [-16, -14, -6, -9, -20, -13, -4, -17, -19, -8, -5, -18],
            ],
            (4, 3) => [
                [3, -4, 2, 7, 1, 9],
                [4, 11, 6, 8, 5, 10],
            ],
            (4, 4) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Face neighbours
            (1, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 2, false) => [2; 1; 0; -1;;],
            (1, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 2, false) => [1; 2; 0; -1;;],
            (2, 2, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 2, true) => [1; 1; 2; 1;;],
            (1, 2, 2, true) => [1 2; 2 1; 2 2; -1 1],
            (1, 3, 2, true) => [1; 3; 2; 1;;],
            (1, 4, 2, true) => [1; 4; 2; 1;;],
            (1, 5, 2, true) => [1; 5; 2; 1;;],
            (1, 6, 2, true) => [1; 6; 2; 1;;],
            (2, 1, 2, true) => [1 2; 2 1; 2 2; -1 1],
            (2, 2, 2, true) => [2; 2; 2; 1;;],
            (2, 3, 2, true) => [2; 3; 2; 1;;],
            (2, 4, 2, true) => [2; 4; 2; 1;;],
            (2, 5, 2, true) => [2; 5; 2; 1;;],
            (2, 6, 2, true) => [2; 6; 2; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (0, 1),
            (0, 4),
            (0, 5),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
        ]
        @test interfaces == [
            (2, 4),
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
        @test Topology.get_interfaces_on_boundary(topology) == [
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
    end

    @testset "two patches: face rotation 1 (local face 2)" verbose=true begin
        topology = Topology.MeshTopology(
            ((1, 2, 3, 4, 5, 6, 7, 8), (10, 3, 2, 9, 12, 7, 6, 11)), Topology.HEX
        )

        @test Topology.get_manifold_dim(topology) == 3
        @test Topology.get_incidence_relations_dim(topology) == 4
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.HEX)
        @test Topology.get_topological_patch(topology) == Topology.HEX
        @test size(topology) == (12, 20, 11, 2)
        @test @expected Dict(1 => 12, 2 => 20, 3 => 11, 4 => 2) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (8, 12, 6, 1)
        expected_local_size_dim = Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 4

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2, 3],
                [1, 4, 5, 6],
                [4, 7, 8, 9],
                [2, 7, 10],
                [3, 11, 12],
                [5, 11, 13, 14],
                [8, 13, 15, 16],
                [10, 12, 15],
                [6, 17, 18],
                [9, 17, 19],
                [14, 18, 20],
                [16, 19, 20],
            ],
            (1, 3) => [
                [1, 2, 3],
                [1, 2, 4, 5, 6],
                [1, 4, 5, 7, 8],
                [1, 3, 7],
                [2, 3, 9],
                [2, 4, 6, 9, 10],
                [4, 7, 8, 9, 10],
                [3, 7, 9],
                [5, 6, 11],
                [5, 8, 11],
                [6, 10, 11],
                [8, 10, 11],
            ],
            (1, 4) => [
                [1],
                [1, 2],
                [1, 2],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [1],
                [2],
                [2],
                [2],
                [2],
            ],
            (2, 1) => [
                [2, 1],
                [4, 1],
                [5, 1],
                [3, 2],
                [6, 2],
                [2, 9],
                [3, 4],
                [7, 3],
                [3, 10],
                [8, 4],
                [6, 5],
                [8, 5],
                [7, 6],
                [6, 11],
                [7, 8],
                [7, 12],
                [9, 10],
                [11, 9],
                [12, 10],
                [11, 12],
            ],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [
                [1, 2],
                [1, 3],
                [2, 3],
                [1, 4, 5],
                [2, 4, 6],
                [5, 6],
                [1, 7],
                [4, 7, 8],
                [5, 8],
                [3, 7],
                [2, 9],
                [3, 9],
                [4, 9, 10],
                [6, 10],
                [7, 9],
                [8, 10],
                [5, 11],
                [6, 11],
                [8, 11],
                [10, 11],
            ],
            (2, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [2],
                [1],
                [1, 2],
                [2],
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [1],
                [2],
                [2],
                [2],
                [2],
                [2],
            ],
            (3, 1) => [
                [3, 2, 1, 4],
                [6, 5, 1, 2],
                [8, 4, 1, 5],
                [7, 6, 2, 3],
                [2, 3, 10, 9],
                [6, 2, 9, 11],
                [7, 3, 4, 8],
                [7, 12, 10, 3],
                [7, 8, 5, 6],
                [6, 11, 12, 7],
                [11, 9, 10, 12],
            ],
            (3, 2) => [
                [7, 1, 4, 2],
                [5, 3, 11, 1],
                [12, 2, 10, 3],
                [8, 5, 13, 4],
                [6, 9, -4, 17],
                [14, 6, 5, 18],
                [15, 7, 8, 10],
                [8, 19, 16, 9],
                [13, 12, 15, 11],
                [-13, 20, 14, 16],
                [20, 17, 18, 19],
            ],
            (3, 3) => Vector{Vector{Int}}(),
            (3, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [2],
                [1],
                [2],
                [1],
                [2],
                [2],
            ],
            (4, 1) => [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [10, 3, 2, 9, 12, 7, 6, 11],
            ],
            (4, 2) => [
                [-15, -11, -1, -7, -13, -12, -2, -4, -8, -10, -3, -5],
                [-14, -16, -9, -6, 13, -20, -17, 4, -5, -18, -19, -8],
            ],
            (4, 3) => [
                [3, 4, 2, 7, 1, 9],
                [11, -4, 8, 6, 5, 10],
            ],
            (4, 4) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            # topology[1, 4]
            (1, 0, 1, 3) => 1,
            (2, 0, 1, 3) => 1,
            (2, 0, 2, 3) => 2,
            (3, 0, 1, 3) => 1,
            (3, 0, 2, 3) => 2,
            (4, 0, 1, 3) => 1,
            (5, 0, 1, 3) => 1,
            (6, 0, 1, 3) => 1,
            (6, 0, 2, 3) => 2,
            (7, 0, 1, 3) => 1,
            (7, 0, 2, 3) => 2,
            (8, 0, 1, 3) => 1,
            (9, 0, 1, 3) => 2,
            (10, 0, 1, 3) => 2,
            (11, 0, 1, 3) => 2,
            (12, 0, 1, 3) => 2,
            # topology[2, 4]
            (1, 1, 1, 3) => 1,
            (2, 1, 1, 3) => 1,
            (3, 1, 1, 3) => 1,
            (4, 1, 1, 3) => 1,
            (4, 1, 2, 3) => 2,
            (5, 1, 1, 3) => 1,
            (5, 1, 2, 3) => 2,
            (6, 1, 1, 3) => 2,
            (7, 1, 1, 3) => 1,
            (8, 1, 1, 3) => 1,
            (8, 1, 2, 3) => 2,
            (9, 1, 1, 3) => 2,
            (10, 1, 1, 3) => 1,
            (11, 1, 1, 3) => 1,
            (12, 1, 1, 3) => 1,
            (13, 1, 1, 3) => 1,
            (13, 1, 2, 3) => 2,
            (14, 1, 1, 3) => 2,
            (15, 1, 1, 3) => 1,
            (16, 1, 1, 3) => 2,
            (17, 1, 1, 3) => 2,
            (18, 1, 1, 3) => 2,
            (19, 1, 1, 3) => 2,
            (20, 1, 1, 3) => 2,
            # topology[3, 4]
            (1, 2, 1, 3) => 1,
            (2, 2, 1, 3) => 1,
            (3, 2, 1, 3) => 1,
            (4, 2, 1, 3) => 1,
            (4, 2, 2, 3) => 2,
            (5, 2, 1, 3) => 2,
            (6, 2, 1, 3) => 2,
            (7, 2, 1, 3) => 1,
            (8, 2, 1, 3) => 2,
            (9, 2, 1, 3) => 1,
            (10, 2, 1, 3) => 2,
            (11, 2, 1, 3) => 2,
            # topology[4, 1]
            (1, 3, 1, 0) => 1,
            (1, 3, 2, 0) => 2,
            (1, 3, 3, 0) => 3,
            (1, 3, 4, 0) => 4,
            (1, 3, 5, 0) => 5,
            (1, 3, 6, 0) => 6,
            (1, 3, 7, 0) => 7,
            (1, 3, 8, 0) => 8,
            (2, 3, 1, 0) => 10,
            (2, 3, 2, 0) => 3,
            (2, 3, 3, 0) => 2,
            (2, 3, 4, 0) => 9,
            (2, 3, 5, 0) => 12,
            (2, 3, 6, 0) => 7,
            (2, 3, 7, 0) => 6,
            (2, 3, 8, 0) => 11,
            # topology[4, 2]
            (1, 3, 1, 1) => -15,
            (1, 3, 2, 1) => -11,
            (1, 3, 3, 1) => -1,
            (1, 3, 4, 1) => -7,
            (1, 3, 5, 1) => -13,
            (1, 3, 6, 1) => -12,
            (1, 3, 7, 1) => -2,
            (1, 3, 8, 1) => -4,
            (1, 3, 9, 1) => -8,
            (1, 3, 10, 1) => -10,
            (1, 3, 11, 1) => -3,
            (1, 3, 12, 1) => -5,
            (2, 3, 1, 1) => -14,
            (2, 3, 2, 1) => -16,
            (2, 3, 3, 1) => -9,
            (2, 3, 4, 1) => -6,
            (2, 3, 5, 1) => 13,
            (2, 3, 6, 1) => -20,
            (2, 3, 7, 1) => -17,
            (2, 3, 8, 1) => 4,
            (2, 3, 9, 1) => -5,
            (2, 3, 10, 1) => -18,
            (2, 3, 11, 1) => -19,
            (2, 3, 12, 1) => -8,
            # topology[4, 3]
            (1, 3, 1, 2) => 3,
            (1, 3, 2, 2) => 4,
            (1, 3, 3, 2) => 2,
            (1, 3, 4, 2) => 7,
            (1, 3, 5, 2) => 1,
            (1, 3, 6, 2) => 9,
            (2, 3, 1, 2) => 11,
            (2, 3, 2, 2) => -4,
            (2, 3, 3, 2) => 8,
            (2, 3, 4, 2) => 6,
            (2, 3, 5, 2) => 5,
            (2, 3, 6, 2) => 10,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (1, 5, 0) => 5,
            (1, 6, 0) => 6,
            (1, 7, 0) => 7,
            (1, 8, 0) => 8,
            (2, 1, 0) => 10,
            (2, 2, 0) => 3,
            (2, 3, 0) => 2,
            (2, 4, 0) => 9,
            (2, 5, 0) => 12,
            (2, 6, 0) => 7,
            (2, 7, 0) => 6,
            (2, 8, 0) => 11,
            (1, 1, 1) => -15,
            (1, 2, 1) => -11,
            (1, 3, 1) => -1,
            (1, 4, 1) => -7,
            (1, 5, 1) => -13,
            (1, 6, 1) => -12,
            (1, 7, 1) => -2,
            (1, 8, 1) => -4,
            (1, 9, 1) => -8,
            (1, 10, 1) => -10,
            (1, 11, 1) => -3,
            (1, 12, 1) => -5,
            (2, 1, 1) => -14,
            (2, 2, 1) => -16,
            (2, 3, 1) => -9,
            (2, 4, 1) => -6,
            (2, 5, 1) => 13,
            (2, 6, 1) => -20,
            (2, 7, 1) => -17,
            (2, 8, 1) => 4,
            (2, 9, 1) => -5,
            (2, 10, 1) => -18,
            (2, 11, 1) => -19,
            (2, 12, 1) => -8,
            (1, 1, 2) => 3,
            (1, 2, 2) => 4,
            (1, 3, 2) => 2,
            (1, 4, 2) => 7,
            (1, 5, 2) => 1,
            (1, 6, 2) => 9,
            (2, 1, 2) => 11,
            (2, 2, 2) => -4,
            (2, 3, 2) => 8,
            (2, 4, 2) => 6,
            (2, 5, 2) => 5,
            (2, 6, 2) => 10,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (1, 3, 0) => 3,
            (1, 4, 0) => 4,
            (1, 5, 0) => 5,
            (1, 6, 0) => 6,
            (1, 7, 0) => 7,
            (1, 8, 0) => 8,
            (2, 10, 0) => 1,
            (2, 3, 0) => 2,
            (2, 2, 0) => 3,
            (2, 9, 0) => 4,
            (2, 12, 0) => 5,
            (2, 7, 0) => 6,
            (2, 6, 0) => 7,
            (2, 11, 0) => 8,
            (1, 15, 1) => -1,
            (1, 11, 1) => -2,
            (1, 1, 1) => -3,
            (1, 7, 1) => -4,
            (1, 13, 1) => -5,
            (1, 12, 1) => -6,
            (1, 2, 1) => -7,
            (1, 4, 1) => -8,
            (1, 8, 1) => -9,
            (1, 10, 1) => -10,
            (1, 3, 1) => -11,
            (1, 5, 1) => -12,
            (2, 14, 1) => -1,
            (2, 16, 1) => -2,
            (2, 9, 1) => -3,
            (2, 6, 1) => -4,
            (2, 13, 1) => 5,
            (2, 20, 1) => -6,
            (2, 17, 1) => -7,
            (2, 4, 1) => 8,
            (2, 5, 1) => -9,
            (2, 18, 1) => -10,
            (2, 19, 1) => -11,
            (2, 8, 1) => -12,
            (1, 3, 2) => 1,
            (1, 4, 2) => 2,
            (1, 2, 2) => 3,
            (1, 7, 2) => 4,
            (1, 1, 2) => 5,
            (1, 9, 2) => 6,
            (2, 11, 2) => 1,
            (2, 4, 2) => -2,
            (2, 8, 2) => 3,
            (2, 6, 2) => 4,
            (2, 5, 2) => 5,
            (2, 10, 2) => 6,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 0, false) => [2; 3; 0; 1;;],
            (1, 3, 0, false) => [2; 2; 0; 1;;],
            (1, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 0, false) => [2; 7; 0; 1;;],
            (1, 7, 0, false) => [2; 6; 0; 1;;],
            (1, 8, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 2, 0, false) => [1; 3; 0; 1;;],
            (2, 3, 0, false) => [1; 2; 0; 1;;],
            (2, 4, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 5, 0, false) => Matrix{Int}(undef, 4, 0),
            (2, 6, 0, false) => [1; 7; 0; 1;;],
            (2, 7, 0, false) => [1; 6; 0; 1;;],
            (2, 8, 0, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 0, true) => [1; 1; 0; 1;;],
            (1, 2, 0, true) => [1 2; 2 3; 0 0; 1 1],
            (1, 3, 0, true) => [1 2; 3 2; 0 0; 1 1],
            (1, 4, 0, true) => [1; 4; 0; 1;;],
            (1, 5, 0, true) => [1; 5; 0; 1;;],
            (1, 6, 0, true) => [1 2; 6 7; 0 0; 1 1],
            (1, 7, 0, true) => [1 2; 7 6; 0 0; 1 1],
            (1, 8, 0, true) => [1; 8; 0; 1;;],
            (2, 1, 0, true) => [2; 1; 0; 1;;],
            (2, 2, 0, true) => [1 2; 3 2; 0 0; 1 1],
            (2, 3, 0, true) => [1 2; 2 3; 0 0; 1 1],
            (2, 4, 0, true) => [2; 4; 0; 1;;],
            (2, 5, 0, true) => [2; 5; 0; 1;;],
            (2, 6, 0, true) => [1 2; 7 6; 0 0; 1 1],
            (2, 7, 0, true) => [1 2; 6 7; 0 0; 1 1],
            (2, 8, 0, true) => [2; 8; 0; 1;;],
            # Edge neighbours
            (1, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 1, false) => [2; 5; 0; -1;;],
            (1, 6, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 7, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 8, 1, false) => [2; 8; 0; -1;;],
            (1, 9, 1, false) => [2; 12; 0; 1;;],
            (1, 10, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 11, 1, false) => Matrix{Int}(undef, 4, 0),
            (1, 12, 1, false) => [2; 9; 0; 1;;],
            (2, 1, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 2, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 5, 1, false) => [1; 5; 0; -1;;],
            (2, 6, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 7, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 8, 1, false) => [1; 8; 0; -1;;],
            (2, 9, 1, false) => [1; 12; 0; 1;;],
            (2, 10, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 11, 1, false) => Matrix{Int}(undef, 4, 0),
            (2, 12, 1, false) => [1; 9; 0; 1;;],
            (1, 1, 1, true) => [1; 1; 0; -1;;],
            (1, 2, 1, true) => [1; 2; 0; -1;;],
            (1, 3, 1, true) => [1; 3; 0; -1;;],
            (1, 4, 1, true) => [1; 4; 0; -1;;],
            (1, 5, 1, true) => [1 2; 5 5; 0 0; -1 1],
            (1, 6, 1, true) => [1; 6; 0; -1;;],
            (1, 7, 1, true) => [1; 7; 0; -1;;],
            (1, 8, 1, true) => [1 2; 8 8; 0 0; -1 1],
            (1, 9, 1, true) => [1 2; 9 12; 0 0; -1 -1],
            (1, 10, 1, true) => [1; 10; 0; -1;;],
            (1, 11, 1, true) => [1; 11; 0; -1;;],
            (1, 12, 1, true) => [1 2; 12 9; 0 0; -1 -1],
            (2, 1, 1, true) => [2; 1; 0; -1;;],
            (2, 2, 1, true) => [2; 2; 0; -1;;],
            (2, 3, 1, true) => [2; 3; 0; -1;;],
            (2, 4, 1, true) => [2; 4; 0; -1;;],
            (2, 5, 1, true) => [1 2; 5 5; 0 0; -1 1],
            (2, 6, 1, true) => [2; 6; 0; -1;;],
            (2, 7, 1, true) => [2; 7; 0; -1;;],
            (2, 8, 1, true) => [1 2; 8 8; 0 0; -1 1],
            (2, 9, 1, true) => [1 2; 12 9; 0 0; -1 -1],
            (2, 10, 1, true) => [2; 10; 0; -1;;],
            (2, 11, 1, true) => [2; 11; 0; -1;;],
            (2, 12, 1, true) => [1 2; 9 12; 0 0; -1 -1],
            # Face neighbours
            (1, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 2, false) => [2; 2; 1; -1;;],
            (1, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 2, 2, false) => [1; 2; 1; -1;;],
            (2, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 2, true) => [1; 1; 2; 1;;],
            (1, 2, 2, true) => [1 2; 2 2; 2 3; 1 -1],
            (1, 3, 2, true) => [1; 3; 2; 1;;],
            (1, 4, 2, true) => [1; 4; 2; 1;;],
            (1, 5, 2, true) => [1; 5; 2; 1;;],
            (1, 6, 2, true) => [1; 6; 2; 1;;],
            (2, 1, 2, true) => [2; 1; 2; 1;;],
            (2, 2, 2, true) => [1 2; 2 2; 2 3; 1 -1],
            (2, 3, 2, true) => [2; 3; 2; 1;;],
            (2, 4, 2, true) => [2; 4; 2; 1;;],
            (2, 5, 2, true) => [2; 5; 2; 1;;],
            (2, 6, 2, true) => [2; 6; 2; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (0, 1),
            (0, 4),
            (0, 5),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
        ]
        @test interfaces == [
            (2, 4),
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
        @test Topology.get_interfaces_on_boundary(topology) == [
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
    end

    @testset "two patches: face rotation 2 (local face 1)" verbose=true begin
        topology = Topology.MeshTopology(
            ((1, 2, 3, 4, 5, 6, 7, 8), (7, 12, 11, 6, 3, 10, 9, 2)), Topology.HEX
        )

        @test Topology.get_manifold_dim(topology) == 3
        @test Topology.get_incidence_relations_dim(topology) == 4
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.HEX)
        @test Topology.get_topological_patch(topology) == Topology.HEX
        @test size(topology) == (12, 20, 11, 2)
        @test @expected Dict(1 => 12, 2 => 20, 3 => 11, 4 => 2) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (8, 12, 6, 1)
        expected_local_size_dim = Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 4

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2, 3],
                [1, 4, 5, 6],
                [4, 7, 8, 9],
                [2, 7, 10],
                [3, 11, 12],
                [5, 11, 13, 14],
                [8, 13, 15, 16],
                [10, 12, 15],
                [6, 17, 18],
                [9, 17, 19],
                [14, 18, 20],
                [16, 19, 20],
            ],
            (1, 3) => [
                [1, 2, 3],
                [1, 2, 4, 5, 6],
                [1, 4, 5, 7, 8],
                [1, 3, 7],
                [2, 3, 9],
                [2, 4, 6, 9, 10],
                [4, 7, 8, 9, 10],
                [3, 7, 9],
                [5, 6, 11],
                [5, 8, 11],
                [6, 10, 11],
                [8, 10, 11],
            ],
            (1, 4) => [
                [1],
                [1, 2],
                [1, 2],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [1],
                [2],
                [2],
                [2],
                [2],
            ],
            (2, 1) => [
                [2, 1],
                [4, 1],
                [5, 1],
                [2, 3],
                [6, 2],
                [9, 2],
                [3, 4],
                [3, 7],
                [10, 3],
                [8, 4],
                [6, 5],
                [8, 5],
                [7, 6],
                [11, 6],
                [7, 8],
                [12, 7],
                [9, 10],
                [9, 11],
                [10, 12],
                [11, 12],
            ],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [
                [1, 2],
                [1, 3],
                [2, 3],
                [1, 4, 5],
                [2, 4, 6],
                [5, 6],
                [1, 7],
                [4, 7, 8],
                [5, 8],
                [3, 7],
                [2, 9],
                [3, 9],
                [4, 9, 10],
                [6, 10],
                [7, 9],
                [8, 10],
                [5, 11],
                [6, 11],
                [8, 11],
                [10, 11],
            ],
            (2, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [2],
                [1],
                [1, 2],
                [2],
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [1],
                [2],
                [2],
                [2],
                [2],
                [2],
            ],
            (3, 1) => [
                [3, 2, 1, 4],
                [6, 5, 1, 2],
                [8, 4, 1, 5],
                [2, 6, 7, 3],
                [9, 2, 3, 10],
                [9, 11, 6, 2],
                [7, 3, 4, 8],
                [10, 3, 7, 12],
                [7, 8, 5, 6],
                [11, 12, 7, 6],
                [9, 10, 12, 11],
            ],
            (3, 2) => [
                [7, 1, -4, 2],
                [5, 3, 11, 1],
                [12, 2, 10, 3],
                [4, -13, -5, 8],
                [17, 4, 6, 9],
                [6, 14, 18, -5],
                [15, 7, -8, 10],
                [19, 8, 9, 16],
                [13, 12, 15, 11],
                [14, 16, 20, -13],
                [18, 19, 17, 20],
            ],
            (3, 3) => Vector{Vector{Int}}(),
            (3, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [2],
                [1],
                [2],
                [1],
                [2],
                [2],
            ],
            (4, 1) => [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [7, 12, 11, 6, 3, 10, 9, 2],
            ],
            (4, 2) => [
                [-15, -11, -1, -7, -13, -12, -2, 4, 8, -10, -3, -5],
                [-6, -9, -16, -14, -17, -4, 13, -20, -18, 5, -8, -19],
            ],
            (4, 3) => [
                [3, -4, 2, 7, 1, 9],
                [4, 11, 8, 6, 10, 5],
            ],
            (4, 4) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Face neighbours
            (1, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 2, false) => [2; 1; 2; -1;;],
            (1, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 2, false) => [1; 2; 2; -1;;],
            (2, 2, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 2, true) => [1; 1; 2; 1;;],
            (1, 2, 2, true) => [1 2; 2 1; 0 2; -1 1],
            (1, 3, 2, true) => [1; 3; 2; 1;;],
            (1, 4, 2, true) => [1; 4; 2; 1;;],
            (1, 5, 2, true) => [1; 5; 2; 1;;],
            (1, 6, 2, true) => [1; 6; 2; 1;;],
            (2, 1, 2, true) => [1 2; 2 1; 0 2; -1 1],
            (2, 2, 2, true) => [2; 2; 2; 1;;],
            (2, 3, 2, true) => [2; 3; 2; 1;;],
            (2, 4, 2, true) => [2; 4; 2; 1;;],
            (2, 5, 2, true) => [2; 5; 2; 1;;],
            (2, 6, 2, true) => [2; 6; 2; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (0, 1),
            (0, 4),
            (0, 5),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
        ]
        @test interfaces == [
            (2, 4),
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
        @test Topology.get_interfaces_on_boundary(topology) == [
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
    end

    @testset "two patches: face rotation 3 (local face 2)" verbose=true begin
        topology = Topology.MeshTopology(
            ((1, 2, 3, 4, 5, 6, 7, 8), (11, 6, 7, 12, 9, 2, 3, 10)), Topology.HEX
        )

        @test Topology.get_manifold_dim(topology) == 3
        @test Topology.get_incidence_relations_dim(topology) == 4
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.HEX)
        @test Topology.get_topological_patch(topology) == Topology.HEX
        @test size(topology) == (12, 20, 11, 2)
        @test @expected Dict(1 => 12, 2 => 20, 3 => 11, 4 => 2) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (8, 12, 6, 1)
        expected_local_size_dim = Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 4

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2, 3],
                [1, 4, 5, 6],
                [4, 7, 8, 9],
                [2, 7, 10],
                [3, 11, 12],
                [5, 11, 13, 14],
                [8, 13, 15, 16],
                [10, 12, 15],
                [6, 17, 18],
                [9, 17, 19],
                [14, 18, 20],
                [16, 19, 20],
            ],
            (1, 3) => [
                [1, 2, 3],
                [1, 2, 4, 5, 6],
                [1, 4, 5, 7, 8],
                [1, 3, 7],
                [2, 3, 9],
                [2, 4, 6, 9, 10],
                [4, 7, 8, 9, 10],
                [3, 7, 9],
                [5, 6, 11],
                [5, 8, 11],
                [6, 10, 11],
                [8, 10, 11],
            ],
            (1, 4) => [
                [1],
                [1, 2],
                [1, 2],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [1],
                [2],
                [2],
                [2],
                [2],
            ],
            (2, 1) => [
                [2, 1],
                [4, 1],
                [5, 1],
                [3, 2],
                [6, 2],
                [2, 9],
                [3, 4],
                [7, 3],
                [3, 10],
                [8, 4],
                [6, 5],
                [8, 5],
                [7, 6],
                [6, 11],
                [7, 8],
                [7, 12],
                [10, 9],
                [9, 11],
                [10, 12],
                [12, 11],
            ],
            (2, 2) => Vector{Vector{Int}}(),
            (2, 3) => [
                [1, 2],
                [1, 3],
                [2, 3],
                [1, 4, 5],
                [2, 4, 6],
                [5, 6],
                [1, 7],
                [4, 7, 8],
                [5, 8],
                [3, 7],
                [2, 9],
                [3, 9],
                [4, 9, 10],
                [6, 10],
                [7, 9],
                [8, 10],
                [5, 11],
                [6, 11],
                [8, 11],
                [10, 11],
            ],
            (2, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [1, 2],
                [2],
                [1],
                [1, 2],
                [2],
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [1],
                [2],
                [2],
                [2],
                [2],
                [2],
            ],
            (3, 1) => [
                [3, 2, 1, 4],
                [6, 5, 1, 2],
                [8, 4, 1, 5],
                [7, 6, 2, 3],
                [3, 10, 9, 2],
                [2, 9, 11, 6],
                [7, 3, 4, 8],
                [3, 7, 12, 10],
                [7, 8, 5, 6],
                [7, 6, 11, 12],
                [10, 12, 11, 9],
            ],
            (3, 2) => [
                [7, 1, 4, 2],
                [5, 3, 11, 1],
                [12, 2, 10, 3],
                [8, 5, 13, 4],
                [4, 17, 9, 6],
                [-5, 18, 6, 14],
                [15, 7, 8, 10],
                [9, 16, -8, 19],
                [13, 12, 15, 11],
                [16, 14, 13, 20],
                [17, 20, 19, 18],
            ],
            (3, 3) => Vector{Vector{Int}}(),
            (3, 4) => [
                [1],
                [1],
                [1],
                [1, 2],
                [2],
                [2],
                [1],
                [2],
                [1],
                [2],
                [2],
            ],
            (4, 1) => [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [11, 6, 7, 12, 9, 2, 3, 10],
            ],
            (4, 2) => [
                [-15, -11, -1, -7, -13, -12, -2, -4, -8, -10, -3, -5],
                [-9, -6, -14, -16, -4, -17, -20, -13, 8, -19, -18, 5],
            ],
            (4, 3) => [
                [3, 4, 2, 7, 1, 9],
                [11, -4, 6, 8, 10, 5],
            ],
            (4, 4) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Face neighbours
            (1, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 2, 2, false) => [2; 2; 3; -1;;],
            (1, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 1, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 2, 2, false) => [1; 2; 3; -1;;],
            (2, 3, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 4, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 5, 2, false) => Matrix{Int}(undef, 4, 0),
            (2, 6, 2, false) => Matrix{Int}(undef, 4, 0),
            (1, 1, 2, true) => [1; 1; 2; 1;;],
            (1, 2, 2, true) => [1 2; 2 2; 2 1; 1 -1],
            (1, 3, 2, true) => [1; 3; 2; 1;;],
            (1, 4, 2, true) => [1; 4; 2; 1;;],
            (1, 5, 2, true) => [1; 5; 2; 1;;],
            (1, 6, 2, true) => [1; 6; 2; 1;;],
            (2, 1, 2, true) => [2; 1; 2; 1;;],
            (2, 2, 2, true) => [1 2; 2 2; 2 1; 1 -1],
            (2, 3, 2, true) => [2; 3; 2; 1;;],
            (2, 4, 2, true) => [2; 4; 2; 1;;],
            (2, 5, 2, true) => [2; 5; 2; 1;;],
            (2, 6, 2, true) => [2; 6; 2; 1;;],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (0, 1),
            (0, 4),
            (0, 5),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
        ]
        @test interfaces == [
            (2, 4),
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
        @test Topology.get_interfaces_on_boundary(topology) == [
            (1, 4),
            (1, 5),
            (1, 8),
            (1, 13),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
        ]
    end

    @testset "structured grid: 2 × 2 × 2" verbose=true begin
        topology = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX, Val(8))

        @test Topology.get_manifold_dim(topology) == 3
        @test Topology.get_incidence_relations_dim(topology) == 4
        @test Topology.get_num_patches(topology) == 8
        @test Topology.get_patch_type(topology) == typeof(Topology.HEX)
        @test Topology.get_topological_patch(topology) == Topology.HEX
        # 27 vertices, 54 edges, 36 faces and 8 patches; Euler: 27 - 54 + 36 - 8 == 1.
        @test size(topology) == (27, 54, 36, 8)
        @test @expected Dict(1 => 27, 2 => 54, 3 => 36, 4 => 8) k -> size(topology, k)
        @test size(topology, 1) - size(topology, 2) + size(topology, 3) -
              size(topology, 4) == 1
        @test Topology.get_local_size(topology) == (8, 12, 6, 1)
        expected_local_size_dim = Dict(1 => 8, 2 => 12, 3 => 6, 4 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 4

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        # 6 sides of 4 faces each are on the boundary; the remaining 12 are shared.
        @test count(object -> object[1] == 2, boundaries) == 24
        @test count(object -> object[1] == 2, interfaces) == 12
        # Only the 8 corner vertices belong to a single patch.
        @test count(object -> object[1] == 0, boundaries) == 8

        # Every patch of a 2x2x2 grid is a corner patch: 3 shared faces and 7 patches that
        # share the centre vertex.
        face_neighbour_counts = Dict(patch_id => 3 for patch_id in 1:8)
        @test @expected face_neighbour_counts patch_id -> sum(
            face -> size(Topology.compute_neighbours(topology, patch_id, face, 2), 2), 1:6
        )
        # Every shared face is traversed in opposite directions by the two patches that
        # share it, since all patches of the grid are consistently oriented.
        face_orientations = Dict(
            (patch_id, face) => true for patch_id in 1:8, face in 1:6
        )
        @test @expected face_orientations k -> all(
            ==(-1), Topology.compute_neighbours(topology, k[1], k[2], 2)[4, :]
        )

        # The centre vertex is local vertex 7 of the first patch.
        @test size(Topology.compute_neighbours(topology, 1, 7, 0), 2) == 7
        @test size(Topology.compute_neighbours(topology, 1, 7, 0, true), 2) == 8
    end
end

############################################################################################
#                                     Argument errors                                      #
############################################################################################
@testset "argument errors" begin
    line_topology = Topology.MeshTopology(line_grid(3), Topology.LINE, Val(3))
    quad_topology = Topology.MeshTopology(quad_grid(2, 2), Topology.QUAD, Val(4))
    hex_topology = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX, Val(8))

    @testset "object dimensions outside the valid range" begin
        # Objects of the manifold dimension are the patches themselves, and are never
        # shared; higher dimensions do not exist.
        @test_throws ArgumentError Topology.compute_neighbours(line_topology, 1, 1, 1)
        @test_throws ArgumentError Topology.compute_neighbours(quad_topology, 1, 1, 2)
        @test_throws ArgumentError Topology.compute_neighbours(hex_topology, 1, 1, 3)
        @test_throws ArgumentError Topology.compute_neighbours(hex_topology, 1, 1, 4)
        @test_throws ArgumentError Topology.compute_neighbours(quad_topology, 1, 1, -1)
        @test_throws ArgumentError Topology.compute_neighbours(hex_topology, 4)

        # Edges are only shared from 2D on, faces only in 3D.
        @test_throws ArgumentError Topology.compute_neighbours(line_topology, 1, 1, 1)
        @test_throws ArgumentError Topology.compute_neighbours(line_topology, 1)
        @test_throws ArgumentError Topology.compute_neighbours(quad_topology, 1, 1, 2)
        @test_throws ArgumentError Topology.compute_neighbours(quad_topology, 2)
    end

    @testset "unknown ids" begin
        # A container cannot contain objects of its own dimension.
        @test_throws ArgumentError Topology.get_global_id(quad_topology, 1, 2, 1, 2)
        # Patch 1 of a 2x2 grid does not contain vertex 9.
        @test_throws ArgumentError Topology.get_local_id(quad_topology, 1, 9, 0)
        # A patch does not contain itself as a local object.
        @test_throws ArgumentError Topology.get_local_id(quad_topology, 1, 1, 2)
        @test_throws ArgumentError Topology.get_local_id(hex_topology, 1, 1, 3)
        @test_throws BoundsError Topology.get_global_id(quad_topology, 1, 5, 0)
    end

    @testset "index bounds" begin
        @test_throws BoundsError quad_topology[0, 1]
        @test_throws BoundsError quad_topology[4, 1]
        @test_throws BoundsError quad_topology[1, 4]
        @test_throws BoundsError size(quad_topology, 4)
    end
end

end

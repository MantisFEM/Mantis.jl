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

"""
    reconstruct_vertices(reference_vertices, rotation, orientation)

Return the vertex sequence obtained by traversing `reference_vertices` starting at
`rotation` (0-based) and stepping in the direction given by `orientation`.

A neighbour's own vertex sequence for a shared object must be reproduced exactly this way
from the reference sequence and the reported rotation and orientation; this is the defining
property of the two values.
"""
function reconstruct_vertices(reference_vertices, rotation, orientation)
    n = length(reference_vertices)
    # For a two-vertex object the reported rotation is always 0 and the shift is folded into
    # the orientation, because reversing a two-cycle and shifting it by one coincide.
    if n == 2
        return orientation == 1 ? collect(reference_vertices) : reverse(reference_vertices)
    end
    return [reference_vertices[mod1(rotation + 1 + orientation * (m - 1), n)] for m in 1:n]
end


############################################################################################
#                                   Construction errors                                    #
############################################################################################
@testset "construction errors" begin
    # The skeleton of a 1D mesh would be a set of points, which is not a topology.
    @test_throws ArgumentError Topology.SkeletonTopology(
        Topology.MeshTopology(line_grid(3), Topology.LINE, Val(3))
    )
    @test_throws ArgumentError Topology.SkeletonTopology(
        Topology.MeshTopology([(1, 2)], Topology.LINE, Val(1))
    )
end

############################################################################################
#                                            2D                                            #
############################################################################################
@testset "2D" verbose=true begin
    @testset "single patch" verbose=true begin
        parent = Topology.MeshTopology(((1, 2, 3, 4),), Topology.QUAD)
        topology = Topology.SkeletonTopology(parent)

        @test Topology.get_parent_topology(topology) === parent
        @test topology == Topology.SkeletonTopology(parent)
        @test hash(topology) == hash(Topology.SkeletonTopology(parent))
        # A skeleton and its parent are different topologies.
        @test topology != parent

        @test Topology.get_manifold_dim(topology) == 1
        @test Topology.get_incidence_relations_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 4
        @test Topology.get_patch_type(topology) == typeof(Topology.LINE)
        @test Topology.get_topological_patch(topology) == Topology.LINE
        @test size(topology) == (4, 4)
        @test @expected Dict(1 => 4, 2 => 4) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (2, 1)
        expected_local_size_dim = Dict(1 => 2, 2 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 2

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2],
                [1, 3],
                [3, 4],
                [2, 4],
            ],
            (2, 1) => [
                [1, 2],
                [1, 4],
                [2, 3],
                [4, 3],
            ],
            (2, 2) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        # The skeleton shares the incidence relations of its parent.
        @test all(
            topology[i, k] === parent[i, k] for i in 1:2,
            k in 1:2
        )

        expected_global_ids = Dict{NTuple{4, Int}, Int}(
            # (container_id, container_dim, local_id, local_dim)
            # topology[container_dim + 1, local_dim + 1][container_id][local_id]
            # topology[1, 2]
            (1, 0, 1, 1) => 1,
            (1, 0, 2, 1) => 2,
            (2, 0, 1, 1) => 1,
            (2, 0, 2, 1) => 3,
            (3, 0, 1, 1) => 3,
            (3, 0, 2, 1) => 4,
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 4,
            # topology[2, 1]
            (1, 1, 1, 0) => 1,
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 1,
            (2, 1, 2, 0) => 4,
            (3, 1, 1, 0) => 2,
            (3, 1, 2, 0) => 3,
            (4, 1, 1, 0) => 4,
            (4, 1, 2, 0) => 3,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 1, 0) => 1,
            (2, 2, 0) => 4,
            (3, 1, 0) => 2,
            (3, 2, 0) => 3,
            (4, 1, 0) => 4,
            (4, 2, 0) => 3,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 1, 0) => 1,
            (2, 4, 0) => 2,
            (3, 2, 0) => 1,
            (3, 3, 0) => 2,
            (4, 4, 0) => 1,
            (4, 3, 0) => 2,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => [2; 1; 0; 1;;],
            (1, 2, 0, false) => [3; 1; 0; 1;;],
            (2, 1, 0, false) => [1; 1; 0; 1;;],
            (2, 2, 0, false) => [4; 1; 0; 1;;],
            (3, 1, 0, false) => [1; 2; 0; 1;;],
            (3, 2, 0, false) => [4; 2; 0; 1;;],
            (4, 1, 0, false) => [2; 2; 0; 1;;],
            (4, 2, 0, false) => [3; 2; 0; 1;;],
            (1, 1, 0, true) => [1 2; 1 1; 0 0; 1 1],
            (1, 2, 0, true) => [1 3; 2 1; 0 0; 1 1],
            (2, 1, 0, true) => [1 2; 1 1; 0 0; 1 1],
            (2, 2, 0, true) => [2 4; 2 1; 0 0; 1 1],
            (3, 1, 0, true) => [1 3; 2 1; 0 0; 1 1],
            (3, 2, 0, true) => [3 4; 2 2; 0 0; 1 1],
            (4, 1, 0, true) => [2 4; 2 1; 0 0; 1 1],
            (4, 2, 0, true) => [3 4; 2 2; 0 0; 1 1],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test isempty(boundaries)
        @test interfaces == [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
        ]
        @test isempty(Topology.get_interfaces_on_boundary(topology))

        expected_patch_parents = Dict{Int, Matrix{Int}}(
            1 => [1; 3; 0; 1;;],
            2 => [1; 1; 0; 1;;],
            3 => [1; 2; 0; 1;;],
            4 => [1; 4; 0; 1;;],
        )
        @test @expected expected_patch_parents k -> Topology.get_patch_parents(topology, k)

        # The rotation and orientation reproduce the vertices of the skeleton patch as
        # traversed by each of its parent patches.
        expected_parent_vertices = Dict{Tuple{Int, Int}, Vector{Int}}(
            (1, 1) => [1, 2],
            (2, 1) => [1, 4],
            (3, 1) => [2, 3],
            (4, 1) => [4, 3],
        )
        @test @expected expected_parent_vertices k -> begin
            parents = Topology.get_patch_parents(topology, k[1])
            reconstruct_vertices(
                Topology.get_parent_topology(topology)[2, 1][k[1]],
                parents[3, k[2]],
                parents[4, k[2]],
            )
        end

    end

    @testset "two patches" verbose=true begin
        parent = Topology.MeshTopology(((1, 2, 3, 4), (2, 5, 6, 3)), Topology.QUAD)
        topology = Topology.SkeletonTopology(parent)

        @test Topology.get_parent_topology(topology) === parent
        @test topology == Topology.SkeletonTopology(parent)
        @test hash(topology) == hash(Topology.SkeletonTopology(parent))
        # A skeleton and its parent are different topologies.
        @test topology != parent

        @test Topology.get_manifold_dim(topology) == 1
        @test Topology.get_incidence_relations_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 7
        @test Topology.get_patch_type(topology) == typeof(Topology.LINE)
        @test Topology.get_topological_patch(topology) == Topology.LINE
        @test size(topology) == (6, 7)
        @test @expected Dict(1 => 6, 2 => 7) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (2, 1)
        expected_local_size_dim = Dict(1 => 2, 2 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 2

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2],
                [1, 3, 4],
                [3, 5, 6],
                [2, 5],
                [4, 7],
                [6, 7],
            ],
            (2, 1) => [
                [1, 2],
                [1, 4],
                [2, 3],
                [2, 5],
                [4, 3],
                [3, 6],
                [5, 6],
            ],
            (2, 2) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        # The skeleton shares the incidence relations of its parent.
        @test all(
            topology[i, k] === parent[i, k] for i in 1:2,
            k in 1:2
        )

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
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 5,
            (5, 0, 1, 1) => 4,
            (5, 0, 2, 1) => 7,
            (6, 0, 1, 1) => 6,
            (6, 0, 2, 1) => 7,
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
            (7, 1, 1, 0) => 5,
            (7, 1, 2, 0) => 6,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 1, 0) => 1,
            (2, 2, 0) => 4,
            (3, 1, 0) => 2,
            (3, 2, 0) => 3,
            (4, 1, 0) => 2,
            (4, 2, 0) => 5,
            (5, 1, 0) => 4,
            (5, 2, 0) => 3,
            (6, 1, 0) => 3,
            (6, 2, 0) => 6,
            (7, 1, 0) => 5,
            (7, 2, 0) => 6,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 1, 0) => 1,
            (2, 4, 0) => 2,
            (3, 2, 0) => 1,
            (3, 3, 0) => 2,
            (4, 2, 0) => 1,
            (4, 5, 0) => 2,
            (5, 4, 0) => 1,
            (5, 3, 0) => 2,
            (6, 3, 0) => 1,
            (6, 6, 0) => 2,
            (7, 5, 0) => 1,
            (7, 6, 0) => 2,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => [2; 1; 0; 1;;],
            (1, 2, 0, false) => [3 4; 1 1; 0 0; 1 1],
            (2, 1, 0, false) => [1; 1; 0; 1;;],
            (2, 2, 0, false) => [5; 1; 0; 1;;],
            (3, 1, 0, false) => [1 4; 2 1; 0 0; 1 1],
            (3, 2, 0, false) => [5 6; 2 1; 0 0; 1 1],
            (4, 1, 0, false) => [1 3; 2 1; 0 0; 1 1],
            (4, 2, 0, false) => [7; 1; 0; 1;;],
            (5, 1, 0, false) => [2; 2; 0; 1;;],
            (5, 2, 0, false) => [3 6; 2 1; 0 0; 1 1],
            (6, 1, 0, false) => [3 5; 2 2; 0 0; 1 1],
            (6, 2, 0, false) => [7; 2; 0; 1;;],
            (7, 1, 0, false) => [4; 2; 0; 1;;],
            (7, 2, 0, false) => [6; 2; 0; 1;;],
            (1, 1, 0, true) => [1 2; 1 1; 0 0; 1 1],
            (1, 2, 0, true) => [1 3 4; 2 1 1; 0 0 0; 1 1 1],
            (2, 1, 0, true) => [1 2; 1 1; 0 0; 1 1],
            (2, 2, 0, true) => [2 5; 2 1; 0 0; 1 1],
            (3, 1, 0, true) => [1 3 4; 2 1 1; 0 0 0; 1 1 1],
            (3, 2, 0, true) => [3 5 6; 2 2 1; 0 0 0; 1 1 1],
            (4, 1, 0, true) => [1 3 4; 2 1 1; 0 0 0; 1 1 1],
            (4, 2, 0, true) => [4 7; 2 1; 0 0; 1 1],
            (5, 1, 0, true) => [2 5; 2 1; 0 0; 1 1],
            (5, 2, 0, true) => [3 5 6; 2 2 1; 0 0 0; 1 1 1],
            (6, 1, 0, true) => [3 5 6; 2 2 1; 0 0 0; 1 1 1],
            (6, 2, 0, true) => [6 7; 2 2; 0 0; 1 1],
            (7, 1, 0, true) => [4 7; 2 1; 0 0; 1 1],
            (7, 2, 0, true) => [6 7; 2 2; 0 0; 1 1],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test isempty(boundaries)
        @test interfaces == [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
        ]
        @test isempty(Topology.get_interfaces_on_boundary(topology))

        expected_patch_parents = Dict{Int, Matrix{Int}}(
            1 => [1; 3; 0; 1;;],
            2 => [1; 1; 0; 1;;],
            3 => [1 2; 2 1; 0 0; 1 1],
            4 => [2; 3; 0; 1;;],
            5 => [1; 4; 0; 1;;],
            6 => [2; 4; 0; 1;;],
            7 => [2; 2; 0; 1;;],
        )
        @test @expected expected_patch_parents k -> Topology.get_patch_parents(topology, k)

        # The rotation and orientation reproduce the vertices of the skeleton patch as
        # traversed by each of its parent patches.
        expected_parent_vertices = Dict{Tuple{Int, Int}, Vector{Int}}(
            (1, 1) => [1, 2],
            (2, 1) => [1, 4],
            (3, 1) => [2, 3],
            (3, 2) => [2, 3],
            (4, 1) => [2, 5],
            (5, 1) => [4, 3],
            (6, 1) => [3, 6],
            (7, 1) => [5, 6],
        )
        @test @expected expected_parent_vertices k -> begin
            parents = Topology.get_patch_parents(topology, k[1])
            reconstruct_vertices(
                Topology.get_parent_topology(topology)[2, 1][k[1]],
                parents[3, k[2]],
                parents[4, k[2]],
            )
        end

    end

    @testset "structured grid: 2 × 2" verbose=true begin
        parent = Topology.MeshTopology(quad_grid(2, 2), Topology.QUAD, Val(4))
        topology = Topology.SkeletonTopology(parent)

        @test Topology.get_parent_topology(topology) === parent
        @test topology == Topology.SkeletonTopology(parent)
        @test hash(topology) == hash(Topology.SkeletonTopology(parent))
        # A skeleton and its parent are different topologies.
        @test topology != parent

        @test Topology.get_manifold_dim(topology) == 1
        @test Topology.get_incidence_relations_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 12
        @test Topology.get_patch_type(topology) == typeof(Topology.LINE)
        @test Topology.get_topological_patch(topology) == Topology.LINE
        @test size(topology) == (9, 12)
        @test @expected Dict(1 => 9, 2 => 12) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (2, 1)
        expected_local_size_dim = Dict(1 => 2, 2 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 2

        expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
            (1, 1) => Vector{Vector{Int}}(),
            (1, 2) => [
                [1, 2],
                [1, 3, 4],
                [3, 5],
                [2, 6, 7],
                [4, 6, 8, 9],
                [5, 8, 10],
                [7, 11],
                [9, 11, 12],
                [10, 12],
            ],
            (2, 1) => [
                [1, 2],
                [1, 4],
                [2, 3],
                [2, 5],
                [3, 6],
                [4, 5],
                [4, 7],
                [5, 6],
                [5, 8],
                [6, 9],
                [7, 8],
                [8, 9],
            ],
            (2, 2) => Vector{Vector{Int}}(),
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        # The skeleton shares the incidence relations of its parent.
        @test all(
            topology[i, k] === parent[i, k] for i in 1:2,
            k in 1:2
        )

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
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 6,
            (4, 0, 3, 1) => 7,
            (5, 0, 1, 1) => 4,
            (5, 0, 2, 1) => 6,
            (5, 0, 3, 1) => 8,
            (5, 0, 4, 1) => 9,
            (6, 0, 1, 1) => 5,
            (6, 0, 2, 1) => 8,
            (6, 0, 3, 1) => 10,
            (7, 0, 1, 1) => 7,
            (7, 0, 2, 1) => 11,
            (8, 0, 1, 1) => 9,
            (8, 0, 2, 1) => 11,
            (8, 0, 3, 1) => 12,
            (9, 0, 1, 1) => 10,
            (9, 0, 2, 1) => 12,
            # topology[2, 1]
            (1, 1, 1, 0) => 1,
            (1, 1, 2, 0) => 2,
            (2, 1, 1, 0) => 1,
            (2, 1, 2, 0) => 4,
            (3, 1, 1, 0) => 2,
            (3, 1, 2, 0) => 3,
            (4, 1, 1, 0) => 2,
            (4, 1, 2, 0) => 5,
            (5, 1, 1, 0) => 3,
            (5, 1, 2, 0) => 6,
            (6, 1, 1, 0) => 4,
            (6, 1, 2, 0) => 5,
            (7, 1, 1, 0) => 4,
            (7, 1, 2, 0) => 7,
            (8, 1, 1, 0) => 5,
            (8, 1, 2, 0) => 6,
            (9, 1, 1, 0) => 5,
            (9, 1, 2, 0) => 8,
            (10, 1, 1, 0) => 6,
            (10, 1, 2, 0) => 9,
            (11, 1, 1, 0) => 7,
            (11, 1, 2, 0) => 8,
            (12, 1, 1, 0) => 8,
            (12, 1, 2, 0) => 9,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 1, 0) => 1,
            (2, 2, 0) => 4,
            (3, 1, 0) => 2,
            (3, 2, 0) => 3,
            (4, 1, 0) => 2,
            (4, 2, 0) => 5,
            (5, 1, 0) => 3,
            (5, 2, 0) => 6,
            (6, 1, 0) => 4,
            (6, 2, 0) => 5,
            (7, 1, 0) => 4,
            (7, 2, 0) => 7,
            (8, 1, 0) => 5,
            (8, 2, 0) => 6,
            (9, 1, 0) => 5,
            (9, 2, 0) => 8,
            (10, 1, 0) => 6,
            (10, 2, 0) => 9,
            (11, 1, 0) => 7,
            (11, 2, 0) => 8,
            (12, 1, 0) => 8,
            (12, 2, 0) => 9,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 1,
            (1, 2, 0) => 2,
            (2, 1, 0) => 1,
            (2, 4, 0) => 2,
            (3, 2, 0) => 1,
            (3, 3, 0) => 2,
            (4, 2, 0) => 1,
            (4, 5, 0) => 2,
            (5, 3, 0) => 1,
            (5, 6, 0) => 2,
            (6, 4, 0) => 1,
            (6, 5, 0) => 2,
            (7, 4, 0) => 1,
            (7, 7, 0) => 2,
            (8, 5, 0) => 1,
            (8, 6, 0) => 2,
            (9, 5, 0) => 1,
            (9, 8, 0) => 2,
            (10, 6, 0) => 1,
            (10, 9, 0) => 2,
            (11, 7, 0) => 1,
            (11, 8, 0) => 2,
            (12, 8, 0) => 1,
            (12, 9, 0) => 2,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => [2; 1; 0; 1;;],
            (1, 2, 0, false) => [3 4; 1 1; 0 0; 1 1],
            (2, 1, 0, false) => [1; 1; 0; 1;;],
            (2, 2, 0, false) => [6 7; 1 1; 0 0; 1 1],
            (3, 1, 0, false) => [1 4; 2 1; 0 0; 1 1],
            (3, 2, 0, false) => [5; 1; 0; 1;;],
            (4, 1, 0, false) => [1 3; 2 1; 0 0; 1 1],
            (4, 2, 0, false) => [6 8 9; 2 1 1; 0 0 0; 1 1 1],
            (5, 1, 0, false) => [3; 2; 0; 1;;],
            (5, 2, 0, false) => [8 10; 2 1; 0 0; 1 1],
            (6, 1, 0, false) => [2 7; 2 1; 0 0; 1 1],
            (6, 2, 0, false) => [4 8 9; 2 1 1; 0 0 0; 1 1 1],
            (7, 1, 0, false) => [2 6; 2 1; 0 0; 1 1],
            (7, 2, 0, false) => [11; 1; 0; 1;;],
            (8, 1, 0, false) => [4 6 9; 2 2 1; 0 0 0; 1 1 1],
            (8, 2, 0, false) => [5 10; 2 1; 0 0; 1 1],
            (9, 1, 0, false) => [4 6 8; 2 2 1; 0 0 0; 1 1 1],
            (9, 2, 0, false) => [11 12; 2 1; 0 0; 1 1],
            (10, 1, 0, false) => [5 8; 2 2; 0 0; 1 1],
            (10, 2, 0, false) => [12; 2; 0; 1;;],
            (11, 1, 0, false) => [7; 2; 0; 1;;],
            (11, 2, 0, false) => [9 12; 2 1; 0 0; 1 1],
            (12, 1, 0, false) => [9 11; 2 2; 0 0; 1 1],
            (12, 2, 0, false) => [10; 2; 0; 1;;],
            (1, 1, 0, true) => [1 2; 1 1; 0 0; 1 1],
            (1, 2, 0, true) => [1 3 4; 2 1 1; 0 0 0; 1 1 1],
            (2, 1, 0, true) => [1 2; 1 1; 0 0; 1 1],
            (2, 2, 0, true) => [2 6 7; 2 1 1; 0 0 0; 1 1 1],
            (3, 1, 0, true) => [1 3 4; 2 1 1; 0 0 0; 1 1 1],
            (3, 2, 0, true) => [3 5; 2 1; 0 0; 1 1],
            (4, 1, 0, true) => [1 3 4; 2 1 1; 0 0 0; 1 1 1],
            (4, 2, 0, true) => [4 6 8 9; 2 2 1 1; 0 0 0 0; 1 1 1 1],
            (5, 1, 0, true) => [3 5; 2 1; 0 0; 1 1],
            (5, 2, 0, true) => [5 8 10; 2 2 1; 0 0 0; 1 1 1],
            (6, 1, 0, true) => [2 6 7; 2 1 1; 0 0 0; 1 1 1],
            (6, 2, 0, true) => [4 6 8 9; 2 2 1 1; 0 0 0 0; 1 1 1 1],
            (7, 1, 0, true) => [2 6 7; 2 1 1; 0 0 0; 1 1 1],
            (7, 2, 0, true) => [7 11; 2 1; 0 0; 1 1],
            (8, 1, 0, true) => [4 6 8 9; 2 2 1 1; 0 0 0 0; 1 1 1 1],
            (8, 2, 0, true) => [5 8 10; 2 2 1; 0 0 0; 1 1 1],
            (9, 1, 0, true) => [4 6 8 9; 2 2 1 1; 0 0 0 0; 1 1 1 1],
            (9, 2, 0, true) => [9 11 12; 2 2 1; 0 0 0; 1 1 1],
            (10, 1, 0, true) => [5 8 10; 2 2 1; 0 0 0; 1 1 1],
            (10, 2, 0, true) => [10 12; 2 2; 0 0; 1 1],
            (11, 1, 0, true) => [7 11; 2 1; 0 0; 1 1],
            (11, 2, 0, true) => [9 11 12; 2 2 1; 0 0 0; 1 1 1],
            (12, 1, 0, true) => [9 11 12; 2 2 1; 0 0 0; 1 1 1],
            (12, 2, 0, true) => [10 12; 2 2; 0 0; 1 1],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test isempty(boundaries)
        @test interfaces == [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
        ]
        @test isempty(Topology.get_interfaces_on_boundary(topology))

        expected_patch_parents = Dict{Int, Matrix{Int}}(
            1 => [1; 3; 0; 1;;],
            2 => [1; 1; 0; 1;;],
            3 => [2; 3; 0; 1;;],
            4 => [1 2; 2 1; 0 0; 1 1],
            5 => [2; 2; 0; 1;;],
            6 => [1 3; 4 3; 0 0; 1 1],
            7 => [3; 1; 0; 1;;],
            8 => [2 4; 4 3; 0 0; 1 1],
            9 => [3 4; 2 1; 0 0; 1 1],
            10 => [4; 2; 0; 1;;],
            11 => [3; 4; 0; 1;;],
            12 => [4; 4; 0; 1;;],
        )
        @test @expected expected_patch_parents k -> Topology.get_patch_parents(topology, k)

        # The rotation and orientation reproduce the vertices of the skeleton patch as
        # traversed by each of its parent patches.
        expected_parent_vertices = Dict{Tuple{Int, Int}, Vector{Int}}(
            (1, 1) => [1, 2],
            (2, 1) => [1, 4],
            (3, 1) => [2, 3],
            (4, 1) => [2, 5],
            (4, 2) => [2, 5],
            (5, 1) => [3, 6],
            (6, 1) => [4, 5],
            (6, 2) => [4, 5],
            (7, 1) => [4, 7],
            (8, 1) => [5, 6],
            (8, 2) => [5, 6],
            (9, 1) => [5, 8],
            (9, 2) => [5, 8],
            (10, 1) => [6, 9],
            (11, 1) => [7, 8],
            (12, 1) => [8, 9],
        )
        @test @expected expected_parent_vertices k -> begin
            parents = Topology.get_patch_parents(topology, k[1])
            reconstruct_vertices(
                Topology.get_parent_topology(topology)[2, 1][k[1]],
                parents[3, k[2]],
                parents[4, k[2]],
            )
        end

    end
end

############################################################################################
#                                            3D                                            #
############################################################################################
@testset "3D" verbose=true begin
    @testset "single patch" verbose=true begin
        parent = Topology.MeshTopology(((1, 2, 3, 4, 5, 6, 7, 8),), Topology.HEX)
        topology = Topology.SkeletonTopology(parent)

        @test Topology.get_parent_topology(topology) === parent
        @test topology == Topology.SkeletonTopology(parent)
        @test hash(topology) == hash(Topology.SkeletonTopology(parent))
        # A skeleton and its parent are different topologies.
        @test topology != parent

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 6
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (8, 12, 6)
        @test @expected Dict(1 => 8, 2 => 12, 3 => 6) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (4, 4, 1)
        expected_local_size_dim = Dict(1 => 4, 2 => 4, 3 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 3

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
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        # The skeleton shares the incidence relations of its parent.
        @test all(
            topology[i, k] === parent[i, k] for i in 1:3,
            k in 1:3
        )

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
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 3,
            (1, 2, 0) => 2,
            (1, 3, 0) => 1,
            (1, 4, 0) => 4,
            (2, 1, 0) => 6,
            (2, 2, 0) => 5,
            (2, 3, 0) => 1,
            (2, 4, 0) => 2,
            (3, 1, 0) => 8,
            (3, 2, 0) => 4,
            (3, 3, 0) => 1,
            (3, 4, 0) => 5,
            (4, 1, 0) => 7,
            (4, 2, 0) => 6,
            (4, 3, 0) => 2,
            (4, 4, 0) => 3,
            (5, 1, 0) => 7,
            (5, 2, 0) => 3,
            (5, 3, 0) => 4,
            (5, 4, 0) => 8,
            (6, 1, 0) => 7,
            (6, 2, 0) => 8,
            (6, 3, 0) => 5,
            (6, 4, 0) => 6,
            (1, 1, 1) => 6,
            (1, 2, 1) => 1,
            (1, 3, 1) => 4,
            (1, 4, 1) => 2,
            (2, 1, 1) => 5,
            (2, 2, 1) => 3,
            (2, 3, 1) => 9,
            (2, 4, 1) => 1,
            (3, 1, 1) => 10,
            (3, 2, 1) => 2,
            (3, 3, 1) => 8,
            (3, 4, 1) => 3,
            (4, 1, 1) => 7,
            (4, 2, 1) => 5,
            (4, 3, 1) => 11,
            (4, 4, 1) => 4,
            (5, 1, 1) => 12,
            (5, 2, 1) => 6,
            (5, 3, 1) => 7,
            (5, 4, 1) => 8,
            (6, 1, 1) => 11,
            (6, 2, 1) => 10,
            (6, 3, 1) => 12,
            (6, 4, 1) => 9,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 3, 0) => 1,
            (1, 2, 0) => 2,
            (1, 1, 0) => 3,
            (1, 4, 0) => 4,
            (2, 6, 0) => 1,
            (2, 5, 0) => 2,
            (2, 1, 0) => 3,
            (2, 2, 0) => 4,
            (3, 8, 0) => 1,
            (3, 4, 0) => 2,
            (3, 1, 0) => 3,
            (3, 5, 0) => 4,
            (4, 7, 0) => 1,
            (4, 6, 0) => 2,
            (4, 2, 0) => 3,
            (4, 3, 0) => 4,
            (5, 7, 0) => 1,
            (5, 3, 0) => 2,
            (5, 4, 0) => 3,
            (5, 8, 0) => 4,
            (6, 7, 0) => 1,
            (6, 8, 0) => 2,
            (6, 5, 0) => 3,
            (6, 6, 0) => 4,
            (1, 6, 1) => 1,
            (1, 1, 1) => 2,
            (1, 4, 1) => 3,
            (1, 2, 1) => 4,
            (2, 5, 1) => 1,
            (2, 3, 1) => 2,
            (2, 9, 1) => 3,
            (2, 1, 1) => 4,
            (3, 10, 1) => 1,
            (3, 2, 1) => 2,
            (3, 8, 1) => 3,
            (3, 3, 1) => 4,
            (4, 7, 1) => 1,
            (4, 5, 1) => 2,
            (4, 11, 1) => 3,
            (4, 4, 1) => 4,
            (5, 12, 1) => 1,
            (5, 6, 1) => 2,
            (5, 7, 1) => 3,
            (5, 8, 1) => 4,
            (6, 11, 1) => 1,
            (6, 10, 1) => 2,
            (6, 12, 1) => 3,
            (6, 9, 1) => 4,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => [4 5; 4 2; 0 0; 1 1],
            (1, 2, 0, false) => [2 4; 4 3; 0 0; 1 1],
            (1, 3, 0, false) => [2 3; 3 3; 0 0; 1 1],
            (1, 4, 0, false) => [3 5; 2 3; 0 0; 1 1],
            (2, 1, 0, false) => [4 6; 2 4; 0 0; 1 1],
            (2, 2, 0, false) => [3 6; 4 3; 0 0; 1 1],
            (2, 3, 0, false) => [1 3; 3 3; 0 0; 1 1],
            (2, 4, 0, false) => [1 4; 2 3; 0 0; 1 1],
            (3, 1, 0, false) => [5 6; 4 2; 0 0; 1 1],
            (3, 2, 0, false) => [1 5; 4 3; 0 0; 1 1],
            (3, 3, 0, false) => [1 2; 3 3; 0 0; 1 1],
            (3, 4, 0, false) => [2 6; 2 3; 0 0; 1 1],
            (4, 1, 0, false) => [5 6; 1 1; 0 0; 1 1],
            (4, 2, 0, false) => [2 6; 1 4; 0 0; 1 1],
            (4, 3, 0, false) => [1 2; 2 4; 0 0; 1 1],
            (4, 4, 0, false) => [1 5; 1 2; 0 0; 1 1],
            (5, 1, 0, false) => [4 6; 1 1; 0 0; 1 1],
            (5, 2, 0, false) => [1 4; 1 4; 0 0; 1 1],
            (5, 3, 0, false) => [1 3; 4 2; 0 0; 1 1],
            (5, 4, 0, false) => [3 6; 1 2; 0 0; 1 1],
            (6, 1, 0, false) => [4 5; 1 1; 0 0; 1 1],
            (6, 2, 0, false) => [3 5; 1 4; 0 0; 1 1],
            (6, 3, 0, false) => [2 3; 2 4; 0 0; 1 1],
            (6, 4, 0, false) => [2 4; 1 2; 0 0; 1 1],
            (1, 1, 0, true) => [1 4 5; 1 4 2; 0 0 0; 1 1 1],
            (1, 2, 0, true) => [1 2 4; 2 4 3; 0 0 0; 1 1 1],
            (1, 3, 0, true) => [1 2 3; 3 3 3; 0 0 0; 1 1 1],
            (1, 4, 0, true) => [1 3 5; 4 2 3; 0 0 0; 1 1 1],
            (2, 1, 0, true) => [2 4 6; 1 2 4; 0 0 0; 1 1 1],
            (2, 2, 0, true) => [2 3 6; 2 4 3; 0 0 0; 1 1 1],
            (2, 3, 0, true) => [1 2 3; 3 3 3; 0 0 0; 1 1 1],
            (2, 4, 0, true) => [1 2 4; 2 4 3; 0 0 0; 1 1 1],
            (3, 1, 0, true) => [3 5 6; 1 4 2; 0 0 0; 1 1 1],
            (3, 2, 0, true) => [1 3 5; 4 2 3; 0 0 0; 1 1 1],
            (3, 3, 0, true) => [1 2 3; 3 3 3; 0 0 0; 1 1 1],
            (3, 4, 0, true) => [2 3 6; 2 4 3; 0 0 0; 1 1 1],
            (4, 1, 0, true) => [4 5 6; 1 1 1; 0 0 0; 1 1 1],
            (4, 2, 0, true) => [2 4 6; 1 2 4; 0 0 0; 1 1 1],
            (4, 3, 0, true) => [1 2 4; 2 4 3; 0 0 0; 1 1 1],
            (4, 4, 0, true) => [1 4 5; 1 4 2; 0 0 0; 1 1 1],
            (5, 1, 0, true) => [4 5 6; 1 1 1; 0 0 0; 1 1 1],
            (5, 2, 0, true) => [1 4 5; 1 4 2; 0 0 0; 1 1 1],
            (5, 3, 0, true) => [1 3 5; 4 2 3; 0 0 0; 1 1 1],
            (5, 4, 0, true) => [3 5 6; 1 4 2; 0 0 0; 1 1 1],
            (6, 1, 0, true) => [4 5 6; 1 1 1; 0 0 0; 1 1 1],
            (6, 2, 0, true) => [3 5 6; 1 4 2; 0 0 0; 1 1 1],
            (6, 3, 0, true) => [2 3 6; 2 4 3; 0 0 0; 1 1 1],
            (6, 4, 0, true) => [2 4 6; 1 2 4; 0 0 0; 1 1 1],
            # Edge neighbours
            (1, 1, 1, false) => [5; 2; 0; 1;;],
            (1, 2, 1, false) => [2; 4; 0; 1;;],
            (1, 3, 1, false) => [4; 4; 0; 1;;],
            (1, 4, 1, false) => [3; 2; 0; 1;;],
            (2, 1, 1, false) => [4; 2; 0; 1;;],
            (2, 2, 1, false) => [3; 4; 0; 1;;],
            (2, 3, 1, false) => [6; 4; 0; 1;;],
            (2, 4, 1, false) => [1; 2; 0; 1;;],
            (3, 1, 1, false) => [6; 2; 0; 1;;],
            (3, 2, 1, false) => [1; 4; 0; 1;;],
            (3, 3, 1, false) => [5; 4; 0; 1;;],
            (3, 4, 1, false) => [2; 2; 0; 1;;],
            (4, 1, 1, false) => [5; 3; 0; 1;;],
            (4, 2, 1, false) => [2; 1; 0; 1;;],
            (4, 3, 1, false) => [6; 1; 0; 1;;],
            (4, 4, 1, false) => [1; 3; 0; 1;;],
            (5, 1, 1, false) => [6; 3; 0; 1;;],
            (5, 2, 1, false) => [1; 1; 0; 1;;],
            (5, 3, 1, false) => [4; 1; 0; 1;;],
            (5, 4, 1, false) => [3; 3; 0; 1;;],
            (6, 1, 1, false) => [4; 3; 0; 1;;],
            (6, 2, 1, false) => [3; 1; 0; 1;;],
            (6, 3, 1, false) => [5; 1; 0; 1;;],
            (6, 4, 1, false) => [2; 3; 0; 1;;],
            (1, 1, 1, true) => [1 5; 1 2; 0 0; 1 1],
            (1, 2, 1, true) => [1 2; 2 4; 0 0; 1 1],
            (1, 3, 1, true) => [1 4; 3 4; 0 0; 1 1],
            (1, 4, 1, true) => [1 3; 4 2; 0 0; 1 1],
            (2, 1, 1, true) => [2 4; 1 2; 0 0; 1 1],
            (2, 2, 1, true) => [2 3; 2 4; 0 0; 1 1],
            (2, 3, 1, true) => [2 6; 3 4; 0 0; 1 1],
            (2, 4, 1, true) => [1 2; 2 4; 0 0; 1 1],
            (3, 1, 1, true) => [3 6; 1 2; 0 0; 1 1],
            (3, 2, 1, true) => [1 3; 4 2; 0 0; 1 1],
            (3, 3, 1, true) => [3 5; 3 4; 0 0; 1 1],
            (3, 4, 1, true) => [2 3; 2 4; 0 0; 1 1],
            (4, 1, 1, true) => [4 5; 1 3; 0 0; 1 1],
            (4, 2, 1, true) => [2 4; 1 2; 0 0; 1 1],
            (4, 3, 1, true) => [4 6; 3 1; 0 0; 1 1],
            (4, 4, 1, true) => [1 4; 3 4; 0 0; 1 1],
            (5, 1, 1, true) => [5 6; 1 3; 0 0; 1 1],
            (5, 2, 1, true) => [1 5; 1 2; 0 0; 1 1],
            (5, 3, 1, true) => [4 5; 1 3; 0 0; 1 1],
            (5, 4, 1, true) => [3 5; 3 4; 0 0; 1 1],
            (6, 1, 1, true) => [4 6; 3 1; 0 0; 1 1],
            (6, 2, 1, true) => [3 6; 1 2; 0 0; 1 1],
            (6, 3, 1, true) => [5 6; 1 3; 0 0; 1 1],
            (6, 4, 1, true) => [2 6; 3 4; 0 0; 1 1],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test isempty(boundaries)
        @test interfaces == [
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
        @test isempty(Topology.get_interfaces_on_boundary(topology))

        expected_patch_parents = Dict{Int, Matrix{Int}}(
            1 => [1; 5; 2; 1;;],
            2 => [1; 3; 2; 1;;],
            3 => [1; 1; 2; 1;;],
            4 => [1; 2; 2; 1;;],
            5 => [1; 4; 2; 1;;],
            6 => [1; 6; 2; 1;;],
        )
        @test @expected expected_patch_parents k -> Topology.get_patch_parents(topology, k)

        # The rotation and orientation reproduce the vertices of the skeleton patch as
        # traversed by each of its parent patches.
        expected_parent_vertices = Dict{Tuple{Int, Int}, Vector{Int}}(
            (1, 1) => [1, 4, 3, 2],
            (2, 1) => [1, 2, 6, 5],
            (3, 1) => [1, 5, 8, 4],
            (4, 1) => [2, 3, 7, 6],
            (5, 1) => [4, 8, 7, 3],
            (6, 1) => [5, 6, 7, 8],
        )
        @test @expected expected_parent_vertices k -> begin
            parents = Topology.get_patch_parents(topology, k[1])
            reconstruct_vertices(
                Topology.get_parent_topology(topology)[3, 1][k[1]],
                parents[3, k[2]],
                parents[4, k[2]],
            )
        end

    end

    # The second patch is glued along local face 2 of the first one, rotated such that the
    # shared face is traversed from a different starting vertex.
    @testset "two patches: face rotation 1 (local face 2)" verbose=true begin
        parent = Topology.MeshTopology(
            ((1, 2, 3, 4, 5, 6, 7, 8), (10, 3, 2, 9, 12, 7, 6, 11)), Topology.HEX
        )
        topology = Topology.SkeletonTopology(parent)

        @test Topology.get_parent_topology(topology) === parent
        @test topology == Topology.SkeletonTopology(parent)
        @test hash(topology) == hash(Topology.SkeletonTopology(parent))
        # A skeleton and its parent are different topologies.
        @test topology != parent

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 11
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (12, 20, 11)
        @test @expected Dict(1 => 12, 2 => 20, 3 => 11) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (4, 4, 1)
        expected_local_size_dim = Dict(1 => 4, 2 => 4, 3 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 3

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
        )
        @test @expected expected_ir k -> topology[k[1], k[2]]

        # The skeleton shares the incidence relations of its parent.
        @test all(
            topology[i, k] === parent[i, k] for i in 1:3,
            k in 1:3
        )

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
            (2, 0, 4, 1) => 6,
            (3, 0, 1, 1) => 4,
            (3, 0, 2, 1) => 7,
            (3, 0, 3, 1) => 8,
            (3, 0, 4, 1) => 9,
            (4, 0, 1, 1) => 2,
            (4, 0, 2, 1) => 7,
            (4, 0, 3, 1) => 10,
            (5, 0, 1, 1) => 3,
            (5, 0, 2, 1) => 11,
            (5, 0, 3, 1) => 12,
            (6, 0, 1, 1) => 5,
            (6, 0, 2, 1) => 11,
            (6, 0, 3, 1) => 13,
            (6, 0, 4, 1) => 14,
            (7, 0, 1, 1) => 8,
            (7, 0, 2, 1) => 13,
            (7, 0, 3, 1) => 15,
            (7, 0, 4, 1) => 16,
            (8, 0, 1, 1) => 10,
            (8, 0, 2, 1) => 12,
            (8, 0, 3, 1) => 15,
            (9, 0, 1, 1) => 6,
            (9, 0, 2, 1) => 17,
            (9, 0, 3, 1) => 18,
            (10, 0, 1, 1) => 9,
            (10, 0, 2, 1) => 17,
            (10, 0, 3, 1) => 19,
            (11, 0, 1, 1) => 14,
            (11, 0, 2, 1) => 18,
            (11, 0, 3, 1) => 20,
            (12, 0, 1, 1) => 16,
            (12, 0, 2, 1) => 19,
            (12, 0, 3, 1) => 20,
            # topology[1, 3]
            (1, 0, 1, 2) => 1,
            (1, 0, 2, 2) => 2,
            (1, 0, 3, 2) => 3,
            (2, 0, 1, 2) => 1,
            (2, 0, 2, 2) => 2,
            (2, 0, 3, 2) => 4,
            (2, 0, 4, 2) => 5,
            (2, 0, 5, 2) => 6,
            (3, 0, 1, 2) => 1,
            (3, 0, 2, 2) => 4,
            (3, 0, 3, 2) => 5,
            (3, 0, 4, 2) => 7,
            (3, 0, 5, 2) => 8,
            (4, 0, 1, 2) => 1,
            (4, 0, 2, 2) => 3,
            (4, 0, 3, 2) => 7,
            (5, 0, 1, 2) => 2,
            (5, 0, 2, 2) => 3,
            (5, 0, 3, 2) => 9,
            (6, 0, 1, 2) => 2,
            (6, 0, 2, 2) => 4,
            (6, 0, 3, 2) => 6,
            (6, 0, 4, 2) => 9,
            (6, 0, 5, 2) => 10,
            (7, 0, 1, 2) => 4,
            (7, 0, 2, 2) => 7,
            (7, 0, 3, 2) => 8,
            (7, 0, 4, 2) => 9,
            (7, 0, 5, 2) => 10,
            (8, 0, 1, 2) => 3,
            (8, 0, 2, 2) => 7,
            (8, 0, 3, 2) => 9,
            (9, 0, 1, 2) => 5,
            (9, 0, 2, 2) => 6,
            (9, 0, 3, 2) => 11,
            (10, 0, 1, 2) => 5,
            (10, 0, 2, 2) => 8,
            (10, 0, 3, 2) => 11,
            (11, 0, 1, 2) => 6,
            (11, 0, 2, 2) => 10,
            (11, 0, 3, 2) => 11,
            (12, 0, 1, 2) => 8,
            (12, 0, 2, 2) => 10,
            (12, 0, 3, 2) => 11,
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
            (6, 1, 1, 0) => 2,
            (6, 1, 2, 0) => 9,
            (7, 1, 1, 0) => 3,
            (7, 1, 2, 0) => 4,
            (8, 1, 1, 0) => 7,
            (8, 1, 2, 0) => 3,
            (9, 1, 1, 0) => 3,
            (9, 1, 2, 0) => 10,
            (10, 1, 1, 0) => 8,
            (10, 1, 2, 0) => 4,
            (11, 1, 1, 0) => 6,
            (11, 1, 2, 0) => 5,
            (12, 1, 1, 0) => 8,
            (12, 1, 2, 0) => 5,
            (13, 1, 1, 0) => 7,
            (13, 1, 2, 0) => 6,
            (14, 1, 1, 0) => 6,
            (14, 1, 2, 0) => 11,
            (15, 1, 1, 0) => 7,
            (15, 1, 2, 0) => 8,
            (16, 1, 1, 0) => 7,
            (16, 1, 2, 0) => 12,
            (17, 1, 1, 0) => 9,
            (17, 1, 2, 0) => 10,
            (18, 1, 1, 0) => 11,
            (18, 1, 2, 0) => 9,
            (19, 1, 1, 0) => 12,
            (19, 1, 2, 0) => 10,
            (20, 1, 1, 0) => 11,
            (20, 1, 2, 0) => 12,
            # topology[2, 3]
            (1, 1, 1, 2) => 1,
            (1, 1, 2, 2) => 2,
            (2, 1, 1, 2) => 1,
            (2, 1, 2, 2) => 3,
            (3, 1, 1, 2) => 2,
            (3, 1, 2, 2) => 3,
            (4, 1, 1, 2) => 1,
            (4, 1, 2, 2) => 4,
            (4, 1, 3, 2) => 5,
            (5, 1, 1, 2) => 2,
            (5, 1, 2, 2) => 4,
            (5, 1, 3, 2) => 6,
            (6, 1, 1, 2) => 5,
            (6, 1, 2, 2) => 6,
            (7, 1, 1, 2) => 1,
            (7, 1, 2, 2) => 7,
            (8, 1, 1, 2) => 4,
            (8, 1, 2, 2) => 7,
            (8, 1, 3, 2) => 8,
            (9, 1, 1, 2) => 5,
            (9, 1, 2, 2) => 8,
            (10, 1, 1, 2) => 3,
            (10, 1, 2, 2) => 7,
            (11, 1, 1, 2) => 2,
            (11, 1, 2, 2) => 9,
            (12, 1, 1, 2) => 3,
            (12, 1, 2, 2) => 9,
            (13, 1, 1, 2) => 4,
            (13, 1, 2, 2) => 9,
            (13, 1, 3, 2) => 10,
            (14, 1, 1, 2) => 6,
            (14, 1, 2, 2) => 10,
            (15, 1, 1, 2) => 7,
            (15, 1, 2, 2) => 9,
            (16, 1, 1, 2) => 8,
            (16, 1, 2, 2) => 10,
            (17, 1, 1, 2) => 5,
            (17, 1, 2, 2) => 11,
            (18, 1, 1, 2) => 6,
            (18, 1, 2, 2) => 11,
            (19, 1, 1, 2) => 8,
            (19, 1, 2, 2) => 11,
            (20, 1, 1, 2) => 10,
            (20, 1, 2, 2) => 11,
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
            (5, 2, 1, 0) => 2,
            (5, 2, 2, 0) => 3,
            (5, 2, 3, 0) => 10,
            (5, 2, 4, 0) => 9,
            (6, 2, 1, 0) => 6,
            (6, 2, 2, 0) => 2,
            (6, 2, 3, 0) => 9,
            (6, 2, 4, 0) => 11,
            (7, 2, 1, 0) => 7,
            (7, 2, 2, 0) => 3,
            (7, 2, 3, 0) => 4,
            (7, 2, 4, 0) => 8,
            (8, 2, 1, 0) => 7,
            (8, 2, 2, 0) => 12,
            (8, 2, 3, 0) => 10,
            (8, 2, 4, 0) => 3,
            (9, 2, 1, 0) => 7,
            (9, 2, 2, 0) => 8,
            (9, 2, 3, 0) => 5,
            (9, 2, 4, 0) => 6,
            (10, 2, 1, 0) => 6,
            (10, 2, 2, 0) => 11,
            (10, 2, 3, 0) => 12,
            (10, 2, 4, 0) => 7,
            (11, 2, 1, 0) => 11,
            (11, 2, 2, 0) => 9,
            (11, 2, 3, 0) => 10,
            (11, 2, 4, 0) => 12,
            # topology[3, 2]
            (1, 2, 1, 1) => 7,
            (1, 2, 2, 1) => 1,
            (1, 2, 3, 1) => 4,
            (1, 2, 4, 1) => 2,
            (2, 2, 1, 1) => 5,
            (2, 2, 2, 1) => 3,
            (2, 2, 3, 1) => 11,
            (2, 2, 4, 1) => 1,
            (3, 2, 1, 1) => 12,
            (3, 2, 2, 1) => 2,
            (3, 2, 3, 1) => 10,
            (3, 2, 4, 1) => 3,
            (4, 2, 1, 1) => 8,
            (4, 2, 2, 1) => 5,
            (4, 2, 3, 1) => 13,
            (4, 2, 4, 1) => 4,
            (5, 2, 1, 1) => 6,
            (5, 2, 2, 1) => 9,
            (5, 2, 3, 1) => -4,
            (5, 2, 4, 1) => 17,
            (6, 2, 1, 1) => 14,
            (6, 2, 2, 1) => 6,
            (6, 2, 3, 1) => 5,
            (6, 2, 4, 1) => 18,
            (7, 2, 1, 1) => 15,
            (7, 2, 2, 1) => 7,
            (7, 2, 3, 1) => 8,
            (7, 2, 4, 1) => 10,
            (8, 2, 1, 1) => 8,
            (8, 2, 2, 1) => 19,
            (8, 2, 3, 1) => 16,
            (8, 2, 4, 1) => 9,
            (9, 2, 1, 1) => 13,
            (9, 2, 2, 1) => 12,
            (9, 2, 3, 1) => 15,
            (9, 2, 4, 1) => 11,
            (10, 2, 1, 1) => -13,
            (10, 2, 2, 1) => 20,
            (10, 2, 3, 1) => 14,
            (10, 2, 4, 1) => 16,
            (11, 2, 1, 1) => 20,
            (11, 2, 2, 1) => 17,
            (11, 2, 3, 1) => 18,
            (11, 2, 4, 1) => 19,
        )
        @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
        expected_global_ids2 = Dict{NTuple{3, Int}, Int}(
            (1, 1, 0) => 3,
            (1, 2, 0) => 2,
            (1, 3, 0) => 1,
            (1, 4, 0) => 4,
            (2, 1, 0) => 6,
            (2, 2, 0) => 5,
            (2, 3, 0) => 1,
            (2, 4, 0) => 2,
            (3, 1, 0) => 8,
            (3, 2, 0) => 4,
            (3, 3, 0) => 1,
            (3, 4, 0) => 5,
            (4, 1, 0) => 7,
            (4, 2, 0) => 6,
            (4, 3, 0) => 2,
            (4, 4, 0) => 3,
            (5, 1, 0) => 2,
            (5, 2, 0) => 3,
            (5, 3, 0) => 10,
            (5, 4, 0) => 9,
            (6, 1, 0) => 6,
            (6, 2, 0) => 2,
            (6, 3, 0) => 9,
            (6, 4, 0) => 11,
            (7, 1, 0) => 7,
            (7, 2, 0) => 3,
            (7, 3, 0) => 4,
            (7, 4, 0) => 8,
            (8, 1, 0) => 7,
            (8, 2, 0) => 12,
            (8, 3, 0) => 10,
            (8, 4, 0) => 3,
            (9, 1, 0) => 7,
            (9, 2, 0) => 8,
            (9, 3, 0) => 5,
            (9, 4, 0) => 6,
            (10, 1, 0) => 6,
            (10, 2, 0) => 11,
            (10, 3, 0) => 12,
            (10, 4, 0) => 7,
            (11, 1, 0) => 11,
            (11, 2, 0) => 9,
            (11, 3, 0) => 10,
            (11, 4, 0) => 12,
            (1, 1, 1) => 7,
            (1, 2, 1) => 1,
            (1, 3, 1) => 4,
            (1, 4, 1) => 2,
            (2, 1, 1) => 5,
            (2, 2, 1) => 3,
            (2, 3, 1) => 11,
            (2, 4, 1) => 1,
            (3, 1, 1) => 12,
            (3, 2, 1) => 2,
            (3, 3, 1) => 10,
            (3, 4, 1) => 3,
            (4, 1, 1) => 8,
            (4, 2, 1) => 5,
            (4, 3, 1) => 13,
            (4, 4, 1) => 4,
            (5, 1, 1) => 6,
            (5, 2, 1) => 9,
            (5, 3, 1) => -4,
            (5, 4, 1) => 17,
            (6, 1, 1) => 14,
            (6, 2, 1) => 6,
            (6, 3, 1) => 5,
            (6, 4, 1) => 18,
            (7, 1, 1) => 15,
            (7, 2, 1) => 7,
            (7, 3, 1) => 8,
            (7, 4, 1) => 10,
            (8, 1, 1) => 8,
            (8, 2, 1) => 19,
            (8, 3, 1) => 16,
            (8, 4, 1) => 9,
            (9, 1, 1) => 13,
            (9, 2, 1) => 12,
            (9, 3, 1) => 15,
            (9, 4, 1) => 11,
            (10, 1, 1) => -13,
            (10, 2, 1) => 20,
            (10, 3, 1) => 14,
            (10, 4, 1) => 16,
            (11, 1, 1) => 20,
            (11, 2, 1) => 17,
            (11, 3, 1) => 18,
            (11, 4, 1) => 19,
        )
        @test @expected expected_global_ids2 k -> Topology.get_global_id(topology, k...)

        expected_local_ids = Dict{NTuple{3, Int}, Int}(
            (1, 3, 0) => 1,
            (1, 2, 0) => 2,
            (1, 1, 0) => 3,
            (1, 4, 0) => 4,
            (2, 6, 0) => 1,
            (2, 5, 0) => 2,
            (2, 1, 0) => 3,
            (2, 2, 0) => 4,
            (3, 8, 0) => 1,
            (3, 4, 0) => 2,
            (3, 1, 0) => 3,
            (3, 5, 0) => 4,
            (4, 7, 0) => 1,
            (4, 6, 0) => 2,
            (4, 2, 0) => 3,
            (4, 3, 0) => 4,
            (5, 2, 0) => 1,
            (5, 3, 0) => 2,
            (5, 10, 0) => 3,
            (5, 9, 0) => 4,
            (6, 6, 0) => 1,
            (6, 2, 0) => 2,
            (6, 9, 0) => 3,
            (6, 11, 0) => 4,
            (7, 7, 0) => 1,
            (7, 3, 0) => 2,
            (7, 4, 0) => 3,
            (7, 8, 0) => 4,
            (8, 7, 0) => 1,
            (8, 12, 0) => 2,
            (8, 10, 0) => 3,
            (8, 3, 0) => 4,
            (9, 7, 0) => 1,
            (9, 8, 0) => 2,
            (9, 5, 0) => 3,
            (9, 6, 0) => 4,
            (10, 6, 0) => 1,
            (10, 11, 0) => 2,
            (10, 12, 0) => 3,
            (10, 7, 0) => 4,
            (11, 11, 0) => 1,
            (11, 9, 0) => 2,
            (11, 10, 0) => 3,
            (11, 12, 0) => 4,
            (1, 7, 1) => 1,
            (1, 1, 1) => 2,
            (1, 4, 1) => 3,
            (1, 2, 1) => 4,
            (2, 5, 1) => 1,
            (2, 3, 1) => 2,
            (2, 11, 1) => 3,
            (2, 1, 1) => 4,
            (3, 12, 1) => 1,
            (3, 2, 1) => 2,
            (3, 10, 1) => 3,
            (3, 3, 1) => 4,
            (4, 8, 1) => 1,
            (4, 5, 1) => 2,
            (4, 13, 1) => 3,
            (4, 4, 1) => 4,
            (5, 6, 1) => 1,
            (5, 9, 1) => 2,
            (5, 4, 1) => -3,
            (5, 17, 1) => 4,
            (6, 14, 1) => 1,
            (6, 6, 1) => 2,
            (6, 5, 1) => 3,
            (6, 18, 1) => 4,
            (7, 15, 1) => 1,
            (7, 7, 1) => 2,
            (7, 8, 1) => 3,
            (7, 10, 1) => 4,
            (8, 8, 1) => 1,
            (8, 19, 1) => 2,
            (8, 16, 1) => 3,
            (8, 9, 1) => 4,
            (9, 13, 1) => 1,
            (9, 12, 1) => 2,
            (9, 15, 1) => 3,
            (9, 11, 1) => 4,
            (10, 13, 1) => -1,
            (10, 20, 1) => 2,
            (10, 14, 1) => 3,
            (10, 16, 1) => 4,
            (11, 20, 1) => 1,
            (11, 17, 1) => 2,
            (11, 18, 1) => 3,
            (11, 19, 1) => 4,
        )
        @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

        expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
            # Vertex neighbours
            (1, 1, 0, false) => [4 5 7 8; 4 2 2 4; 0 0 0 0; 1 1 1 1],
            (1, 2, 0, false) => [2 4 5 6; 4 3 1 2; 0 0 0 0; 1 1 1 1],
            (1, 3, 0, false) => [2 3; 3 3; 0 0; 1 1],
            (1, 4, 0, false) => [3 7; 2 3; 0 0; 1 1],
            (2, 1, 0, false) => [4 6 9 10; 2 1 4 1; 0 0 0 0; 1 1 1 1],
            (2, 2, 0, false) => [3 9; 4 3; 0 0; 1 1],
            (2, 3, 0, false) => [1 3; 3 3; 0 0; 1 1],
            (2, 4, 0, false) => [1 4 5 6; 2 3 1 2; 0 0 0 0; 1 1 1 1],
            (3, 1, 0, false) => [7 9; 4 2; 0 0; 1 1],
            (3, 2, 0, false) => [1 7; 4 3; 0 0; 1 1],
            (3, 3, 0, false) => [1 2; 3 3; 0 0; 1 1],
            (3, 4, 0, false) => [2 9; 2 3; 0 0; 1 1],
            (4, 1, 0, false) => [7 8 9 10; 1 1 1 4; 0 0 0 0; 1 1 1 1],
            (4, 2, 0, false) => [2 6 9 10; 1 1 4 1; 0 0 0 0; 1 1 1 1],
            (4, 3, 0, false) => [1 2 5 6; 2 4 1 2; 0 0 0 0; 1 1 1 1],
            (4, 4, 0, false) => [1 5 7 8; 1 2 2 4; 0 0 0 0; 1 1 1 1],
            (5, 1, 0, false) => [1 2 4 6; 2 4 3 2; 0 0 0 0; 1 1 1 1],
            (5, 2, 0, false) => [1 4 7 8; 1 4 2 4; 0 0 0 0; 1 1 1 1],
            (5, 3, 0, false) => [8 11; 3 3; 0 0; 1 1],
            (5, 4, 0, false) => [6 11; 3 2; 0 0; 1 1],
            (6, 1, 0, false) => [2 4 9 10; 1 2 4 1; 0 0 0 0; 1 1 1 1],
            (6, 2, 0, false) => [1 2 4 5; 2 4 3 1; 0 0 0 0; 1 1 1 1],
            (6, 3, 0, false) => [5 11; 4 2; 0 0; 1 1],
            (6, 4, 0, false) => [10 11; 2 1; 0 0; 1 1],
            (7, 1, 0, false) => [4 8 9 10; 1 1 1 4; 0 0 0 0; 1 1 1 1],
            (7, 2, 0, false) => [1 4 5 8; 1 4 2 4; 0 0 0 0; 1 1 1 1],
            (7, 3, 0, false) => [1 3; 4 2; 0 0; 1 1],
            (7, 4, 0, false) => [3 9; 1 2; 0 0; 1 1],
            (8, 1, 0, false) => [4 7 9 10; 1 1 1 4; 0 0 0 0; 1 1 1 1],
            (8, 2, 0, false) => [10 11; 3 4; 0 0; 1 1],
            (8, 3, 0, false) => [5 11; 3 3; 0 0; 1 1],
            (8, 4, 0, false) => [1 4 5 7; 1 4 2 2; 0 0 0 0; 1 1 1 1],
            (9, 1, 0, false) => [4 7 8 10; 1 1 1 4; 0 0 0 0; 1 1 1 1],
            (9, 2, 0, false) => [3 7; 1 4; 0 0; 1 1],
            (9, 3, 0, false) => [2 3; 2 4; 0 0; 1 1],
            (9, 4, 0, false) => [2 4 6 10; 1 2 1 1; 0 0 0 0; 1 1 1 1],
            (10, 1, 0, false) => [2 4 6 9; 1 2 1 4; 0 0 0 0; 1 1 1 1],
            (10, 2, 0, false) => [6 11; 4 1; 0 0; 1 1],
            (10, 3, 0, false) => [8 11; 2 4; 0 0; 1 1],
            (10, 4, 0, false) => [4 7 8 9; 1 1 1 1; 0 0 0 0; 1 1 1 1],
            (11, 1, 0, false) => [6 10; 4 2; 0 0; 1 1],
            (11, 2, 0, false) => [5 6; 4 3; 0 0; 1 1],
            (11, 3, 0, false) => [5 8; 3 3; 0 0; 1 1],
            (11, 4, 0, false) => [8 10; 2 3; 0 0; 1 1],
            (1, 1, 0, true) => [1 4 5 7 8; 1 4 2 2 4; 0 0 0 0 0; 1 1 1 1 1],
            (1, 2, 0, true) => [1 2 4 5 6; 2 4 3 1 2; 0 0 0 0 0; 1 1 1 1 1],
            (1, 3, 0, true) => [1 2 3; 3 3 3; 0 0 0; 1 1 1],
            (1, 4, 0, true) => [1 3 7; 4 2 3; 0 0 0; 1 1 1],
            (2, 1, 0, true) => [2 4 6 9 10; 1 2 1 4 1; 0 0 0 0 0; 1 1 1 1 1],
            (2, 2, 0, true) => [2 3 9; 2 4 3; 0 0 0; 1 1 1],
            (2, 3, 0, true) => [1 2 3; 3 3 3; 0 0 0; 1 1 1],
            (2, 4, 0, true) => [1 2 4 5 6; 2 4 3 1 2; 0 0 0 0 0; 1 1 1 1 1],
            (3, 1, 0, true) => [3 7 9; 1 4 2; 0 0 0; 1 1 1],
            (3, 2, 0, true) => [1 3 7; 4 2 3; 0 0 0; 1 1 1],
            (3, 3, 0, true) => [1 2 3; 3 3 3; 0 0 0; 1 1 1],
            (3, 4, 0, true) => [2 3 9; 2 4 3; 0 0 0; 1 1 1],
            (4, 1, 0, true) => [4 7 8 9 10; 1 1 1 1 4; 0 0 0 0 0; 1 1 1 1 1],
            (4, 2, 0, true) => [2 4 6 9 10; 1 2 1 4 1; 0 0 0 0 0; 1 1 1 1 1],
            (4, 3, 0, true) => [1 2 4 5 6; 2 4 3 1 2; 0 0 0 0 0; 1 1 1 1 1],
            (4, 4, 0, true) => [1 4 5 7 8; 1 4 2 2 4; 0 0 0 0 0; 1 1 1 1 1],
            (5, 1, 0, true) => [1 2 4 5 6; 2 4 3 1 2; 0 0 0 0 0; 1 1 1 1 1],
            (5, 2, 0, true) => [1 4 5 7 8; 1 4 2 2 4; 0 0 0 0 0; 1 1 1 1 1],
            (5, 3, 0, true) => [5 8 11; 3 3 3; 0 0 0; 1 1 1],
            (5, 4, 0, true) => [5 6 11; 4 3 2; 0 0 0; 1 1 1],
            (6, 1, 0, true) => [2 4 6 9 10; 1 2 1 4 1; 0 0 0 0 0; 1 1 1 1 1],
            (6, 2, 0, true) => [1 2 4 5 6; 2 4 3 1 2; 0 0 0 0 0; 1 1 1 1 1],
            (6, 3, 0, true) => [5 6 11; 4 3 2; 0 0 0; 1 1 1],
            (6, 4, 0, true) => [6 10 11; 4 2 1; 0 0 0; 1 1 1],
            (7, 1, 0, true) => [4 7 8 9 10; 1 1 1 1 4; 0 0 0 0 0; 1 1 1 1 1],
            (7, 2, 0, true) => [1 4 5 7 8; 1 4 2 2 4; 0 0 0 0 0; 1 1 1 1 1],
            (7, 3, 0, true) => [1 3 7; 4 2 3; 0 0 0; 1 1 1],
            (7, 4, 0, true) => [3 7 9; 1 4 2; 0 0 0; 1 1 1],
            (8, 1, 0, true) => [4 7 8 9 10; 1 1 1 1 4; 0 0 0 0 0; 1 1 1 1 1],
            (8, 2, 0, true) => [8 10 11; 2 3 4; 0 0 0; 1 1 1],
            (8, 3, 0, true) => [5 8 11; 3 3 3; 0 0 0; 1 1 1],
            (8, 4, 0, true) => [1 4 5 7 8; 1 4 2 2 4; 0 0 0 0 0; 1 1 1 1 1],
            (9, 1, 0, true) => [4 7 8 9 10; 1 1 1 1 4; 0 0 0 0 0; 1 1 1 1 1],
            (9, 2, 0, true) => [3 7 9; 1 4 2; 0 0 0; 1 1 1],
            (9, 3, 0, true) => [2 3 9; 2 4 3; 0 0 0; 1 1 1],
            (9, 4, 0, true) => [2 4 6 9 10; 1 2 1 4 1; 0 0 0 0 0; 1 1 1 1 1],
            (10, 1, 0, true) => [2 4 6 9 10; 1 2 1 4 1; 0 0 0 0 0; 1 1 1 1 1],
            (10, 2, 0, true) => [6 10 11; 4 2 1; 0 0 0; 1 1 1],
            (10, 3, 0, true) => [8 10 11; 2 3 4; 0 0 0; 1 1 1],
            (10, 4, 0, true) => [4 7 8 9 10; 1 1 1 1 4; 0 0 0 0 0; 1 1 1 1 1],
            (11, 1, 0, true) => [6 10 11; 4 2 1; 0 0 0; 1 1 1],
            (11, 2, 0, true) => [5 6 11; 4 3 2; 0 0 0; 1 1 1],
            (11, 3, 0, true) => [5 8 11; 3 3 3; 0 0 0; 1 1 1],
            (11, 4, 0, true) => [8 10 11; 2 3 4; 0 0 0; 1 1 1],
            # Edge neighbours
            (1, 1, 1, false) => [7; 2; 0; 1;;],
            (1, 2, 1, false) => [2; 4; 0; 1;;],
            (1, 3, 1, false) => [4 5; 4 3; 0 0; 1 -1],
            (1, 4, 1, false) => [3; 2; 0; 1;;],
            (2, 1, 1, false) => [4 6; 2 3; 0 0; 1 1],
            (2, 2, 1, false) => [3; 4; 0; 1;;],
            (2, 3, 1, false) => [9; 4; 0; 1;;],
            (2, 4, 1, false) => [1; 2; 0; 1;;],
            (3, 1, 1, false) => [9; 2; 0; 1;;],
            (3, 2, 1, false) => [1; 4; 0; 1;;],
            (3, 3, 1, false) => [7; 4; 0; 1;;],
            (3, 4, 1, false) => [2; 2; 0; 1;;],
            (4, 1, 1, false) => [7 8; 3 1; 0 0; 1 1],
            (4, 2, 1, false) => [2 6; 1 3; 0 0; 1 1],
            (4, 3, 1, false) => [9 10; 1 1; 0 0; 1 -1],
            (4, 4, 1, false) => [1 5; 3 3; 0 0; 1 -1],
            (5, 1, 1, false) => [6; 2; 0; 1;;],
            (5, 2, 1, false) => [8; 4; 0; 1;;],
            (5, 3, 1, false) => [1 4; 3 4; 0 0; -1 -1],
            (5, 4, 1, false) => [11; 2; 0; 1;;],
            (6, 1, 1, false) => [10; 3; 0; 1;;],
            (6, 2, 1, false) => [5; 1; 0; 1;;],
            (6, 3, 1, false) => [2 4; 1 2; 0 0; 1 1],
            (6, 4, 1, false) => [11; 3; 0; 1;;],
            (7, 1, 1, false) => [9; 3; 0; 1;;],
            (7, 2, 1, false) => [1; 1; 0; 1;;],
            (7, 3, 1, false) => [4 8; 1 1; 0 0; 1 1],
            (7, 4, 1, false) => [3; 3; 0; 1;;],
            (8, 1, 1, false) => [4 7; 1 3; 0 0; 1 1],
            (8, 2, 1, false) => [11; 4; 0; 1;;],
            (8, 3, 1, false) => [10; 4; 0; 1;;],
            (8, 4, 1, false) => [5; 2; 0; 1;;],
            (9, 1, 1, false) => [4 10; 3 1; 0 0; 1 -1],
            (9, 2, 1, false) => [3; 1; 0; 1;;],
            (9, 3, 1, false) => [7; 1; 0; 1;;],
            (9, 4, 1, false) => [2; 3; 0; 1;;],
            (10, 1, 1, false) => [4 9; 3 1; 0 0; -1 -1],
            (10, 2, 1, false) => [11; 1; 0; 1;;],
            (10, 3, 1, false) => [6; 1; 0; 1;;],
            (10, 4, 1, false) => [8; 3; 0; 1;;],
            (11, 1, 1, false) => [10; 2; 0; 1;;],
            (11, 2, 1, false) => [5; 4; 0; 1;;],
            (11, 3, 1, false) => [6; 4; 0; 1;;],
            (11, 4, 1, false) => [8; 2; 0; 1;;],
            (1, 1, 1, true) => [1 7; 1 2; 0 0; 1 1],
            (1, 2, 1, true) => [1 2; 2 4; 0 0; 1 1],
            (1, 3, 1, true) => [1 4 5; 3 4 3; 0 0 0; 1 1 -1],
            (1, 4, 1, true) => [1 3; 4 2; 0 0; 1 1],
            (2, 1, 1, true) => [2 4 6; 1 2 3; 0 0 0; 1 1 1],
            (2, 2, 1, true) => [2 3; 2 4; 0 0; 1 1],
            (2, 3, 1, true) => [2 9; 3 4; 0 0; 1 1],
            (2, 4, 1, true) => [1 2; 2 4; 0 0; 1 1],
            (3, 1, 1, true) => [3 9; 1 2; 0 0; 1 1],
            (3, 2, 1, true) => [1 3; 4 2; 0 0; 1 1],
            (3, 3, 1, true) => [3 7; 3 4; 0 0; 1 1],
            (3, 4, 1, true) => [2 3; 2 4; 0 0; 1 1],
            (4, 1, 1, true) => [4 7 8; 1 3 1; 0 0 0; 1 1 1],
            (4, 2, 1, true) => [2 4 6; 1 2 3; 0 0 0; 1 1 1],
            (4, 3, 1, true) => [4 9 10; 3 1 1; 0 0 0; 1 1 -1],
            (4, 4, 1, true) => [1 4 5; 3 4 3; 0 0 0; 1 1 -1],
            (5, 1, 1, true) => [5 6; 1 2; 0 0; 1 1],
            (5, 2, 1, true) => [5 8; 2 4; 0 0; 1 1],
            (5, 3, 1, true) => [1 4 5; 3 4 3; 0 0 0; 1 1 -1],
            (5, 4, 1, true) => [5 11; 4 2; 0 0; 1 1],
            (6, 1, 1, true) => [6 10; 1 3; 0 0; 1 1],
            (6, 2, 1, true) => [5 6; 1 2; 0 0; 1 1],
            (6, 3, 1, true) => [2 4 6; 1 2 3; 0 0 0; 1 1 1],
            (6, 4, 1, true) => [6 11; 4 3; 0 0; 1 1],
            (7, 1, 1, true) => [7 9; 1 3; 0 0; 1 1],
            (7, 2, 1, true) => [1 7; 1 2; 0 0; 1 1],
            (7, 3, 1, true) => [4 7 8; 1 3 1; 0 0 0; 1 1 1],
            (7, 4, 1, true) => [3 7; 3 4; 0 0; 1 1],
            (8, 1, 1, true) => [4 7 8; 1 3 1; 0 0 0; 1 1 1],
            (8, 2, 1, true) => [8 11; 2 4; 0 0; 1 1],
            (8, 3, 1, true) => [8 10; 3 4; 0 0; 1 1],
            (8, 4, 1, true) => [5 8; 2 4; 0 0; 1 1],
            (9, 1, 1, true) => [4 9 10; 3 1 1; 0 0 0; 1 1 -1],
            (9, 2, 1, true) => [3 9; 1 2; 0 0; 1 1],
            (9, 3, 1, true) => [7 9; 1 3; 0 0; 1 1],
            (9, 4, 1, true) => [2 9; 3 4; 0 0; 1 1],
            (10, 1, 1, true) => [4 9 10; 3 1 1; 0 0 0; 1 1 -1],
            (10, 2, 1, true) => [10 11; 2 1; 0 0; 1 1],
            (10, 3, 1, true) => [6 10; 1 3; 0 0; 1 1],
            (10, 4, 1, true) => [8 10; 3 4; 0 0; 1 1],
            (11, 1, 1, true) => [10 11; 2 1; 0 0; 1 1],
            (11, 2, 1, true) => [5 11; 4 2; 0 0; 1 1],
            (11, 3, 1, true) => [6 11; 4 3; 0 0; 1 1],
            (11, 4, 1, true) => [8 11; 2 4; 0 0; 1 1],
        )
        @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test isempty(boundaries)
        @test interfaces == [
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
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
        ]
        @test isempty(Topology.get_interfaces_on_boundary(topology))

        expected_patch_parents = Dict{Int, Matrix{Int}}(
            1 => [1; 5; 2; 1;;],
            2 => [1; 3; 2; 1;;],
            3 => [1; 1; 2; 1;;],
            4 => [1 2; 2 2; 2 3; 1 -1],
            5 => [2; 5; 2; 1;;],
            6 => [2; 4; 2; 1;;],
            7 => [1; 4; 2; 1;;],
            8 => [2; 3; 2; 1;;],
            9 => [1; 6; 2; 1;;],
            10 => [2; 6; 2; 1;;],
            11 => [2; 1; 2; 1;;],
        )
        @test @expected expected_patch_parents k -> Topology.get_patch_parents(topology, k)

        # The rotation and orientation reproduce the vertices of the skeleton patch as
        # traversed by each of its parent patches.
        expected_parent_vertices = Dict{Tuple{Int, Int}, Vector{Int}}(
            (1, 1) => [1, 4, 3, 2],
            (2, 1) => [1, 2, 6, 5],
            (3, 1) => [1, 5, 8, 4],
            (4, 1) => [2, 3, 7, 6],
            (4, 2) => [3, 2, 6, 7],
            (5, 1) => [10, 9, 2, 3],
            (6, 1) => [9, 11, 6, 2],
            (7, 1) => [4, 8, 7, 3],
            (8, 1) => [10, 3, 7, 12],
            (9, 1) => [5, 6, 7, 8],
            (10, 1) => [12, 7, 6, 11],
            (11, 1) => [10, 12, 11, 9],
        )
        @test @expected expected_parent_vertices k -> begin
            parents = Topology.get_patch_parents(topology, k[1])
            reconstruct_vertices(
                Topology.get_parent_topology(topology)[3, 1][k[1]],
                parents[3, k[2]],
                parents[4, k[2]],
            )
        end

    end

    @testset "structured grid: 2 × 2 × 2" verbose=true begin
        parent = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX, Val(8))
        topology = Topology.SkeletonTopology(parent)

        @test Topology.get_parent_topology(topology) === parent
        @test topology == Topology.SkeletonTopology(parent)
        @test hash(topology) == hash(Topology.SkeletonTopology(parent))
        # A skeleton and its parent are different topologies.
        @test topology != parent

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 36
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (27, 54, 36)
        @test @expected Dict(1 => 27, 2 => 54, 3 => 36) k -> size(topology, k)
        @test Topology.get_local_size(topology) == (4, 4, 1)
        expected_local_size_dim = Dict(1 => 4, 2 => 4, 3 => 1)
        @test @expected expected_local_size_dim k -> Topology.get_local_size(topology, k)
        @test lastindex(topology) == 3

        # The skeleton shares the incidence relations of its parent.
        @test all(
            topology[i, k] === parent[i, k] for i in 1:3,
            k in 1:3
        )

        expected_patch_parents = Dict{Int, Matrix{Int}}(
            1 => [1; 5; 2; 1;;],
            2 => [1; 3; 2; 1;;],
            3 => [1; 1; 2; 1;;],
            4 => [2; 5; 2; 1;;],
            5 => [2; 3; 2; 1;;],
            6 => [1 2; 2 1; 2 2; -1 1],
            7 => [2; 2; 2; 1;;],
            8 => [3; 5; 2; 1;;],
            9 => [1 3; 4 3; 2 2; -1 1],
            10 => [3; 1; 2; 1;;],
            11 => [4; 5; 2; 1;;],
            12 => [2 4; 4 3; 2 2; -1 1],
            13 => [3 4; 2 1; 2 2; -1 1],
            14 => [4; 2; 2; 1;;],
            15 => [3; 4; 2; 1;;],
            16 => [4; 4; 2; 1;;],
            17 => [1 5; 6 5; 2 2; -1 1],
            18 => [5; 3; 2; 1;;],
            19 => [5; 1; 2; 1;;],
            20 => [2 6; 6 5; 2 2; -1 1],
            21 => [6; 3; 2; 1;;],
            22 => [5 6; 2 1; 2 2; -1 1],
            23 => [6; 2; 2; 1;;],
            24 => [3 7; 6 5; 2 2; -1 1],
            25 => [5 7; 4 3; 2 2; -1 1],
            26 => [7; 1; 2; 1;;],
            27 => [4 8; 6 5; 2 2; -1 1],
            28 => [6 8; 4 3; 2 2; -1 1],
            29 => [7 8; 2 1; 2 2; -1 1],
            30 => [8; 2; 2; 1;;],
            31 => [7; 4; 2; 1;;],
            32 => [8; 4; 2; 1;;],
            33 => [5; 6; 2; 1;;],
            34 => [6; 6; 2; 1;;],
            35 => [7; 6; 2; 1;;],
            36 => [8; 6; 2; 1;;],
        )
        @test @expected expected_patch_parents k -> Topology.get_patch_parents(topology, k)

        # The rotation and orientation reproduce the vertices of the skeleton patch as
        # traversed by each of its parent patches.
        expected_parent_vertices = Dict{Tuple{Int, Int}, Vector{Int}}(
            (1, 1) => [1, 4, 5, 2],
            (2, 1) => [1, 2, 11, 10],
            (3, 1) => [1, 10, 13, 4],
            (4, 1) => [2, 5, 6, 3],
            (5, 1) => [2, 3, 12, 11],
            (6, 1) => [2, 5, 14, 11],
            (6, 2) => [2, 11, 14, 5],
            (7, 1) => [3, 6, 15, 12],
            (8, 1) => [4, 7, 8, 5],
            (9, 1) => [4, 13, 14, 5],
            (9, 2) => [4, 5, 14, 13],
            (10, 1) => [4, 13, 16, 7],
            (11, 1) => [5, 8, 9, 6],
            (12, 1) => [5, 14, 15, 6],
            (12, 2) => [5, 6, 15, 14],
            (13, 1) => [5, 8, 17, 14],
            (13, 2) => [5, 14, 17, 8],
            (14, 1) => [6, 9, 18, 15],
            (15, 1) => [7, 16, 17, 8],
            (16, 1) => [8, 17, 18, 9],
            (17, 1) => [10, 11, 14, 13],
            (17, 2) => [10, 13, 14, 11],
            (18, 1) => [10, 11, 20, 19],
            (19, 1) => [10, 19, 22, 13],
            (20, 1) => [11, 12, 15, 14],
            (20, 2) => [11, 14, 15, 12],
            (21, 1) => [11, 12, 21, 20],
            (22, 1) => [11, 14, 23, 20],
            (22, 2) => [11, 20, 23, 14],
            (23, 1) => [12, 15, 24, 21],
            (24, 1) => [13, 14, 17, 16],
            (24, 2) => [13, 16, 17, 14],
            (25, 1) => [13, 22, 23, 14],
            (25, 2) => [13, 14, 23, 22],
            (26, 1) => [13, 22, 25, 16],
            (27, 1) => [14, 15, 18, 17],
            (27, 2) => [14, 17, 18, 15],
            (28, 1) => [14, 23, 24, 15],
            (28, 2) => [14, 15, 24, 23],
            (29, 1) => [14, 17, 26, 23],
            (29, 2) => [14, 23, 26, 17],
            (30, 1) => [15, 18, 27, 24],
            (31, 1) => [16, 25, 26, 17],
            (32, 1) => [17, 26, 27, 18],
            (33, 1) => [19, 20, 23, 22],
            (34, 1) => [20, 21, 24, 23],
            (35, 1) => [22, 23, 26, 25],
            (36, 1) => [23, 24, 27, 26],
        )
        @test @expected expected_parent_vertices k -> begin
            parents = Topology.get_patch_parents(topology, k[1])
            reconstruct_vertices(
                Topology.get_parent_topology(topology)[3, 1][k[1]],
                parents[3, k[2]],
                parents[4, k[2]],
            )
        end

    end
end

############################################################################################
#                                     Argument errors                                      #
############################################################################################
@testset "Argument errors" begin
    hex_topology = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX, Val(8))

    @testset "Index bounds" begin
        skeleton = Topology.SkeletonTopology(hex_topology)
        @test_throws BoundsError skeleton[4, 1]
        @test_throws BoundsError skeleton[1, 4]
        # Checking the bounds of size with 4 will trigger the boundscheck for the skeleton.
        # If this is ever removed, this test will fail, because 4 is not out-of-bounds for
        # the parent (size(skeleton) calls size on the parent).
        @test_throws BoundsError size(skeleton, 4)
    end
end

end

module MeshTopologyTestModule

using Test
using Mantis
import MeshCore

const Topology = Mantis.Topology

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
    test_incidence_relations(topology, expected)

Check `topology[i, k] == expected[(i, k)]` for every key of `expected`.
"""
function test_incidence_relations(topology, expected)
    @testset "incidence relation $key" for (key, value) in pairs(expected)
        @test topology[key[1], key[2]] == value
    end
end

"""
    test_global_ids(topology, expected)

Check `get_global_id` for every key `(container_id, container_dim, local_id, local_dim)` of
`expected`, both through the five- and the four-argument method where applicable.
"""
function test_global_ids(topology, expected)
    manifold_dim = Topology.get_manifold_dim(topology)
    @testset "global id of $key" for (key, value) in pairs(expected)
        container_id, container_dim, local_id, local_dim = key
        @test Topology.get_global_id(
            topology, container_id, container_dim, local_id, local_dim
        ) == value
        if container_dim == manifold_dim
            @test Topology.get_global_id(topology, container_id, local_id, local_dim) ==
                value
        end
    end
end

"""
    test_local_ids(topology, expected)

Check `get_local_id` for every key `(patch_id, global_object_id, object_dim)` of `expected`.
"""
function test_local_ids(topology, expected)
    @testset "local id of $key" for (key, value) in pairs(expected)
        @test Topology.get_local_id(topology, key[1], key[2], key[3]) == value
    end
end

"""
    test_neighbours(topology, object_dim, expected)

Check `compute_neighbours` for every key `(patch_id, local_object_id, include_local_patch)`
of `expected`, and check that the per-object and the whole-mesh methods agree.
"""
function test_neighbours(topology, object_dim, expected)
    @testset "neighbours of $key" for (key, value) in pairs(expected)
        patch_id, local_object_id, include_local_patch = key
        @test Topology.compute_neighbours(
            topology, patch_id, local_object_id, object_dim; include_local_patch
        ) == value
    end

    @testset "whole-mesh neighbours, include_local_patch = $include_local_patch" for include_local_patch in
                                                                                     (
        false, true
    )
        all_neighbours = Topology.compute_neighbours(
            topology, object_dim; include_local_patch
        )
        @test size(all_neighbours) == (
            size(topology, Topology.get_manifold_dim(topology) + 1),
            Topology.get_local_size(topology, object_dim + 1),
        )
        for patch_id in axes(all_neighbours, 1), local_id in axes(all_neighbours, 2)
            @test all_neighbours[patch_id, local_id] == Topology.compute_neighbours(
                topology, patch_id, local_id, object_dim; include_local_patch
            )
        end
    end
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

"""
    test_neighbour_invariants(topology)

Check the properties that any well-formed topology must satisfy: neighbourhood is symmetric,
`include_local_patch` adds exactly the patch itself, and the reported rotation and
orientation really do map the reference vertex sequence onto the neighbour's one.
"""
function test_neighbour_invariants(topology)
    manifold_dim = Topology.get_manifold_dim(topology)
    num_patches = size(topology, manifold_dim + 1)

    @testset "objects of dimension $object_dim" for object_dim in 0:(manifold_dim - 1)
        num_local_objects = Topology.get_local_size(topology, object_dim + 1)

        for patch_id in 1:num_patches, local_id in 1:num_local_objects
            neighbours = Topology.compute_neighbours(
                topology, patch_id, local_id, object_dim
            )
            with_self = Topology.compute_neighbours(
                topology, patch_id, local_id, object_dim; include_local_patch=true
            )

            # Including the local patch adds exactly one column, holding the patch itself.
            @test size(with_self, 2) == size(neighbours, 2) + 1
            @test count(
                k -> with_self[1, k] == patch_id && with_self[2, k] == local_id,
                axes(with_self, 2),
            ) == 1

            for column in axes(neighbours, 2)
                neighbour_id, neighbour_local_id, rotation, orientation = neighbours[
                    :, column
                ]

                @test neighbour_id != patch_id
                @test orientation in (-1, 1)
                @test 0 <= rotation < Topology.get_local_size(topology, 1)

                # Neighbourhood is symmetric.
                back = Topology.compute_neighbours(
                    topology, neighbour_id, neighbour_local_id, object_dim
                )
                @test any(
                    k -> back[1, k] == patch_id && back[2, k] == local_id, axes(back, 2)
                )

                # Both patches agree on which global object they share.
                @test abs(
                    Topology.get_global_id(topology, patch_id, local_id, object_dim)
                ) == abs(
                    Topology.get_global_id(
                        topology, neighbour_id, neighbour_local_id, object_dim
                    ),
                )

                # Rotation and orientation map the reference vertex sequence, i.e. the one
                # of the current patch, onto the neighbour's one. Vertices have no vertex
                # sequence of their own.
                if object_dim > 0
                    reference = Topology._object_vertices_in_patch(
                        topology, patch_id, local_id, object_dim
                    )
                    neighbour_vertices = Topology._object_vertices_in_patch(
                        topology, neighbour_id, neighbour_local_id, object_dim
                    )
                    @test reconstruct_vertices(reference, rotation, orientation) ==
                        neighbour_vertices
                end
            end
        end
    end
end

"""
    test_structural_invariants(topology)

Check the properties that relate the incidence relations of a topology to each other: sizes
agree with the stored relations, transposed relations are consistent, and every object of a
patch is listed by the patch.
"""
function test_structural_invariants(topology)
    manifold_dim = Topology.get_manifold_dim(topology)
    num_objects = size(topology)

    @test length(num_objects) == manifold_dim + 1
    @test Topology.get_incidence_relations_dim(topology) == manifold_dim + 1
    @test Topology.get_num_patches(topology) == num_objects[manifold_dim + 1]
    @test lastindex(topology) == manifold_dim + 1

    @testset "relation ($i, $k)" for i in 1:(manifold_dim + 1), k in 1:(manifold_dim + 1)
        relation = topology[i, k]
        if i == k
            # The relation of an object with itself is trivial and left empty.
            @test isempty(relation)
        else
            @test length(relation) == num_objects[i]
            @test all(!isempty, relation)

            # Every relation is the transpose of its mirror image.
            transposed = topology[k, i]
            for (object_id, related) in pairs(relation), related_id in related
                @test object_id in abs.(transposed[abs(related_id)])
            end
        end
    end

    @test Topology.get_local_size(topology) ==
        size(Topology.get_topological_patch(topology))

    # Objects of the manifold dimension are the patches themselves and are covered by the
    # relation checks above.
    @testset "local objects of dimension $object_dim" for object_dim in 0:(manifold_dim - 1)
        num_local = Topology.get_local_size(topology, object_dim + 1)
        @test num_local == size(Topology.get_topological_patch(topology), object_dim + 1)

        objects = topology[manifold_dim + 1, object_dim + 1]
        for patch_id in 1:num_objects[manifold_dim + 1]
            @test length(objects[patch_id]) == num_local

            for local_id in 1:num_local
                global_id = Topology.get_global_id(topology, patch_id, local_id, object_dim)
                @test abs(
                    Topology.get_local_id(topology, patch_id, global_id, object_dim)
                ) == local_id

                # The vertices a patch assigns to one of its objects are the vertices that
                # the object itself lists. A vertex has no vertex sequence of its own.
                object_dim == 0 && continue
                @test sort(
                    Topology._object_vertices_in_patch(
                        topology, patch_id, local_id, object_dim
                    ),
                ) == sort(topology[object_dim + 1, 1][abs(global_id)])
            end
        end
    end
end

############################################################################################
#                                   Construction errors                                    #
############################################################################################
@testset "construction errors" begin
    # Incorrect type without vertices. The BoundsError is thrown in MeshCore.
    @test_throws MethodError Topology.MeshTopology([()], Topology.LINE)
    # Correct type but no actual vertices. The BoundsError is thrown in MeshCore.
    @test_throws BoundsError Topology.MeshTopology(NTuple{2, Int}[], Topology.LINE)
    # Invalid (0) vertex id.
    @test_throws ArgumentError Topology.MeshTopology([(0, 1)], Topology.LINE)
    # Invalid negative vertex id.
    @test_throws ArgumentError Topology.MeshTopology([(-1, 1)], Topology.LINE)
    # Vertex 3 does not belong to any patch.
    @test_throws ArgumentError Topology.MeshTopology([(1, 2), (4, 5)], Topology.LINE)
    # Vertex 6 does not belong to any patch.
    @test_throws ArgumentError Topology.MeshTopology(
        [(1, 2, 3, 4), (2, 5, 7, 3)], Topology.QUAD
    )
    # Vertex 10 does not belong to any patch.
    @test_throws ArgumentError Topology.MeshTopology(
        [(1, 2, 3, 4, 5, 6, 7, 8), (2, 9, 11, 3, 5, 12, 13, 7)], Topology.HEX
    )
    # The number of vertices per patch has to match the topological patch.
    @test_throws MethodError Topology.MeshTopology([(1, 2, 3, 4)], Topology.LINE)
    @test_throws MethodError Topology.MeshTopology([(1, 2)], Topology.QUAD)
    @test_throws MethodError Topology.MeshTopology([(1, 2, 3, 4)], Topology.HEX)
end

############################################################################################
#                                            1D                                            #
############################################################################################
@testset "1D" begin
    @testset "single patch: 1 --> 2" begin
        topology = Topology.MeshTopology([(1, 2)], Topology.LINE)

        @test Topology.get_manifold_dim(topology) == 1
        @test Topology.get_incidence_relations_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 1
        @test Topology.get_patch_type(topology) == typeof(Topology.LINE)
        @test Topology.get_topological_patch(topology) == Topology.LINE
        @test size(topology) == (2, 1)
        @test size(topology, 1) == 2
        @test size(topology, 2) == 1
        @test Topology.get_local_size(topology) == (2, 1)
        @test Topology.get_local_size(topology, 1) == 2
        @test Topology.get_local_size(topology, 2) == 1

        test_incidence_relations(
            topology,
            Dict(
                (1, 1) => Vector{Vector{Int}}(),
                (1, 2) => [[1], [1]],
                (2, 1) => [[1, 2]],
                (2, 2) => Vector{Vector{Int}}(),
            ),
        )
        test_global_ids(
            topology,
            Dict(
                (1, 1, 1, 0) => 1, (1, 1, 2, 0) => 2, (1, 0, 1, 1) => 1, (2, 0, 1, 1) => 1
            ),
        )
        test_local_ids(topology, Dict((1, 1, 0) => 1, (1, 2, 0) => 2))
        test_neighbours(
            topology,
            0,
            Dict(
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, false) => Matrix{Int}(undef, 4, 0),
                (1, 1, true) => [1; 1; 0; 1;;],
                (1, 2, true) => [1; 2; 0; 1;;],
            ),
        )

        # A single patch has only boundary vertices.
        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [(0, 1), (0, 2)]
        @test isempty(interfaces)
        @test isempty(Topology.get_interfaces_on_boundary(topology))

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    @testset "two patches: 1 --> 2 --> 3" begin
        topology = Topology.MeshTopology([(1, 2), (2, 3)], Topology.LINE)

        @test Topology.get_num_patches(topology) == 2
        @test size(topology) == (3, 2)

        test_incidence_relations(
            topology,
            Dict(
                (1, 1) => Vector{Vector{Int}}(),
                (1, 2) => [[1], [1, 2], [2]],
                (2, 1) => [[1, 2], [2, 3]],
                (2, 2) => Vector{Vector{Int}}(),
            ),
        )
        test_global_ids(
            topology,
            Dict(
                # (container_id, container_dim, local_id, local_dim), read as
                # topology[container_dim + 1, local_dim + 1][container_id][local_id]
                (1, 1, 1, 0) => 1,
                (1, 1, 2, 0) => 2,
                (2, 1, 1, 0) => 2,
                (2, 1, 2, 0) => 3,
                (1, 0, 1, 1) => 1,
                (2, 0, 1, 1) => 1,
                (2, 0, 2, 1) => 2,
                (3, 0, 1, 1) => 2,
            ),
        )
        test_local_ids(
            topology, Dict((1, 1, 0) => 1, (1, 2, 0) => 2, (2, 2, 0) => 1, (2, 3, 0) => 2)
        )
        test_neighbours(
            topology,
            0,
            Dict(
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, false) => [2; 1; 0; 1;;],
                (2, 1, false) => [1; 2; 0; 1;;],
                (2, 2, false) => Matrix{Int}(undef, 4, 0),
                (1, 1, true) => [1; 1; 0; 1;;],
                (1, 2, true) => [1 2; 2 1; 0 0; 1 1],
                (2, 1, true) => [1 2; 2 1; 0 0; 1 1],
                (2, 2, true) => [2; 2; 0; 1;;],
            ),
        )

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [(0, 1), (0, 3)]
        @test interfaces == [(0, 2)]

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    @testset "opposing patches: 1 --> 2 <-- 3 --> 4" begin
        topology = Topology.MeshTopology([(1, 2), (3, 2), (3, 4)], Topology.LINE)

        @test size(topology) == (4, 3)

        test_incidence_relations(
            topology,
            Dict((1, 2) => [[1], [1, 2], [2, 3], [3]], (2, 1) => [[1, 2], [3, 2], [3, 4]]),
        )
        test_global_ids(
            topology,
            Dict(
                (2, 1, 1, 0) => 3, # patch 2 starts at vertex 3
                (2, 1, 2, 0) => 2,
                (3, 1, 1, 0) => 3,
                (3, 1, 2, 0) => 4,
                (2, 0, 1, 1) => 1,
                (2, 0, 2, 1) => 2,
                (3, 0, 1, 1) => 2,
                (3, 0, 2, 1) => 3,
            ),
        )
        test_local_ids(
            topology,
            Dict(
                (1, 1, 0) => 1,
                (1, 2, 0) => 2,
                (2, 3, 0) => 1,
                (2, 2, 0) => 2,
                (3, 3, 0) => 1,
                (3, 4, 0) => 2,
            ),
        )
        test_neighbours(
            topology,
            0,
            Dict(
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, false) => [2; 2; 0; 1;;],
                (2, 1, false) => [3; 1; 0; 1;;],
                (2, 2, false) => [1; 2; 0; 1;;],
                (3, 1, false) => [2; 1; 0; 1;;],
                (3, 2, false) => Matrix{Int}(undef, 4, 0),
                (1, 1, true) => [1; 1; 0; 1;;],
                (1, 2, true) => [1 2; 2 2; 0 0; 1 1],
                (2, 1, true) => [2 3; 1 1; 0 0; 1 1],
                (2, 2, true) => [1 2; 2 2; 0 0; 1 1],
                (3, 1, true) => [2 3; 1 1; 0 0; 1 1],
                (3, 2, true) => [3; 2; 0; 1;;],
            ),
        )

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [(0, 1), (0, 4)]
        @test interfaces == [(0, 2), (0, 3)]

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    @testset "periodic: 1 --> 2 --> 3 --> 1" begin
        topology = Topology.MeshTopology([(1, 2), (2, 3), (3, 1)], Topology.LINE)

        # A periodic mesh has as many vertices as patches, and no boundary.
        @test size(topology) == (3, 3)

        test_incidence_relations(
            topology,
            Dict((1, 2) => [[1, 3], [1, 2], [2, 3]], (2, 1) => [[1, 2], [2, 3], [3, 1]]),
        )
        test_neighbours(
            topology,
            0,
            Dict(
                (1, 1, false) => [3; 2; 0; 1;;],
                (1, 2, false) => [2; 1; 0; 1;;],
                (3, 2, false) => [1; 1; 0; 1;;],
                (1, 1, true) => [1 3; 1 2; 0 0; 1 1],
                (3, 2, true) => [1 3; 1 2; 0 0; 1 1],
            ),
        )

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test isempty(boundaries)
        @test interfaces == [(0, 1), (0, 2), (0, 3)]

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    @testset "structured grid" begin
        topology = Topology.MeshTopology(line_grid(6), Topology.LINE)

        @test size(topology) == (7, 6)
        @test Topology.get_num_patches(topology) == 6

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [(0, 1), (0, 7)]
        @test length(interfaces) == 5

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end
end

############################################################################################
#                                            2D                                            #
############################################################################################
@testset "2D" begin
    # 4 -- 3 -- 6
    # |    |    |
    # 1 -- 2 -- 5
    @testset "two aligned patches" begin
        topology = Topology.MeshTopology([(1, 2, 3, 4), (2, 5, 6, 3)], Topology.QUAD)

        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_incidence_relations_dim(topology) == 3
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test Topology.get_topological_patch(topology) == Topology.QUAD
        @test size(topology) == (6, 7, 2)
        @test Topology.get_local_size(topology) == (4, 4, 1)

        test_incidence_relations(
            topology,
            Dict(
                (1, 1) => Vector{Vector{Int}}(),
                (1, 2) => [[1, 2], [1, 3, 4], [3, 5, 6], [2, 5], [4, 7], [6, 7]],
                (1, 3) => [[1], [1, 2], [1, 2], [1], [2], [2]],
                (2, 1) => [[1, 2], [1, 4], [2, 3], [2, 5], [4, 3], [3, 6], [5, 6]],
                (2, 2) => Vector{Vector{Int}}(),
                (2, 3) => [[1], [1], [1, 2], [2], [1], [2], [2]],
                (3, 1) => [[1, 2, 3, 4], [2, 5, 6, 3]],
                (3, 2) => [[2, 3, 1, 5], [3, 7, 4, 6]],
                (3, 3) => Vector{Vector{Int}}(),
            ),
        )
        test_global_ids(
            topology,
            Dict(
                (1, 2, 1, 0) => 1,
                (1, 2, 3, 0) => 3,
                (2, 2, 1, 0) => 2,
                (2, 2, 3, 0) => 6,
                (1, 2, 2, 1) => 3, # local edge 2 of patch 1 is the shared edge
                (2, 2, 1, 1) => 3, # and it is local edge 1 of patch 2
                (1, 1, 1, 0) => 1,
                (3, 1, 1, 0) => 2,
                (3, 1, 2, 0) => 3,
                (2, 0, 2, 1) => 3,
                (3, 0, 1, 2) => 1,
                (3, 0, 2, 2) => 2,
            ),
        )
        test_local_ids(
            topology,
            Dict(
                (1, 1, 0) => 1,
                (1, 3, 0) => 3,
                (2, 3, 0) => 4,
                (1, 3, 1) => 2,
                (2, 3, 1) => 1,
            ),
        )
        test_neighbours(
            topology,
            1,
            Dict(
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, false) => [2; 1; 0; 1;;],
                (2, 1, false) => [1; 2; 0; 1;;],
                (2, 4, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, true) => [1 2; 2 1; 0 0; 1 1],
                (2, 1, true) => [1 2; 2 1; 0 0; 1 1],
                (1, 3, true) => [1; 3; 0; 1;;],
            ),
        )
        test_neighbours(
            topology,
            0,
            Dict(
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, false) => [2; 1; 0; 1;;],
                (1, 3, false) => [2; 4; 0; 1;;],
                (2, 1, false) => [1; 2; 0; 1;;],
                (2, 4, false) => [1; 3; 0; 1;;],
                (1, 2, true) => [1 2; 2 1; 0 0; 1 1],
                (1, 3, true) => [1 2; 3 4; 0 0; 1 1],
            ),
        )

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        @test boundaries == [
            (1, 1), (1, 2), (1, 4), (1, 5), (1, 6), (1, 7), (0, 1), (0, 4), (0, 5), (0, 6)
        ]
        @test interfaces == [(1, 3), (0, 2), (0, 3)]
        # The two shared vertices lie on boundary edges.
        @test Topology.get_interfaces_on_boundary(topology) == [(0, 2), (0, 3)]

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    # Same mesh, but the second patch is listed starting from another corner so that it
    # traverses the shared edge in the opposite direction.
    @testset "two patches with a flipped shared edge" begin
        topology = Topology.MeshTopology([(1, 2, 3, 4), (3, 2, 5, 6)], Topology.QUAD)

        @test size(topology) == (6, 7, 2)

        test_incidence_relations(
            topology,
            Dict(
                (1, 2) => [[1, 2], [1, 3, 4], [3, 5, 6], [2, 5], [4, 7], [6, 7]],
                (1, 3) => [[1], [1, 2], [1, 2], [1], [2], [2]],
                (2, 1) => [[1, 2], [1, 4], [2, 3], [2, 5], [4, 3], [3, 6], [6, 5]],
                (2, 3) => [[1], [1], [1, 2], [2], [1], [2], [2]],
                (3, 1) => [[1, 2, 3, 4], [3, 2, 5, 6]],
                # The sign of the shared edge records that patch 2 traverses it backwards.
                (3, 2) => [[2, 3, 1, 5], [6, 4, -3, 7]],
            ),
        )
        # Signed ids must still be found by get_local_id.
        test_local_ids(topology, Dict((1, 3, 1) => 2, (2, 3, 1) => -3, (2, -3, 1) => -3))
        test_neighbours(
            topology,
            1,
            Dict(
                (1, 2, false) => [2; 3; 0; -1;;],
                (2, 3, false) => [1; 2; 0; -1;;],
                # Relative to the global definition of the edge, patch 1 agrees with it and
                # patch 2 does not.
                (1, 2, true) => [1 2; 2 3; 0 0; 1 -1],
                (2, 3, true) => [1 2; 2 3; 0 0; 1 -1],
            ),
        )
        test_neighbours(
            topology,
            0,
            Dict(
                (1, 2, false) => [2; 2; 0; 1;;],
                (1, 3, false) => [2; 1; 0; 1;;],
                (2, 1, false) => [1; 3; 0; 1;;],
                (2, 2, false) => [1; 2; 0; 1;;],
            ),
        )

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    @testset "structured grid" begin
        topology = Topology.MeshTopology(quad_grid(3, 3), Topology.QUAD)

        # 16 vertices, 24 edges and 9 patches; Euler: 16 - 24 + 9 == 1.
        @test size(topology) == (16, 24, 9)
        @test size(topology, 1) - size(topology, 2) + size(topology, 3) == 1

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        # 12 boundary edges and the 12 vertices on the boundary of the 3x3 grid.
        @test count(object -> object[1] == 1, boundaries) == 12
        @test count(object -> object[1] == 0, boundaries) == 4 # only the corners
        @test count(object -> object[1] == 1, interfaces) == 12
        @test count(object -> object[1] == 0, interfaces) == 12

        # The centre patch has a neighbour across each of its four edges; the corner
        # patches have two.
        @test size(Topology.compute_edge_neighbours(topology, 5, 1), 2) == 1
        @test sum(
            edge -> size(Topology.compute_edge_neighbours(topology, 5, edge), 2), 1:4
        ) == 4
        @test sum(
            edge -> size(Topology.compute_edge_neighbours(topology, 1, edge), 2), 1:4
        ) == 2
        # The centre vertex of the grid is shared by four patches.
        @test size(Topology.compute_vertex_neighbours(topology, 1, 3), 2) == 3

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end
end

############################################################################################
#                                            3D                                            #
############################################################################################
@testset "3D" begin
    @testset "two patches sharing a face" begin
        topology = Topology.MeshTopology(
            [(1, 2, 3, 4, 5, 6, 7, 8), (7, 6, 2, 3, 12, 11, 9, 10)], Topology.HEX
        )

        @test Topology.get_manifold_dim(topology) == 3
        @test Topology.get_incidence_relations_dim(topology) == 4
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.HEX)
        @test size(topology) == (12, 20, 11, 2)
        @test Topology.get_local_size(topology) == (8, 12, 6, 1)

        test_incidence_relations(
            topology,
            Dict(
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
                (3, 4) => [[1], [1], [1], [1, 2], [2], [2], [1], [2], [1], [2], [2]],
                (4, 1) => [[1, 2, 3, 4, 5, 6, 7, 8], [7, 6, 2, 3, 12, 11, 9, 10]],
                (4, 2) => [
                    [-15, -11, -1, -7, -13, -12, -2, 4, -8, -10, -3, -5],
                    [-17, -20, 13, -4, -18, -19, 8, 5, -6, -9, -16, -14],
                ],
                (4, 3) => [[3, 4, 2, 7, 1, 9], [8, 6, 10, 5, -4, 11]],
                (4, 4) => Vector{Vector{Int}}(),
            ),
        )
        test_global_ids(
            topology,
            Dict(
                (1, 3, 2, 2) => 4, # local face 2 of patch 1 is the shared face
                (2, 3, 5, 2) => -4, # it is local face 5 of patch 2, traversed backwards
                (1, 3, 1, 0) => 1,
                (2, 3, 1, 0) => 7,
                (4, 2, 1, 0) => 7, # the shared face, as an object of its own
                (4, 2, 2, 0) => 6,
            ),
        )
        test_local_ids(
            topology,
            Dict((1, 4, 2) => 2, (2, 4, 2) => -5, (2, -4, 2) => -5, (1, 7, 0) => 7),
        )

        # The shared face is seen by patch 1 as [2, 3, 7, 6] and by patch 2 as [7, 3, 2, 6],
        # i.e. shifted by two and traversed in the opposite direction.
        test_neighbours(
            topology,
            2,
            Dict(
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, false) => [2; 5; 2; -1;;],
                (2, 5, false) => [1; 2; 2; -1;;],
                (2, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, true) => [1 2; 2 5; 2 0; 1 -1],
                (2, 5, true) => [1 2; 2 5; 2 0; 1 -1],
            ),
        )
        test_neighbours(
            topology,
            1,
            Dict(
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 5, false) => [2; 3; 0; -1;;],
                (1, 8, false) => [2; 4; 0; -1;;],
                (1, 9, false) => [2; 7; 0; -1;;],
                (1, 12, false) => [2; 8; 0; -1;;],
                (2, 3, false) => [1; 5; 0; -1;;],
            ),
        )
        test_neighbours(
            topology,
            0,
            Dict(
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
                (1, 2, false) => [2; 3; 0; 1;;],
                (1, 3, false) => [2; 4; 0; 1;;],
                (1, 6, false) => [2; 2; 0; 1;;],
                (1, 7, false) => [2; 1; 0; 1;;],
                (2, 1, false) => [1; 7; 0; 1;;],
            ),
        )

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    @testset "two patches sharing a rotated face" begin
        topology = Topology.MeshTopology(
            [(1, 2, 3, 4, 5, 6, 7, 8), (10, 3, 2, 9, 12, 7, 6, 11)], Topology.HEX
        )

        @test size(topology) == (12, 20, 11, 2)

        test_incidence_relations(
            topology,
            Dict(
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
                (4, 1) => [[1, 2, 3, 4, 5, 6, 7, 8], [10, 3, 2, 9, 12, 7, 6, 11]],
                (4, 2) => [
                    [-15, -11, -1, -7, -13, -12, -2, -4, -8, -10, -3, -5],
                    [-14, -16, -9, -6, 13, -20, -17, 4, -5, -18, -19, -8],
                ],
                (4, 3) => [[3, 4, 2, 7, 1, 9], [11, -4, 8, 6, 5, 10]],
            ),
        )
        test_neighbours(
            topology,
            2,
            Dict(
                (1, 2, false) => [2; 2; 1; -1;;],
                (2, 2, false) => [1; 2; 1; -1;;],
                (1, 1, false) => Matrix{Int}(undef, 4, 0),
            ),
        )
        test_neighbours(
            topology,
            1,
            Dict(
                (1, 5, false) => [2; 5; 0; -1;;],
                (1, 8, false) => [2; 8; 0; -1;;],
                (1, 9, false) => [2; 12; 0; 1;;],
                (1, 12, false) => [2; 9; 0; 1;;],
            ),
        )

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    # Two hexahedra glued along local face 2 of the first one. The four connectivities are
    # obtained by rotating the second hexahedron about the shared face, and therefore cover
    # every rotation that can be reported. The orientation is always -1: two hexahedra on
    # opposite sides of a face traverse it in opposite cyclic directions.
    @testset "shared face with rotation $rotation" for (rotation, patch_2, local_face) in (
        (0, (2, 9, 10, 3, 6, 11, 12, 7), 1),
        (1, (10, 3, 2, 9, 12, 7, 6, 11), 2),
        (2, (7, 12, 11, 6, 3, 10, 9, 2), 1),
        (3, (11, 6, 7, 12, 9, 2, 3, 10), 2),
    )
        topology = Topology.MeshTopology([(1, 2, 3, 4, 5, 6, 7, 8), patch_2], Topology.HEX)

        @test size(topology) == (12, 20, 11, 2)

        neighbours = Topology.compute_face_neighbours(topology, 1, 2)
        @test neighbours == [2; local_face; rotation; -1;;]

        # Seen from the neighbour, the same shift has to be undone.
        back = Topology.compute_face_neighbours(topology, 2, local_face)
        @test back[1, 1] == 1
        @test back[2, 1] == 2
        @test back[4, 1] == -1
        # A reversed traversal is its own inverse, so both patches report the same shift:
        # if B[k] = A[r - k] then A[k] = B[r - k].
        @test back[3, 1] == rotation

        # Relative to the global face the two patches must disagree on the direction.
        with_self = Topology.compute_face_neighbours(
            topology, 1, 2; include_local_patch=true
        )
        @test sort(with_self[4, :]) == [-1, 1]

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end

    @testset "structured grid" begin
        topology = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX)

        # 27 vertices, 54 edges, 36 faces and 8 patches; Euler: 27 - 54 + 36 - 8 == 1.
        @test size(topology) == (27, 54, 36, 8)
        @test size(topology, 1) - size(topology, 2) + size(topology, 3) -
              size(topology, 4) == 1

        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        # 6 sides of 4 faces each are on the boundary; the remaining 12 are shared.
        @test count(object -> object[1] == 2, boundaries) == 24
        @test count(object -> object[1] == 2, interfaces) == 12
        # Only the 8 corner vertices belong to a single patch.
        @test count(object -> object[1] == 0, boundaries) == 8

        # Every patch of a 2x2x2 grid is a corner patch: 3 face, 3 edge and 1 vertex
        # neighbours, sharing the centre vertex with all 7 others.
        for patch_id in 1:8
            @test sum(
                face -> size(Topology.compute_face_neighbours(topology, patch_id, face), 2),
                1:6,
            ) == 3
        end
        # Every shared face is traversed in opposite directions by the two patches that
        # share it, since all patches of the grid are consistently oriented.
        for patch_id in 1:8, face in 1:6
            face_neighbours = Topology.compute_face_neighbours(topology, patch_id, face)
            @test all(==(-1), face_neighbours[4, :])
        end

        # The centre vertex is local vertex 7 of the first patch.
        @test size(Topology.compute_vertex_neighbours(topology, 1, 7), 2) == 7
        @test size(
            Topology.compute_vertex_neighbours(topology, 1, 7; include_local_patch=true), 2
        ) == 8

        test_structural_invariants(topology)
        test_neighbour_invariants(topology)
    end
end

############################################################################################
#                                     SkeletonTopology                                     #
############################################################################################
@testset "SkeletonTopology" begin
    @testset "of a 3D mesh" begin
        parent = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX)
        skeleton = Topology.SkeletonTopology(parent)

        @test Topology.get_manifold_dim(skeleton) == 2
        @test Topology.get_incidence_relations_dim(skeleton) == 3
        # The patches of the skeleton are the faces of the parent.
        @test Topology.get_num_patches(skeleton) == size(parent, 3)
        @test Topology.get_parent_topology(skeleton) === parent
        @test Topology.get_topological_patch(skeleton) == Topology.QUAD
        @test Topology.get_patch_type(skeleton) == typeof(Topology.QUAD)

        @test size(skeleton) == size(parent)[1:3]
        @test size(skeleton, 3) == size(parent, 3)
        # The skeleton patches are quadrilaterals, not hexahedra.
        @test Topology.get_local_size(skeleton) == (4, 4, 1)
        @test Topology.get_local_size(skeleton, 2) == 4

        @testset "relation ($i, $k)" for i in 1:3, k in 1:3
            @test skeleton[i, k] === parent[i, k]
        end

        @testset "patch $patch_id parents" for patch_id in 1:size(parent, 3)
            parents = Topology.get_patch_parents(skeleton, patch_id)
            @test size(parents, 1) == 4
            # A face is shared by at most two hexahedra.
            @test size(parents, 2) == length(parent[3, 4][patch_id])
            @test sort(parents[1, :]) == sort(parent[3, 4][patch_id])

            # The reported rotation and orientation reproduce how each parent patch
            # traverses the face.
            face_vertices = parent[3, 1][patch_id]
            for column in axes(parents, 2)
                parent_id, local_id, rotation, orientation = parents[:, column]
                @test reconstruct_vertices(face_vertices, rotation, orientation) ==
                    Topology._object_vertices_in_patch(parent, parent_id, local_id, 2)
            end
        end
    end

    @testset "of a 2D mesh" begin
        parent = Topology.MeshTopology(quad_grid(2, 2), Topology.QUAD)
        skeleton = Topology.SkeletonTopology(parent)

        @test Topology.get_manifold_dim(skeleton) == 1
        @test Topology.get_num_patches(skeleton) == size(parent, 2)
        @test Topology.get_topological_patch(skeleton) == Topology.LINE
        @test size(skeleton) == size(parent)[1:2]
        @test Topology.get_local_size(skeleton) == (2, 1)

        @testset "patch $patch_id parents" for patch_id in 1:size(parent, 2)
            parents = Topology.get_patch_parents(skeleton, patch_id)
            @test sort(parents[1, :]) == sort(parent[2, 3][patch_id])
            @test all(iszero, parents[3, :]) # edges do not rotate

            edge_vertices = parent[2, 1][patch_id]
            for column in axes(parents, 2)
                parent_id, local_id, rotation, orientation = parents[:, column]
                @test reconstruct_vertices(edge_vertices, rotation, orientation) ==
                    Topology._object_vertices_in_patch(parent, parent_id, local_id, 1)
            end
        end
    end

    @testset "of a 1D mesh" begin
        # The skeleton of a 1D mesh would be a set of points, which is not a topology.
        parent = Topology.MeshTopology(line_grid(3), Topology.LINE)
        @test_throws ArgumentError Topology.SkeletonTopology(parent)
    end
end

############################################################################################
#                                     Argument errors                                      #
############################################################################################
@testset "argument errors" begin
    line_topology = Topology.MeshTopology(line_grid(3), Topology.LINE)
    quad_topology = Topology.MeshTopology(quad_grid(2, 2), Topology.QUAD)
    hex_topology = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX)

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
        @test_throws ArgumentError Topology.compute_edge_neighbours(line_topology, 1, 1)
        @test_throws ArgumentError Topology.compute_edge_neighbours(line_topology)
        @test_throws ArgumentError Topology.compute_face_neighbours(quad_topology, 1, 1)
        @test_throws ArgumentError Topology.compute_face_neighbours(quad_topology)
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

        skeleton = Topology.SkeletonTopology(hex_topology)
        @test_throws BoundsError skeleton[4, 1]
        @test_throws BoundsError skeleton[1, 4]
        # Checking the bounds of size with 4 will trigger the boundscheck for the skeleton.
        # If this is ever removed, this test will fail, because 4 is not out-of-bounds for
        # the parent (size(skeleton) calls size on the parent).
        @test_throws BoundsError size(skeleton, 4)
    end
end

############################################################################################
#                                      Type stability                                      #
############################################################################################
@testset "type stability" begin
    line_topology = Topology.MeshTopology(line_grid(3), Topology.LINE)
    quad_topology = Topology.MeshTopology(quad_grid(2, 2), Topology.QUAD)
    hex_topology = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX)
    skeleton = Topology.SkeletonTopology(hex_topology)

    @testset "$name" for (name, topology) in (
        "1D" => line_topology,
        "2D" => quad_topology,
        "3D" => hex_topology,
        "skeleton" => skeleton,
    )
        manifold_dim = Topology.get_manifold_dim(topology)

        @test (@inferred Topology.get_manifold_dim(topology)) == manifold_dim
        @test (@inferred size(topology)) isa NTuple{<:Any, Int}
        @test (@inferred size(topology, 1)) isa Int
        @test (@inferred Topology.get_local_size(topology)) isa NTuple{<:Any, Int}
        @test (@inferred Topology.get_local_size(topology, 1)) isa Int
        @test (@inferred topology[manifold_dim + 1, 1]) isa Vector{Vector{Int}}
        @test (@inferred Topology.get_global_id(topology, 1, 1, 0)) isa Int
        @test (@inferred Topology.get_local_id(topology, 1, 1, 0)) isa Int
        @test (@inferred Topology.compute_vertex_neighbours(topology, 1, 1)) isa Matrix{Int}
        @test (@inferred Topology.compute_vertex_neighbours(topology)) isa
            Matrix{Matrix{Int}}
        @test (@inferred Topology.get_boundaries_and_interfaces(topology)) isa
            Tuple{Vector{Tuple{Int, Int}}, Vector{Tuple{Int, Int}}}
        @test (@inferred Topology.get_interfaces_on_boundary(topology)) isa
            Vector{Tuple{Int, Int}}

        if manifold_dim >= 2
            @test (@inferred Topology.compute_edge_neighbours(topology, 1, 1)) isa
                Matrix{Int}
            @test (@inferred Topology.compute_edge_neighbours(topology)) isa
                Matrix{Matrix{Int}}
        end
        if manifold_dim == 3
            @test (@inferred Topology.compute_face_neighbours(topology, 1, 1)) isa
                Matrix{Int}
            @test (@inferred Topology.compute_face_neighbours(topology)) isa
                Matrix{Matrix{Int}}
        end
    end

    # Construction itself cannot be inferred, because the number of patches is only known
    # at run time yet is part of the type. Only the manifold dimension, which follows from
    # the topological patch, is inferable.
    @test Base.infer_return_type(
        Topology.MeshTopology, Tuple{Vector{NTuple{2, Int}}, Topology.Line}
    ) <: Topology.MeshTopology{1, 2}
    @test Base.infer_return_type(
        Topology.MeshTopology, Tuple{Vector{NTuple{4, Int}}, Topology.Quad}
    ) <: Topology.MeshTopology{2, 3}
    @test Base.infer_return_type(
        Topology.MeshTopology, Tuple{Vector{NTuple{8, Int}}, Topology.Hex}
    ) <: Topology.MeshTopology{3, 4}
end

end

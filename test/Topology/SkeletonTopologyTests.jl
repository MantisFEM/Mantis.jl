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

@testset "of a 1D mesh" begin
    # The skeleton of a 1D mesh would be a set of points, which is not a topology.
    parent = Topology.MeshTopology(line_grid(3), Topology.LINE, Val(3))
    @test_throws ArgumentError Topology.SkeletonTopology(parent)
end

@testset "of a 2D mesh" begin
    parent = Topology.MeshTopology(quad_grid(2, 2), Topology.QUAD, Val(4))
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

@testset "of a 3D mesh" begin
    parent = Topology.MeshTopology(hex_grid(2, 2, 2), Topology.HEX, Val(8))
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

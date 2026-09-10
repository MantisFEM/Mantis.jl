module TensorProductTopologyTestModule

using Test
using Mantis

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
#                                       Test meshes                                        #
############################################################################################
segment = Topology.MeshTopology(((1, 2),), Topology.LINE)
chain = Topology.MeshTopology(line_grid(2), Topology.LINE, Val(2))
# The middle patch is traversed against the direction of its neighbours.
flipped_chain = Topology.MeshTopology(((1, 2), (3, 2), (3, 4)), Topology.LINE)
ring = Topology.MeshTopology(((1, 2), (2, 3), (3, 1)), Topology.LINE)
quads = Topology.MeshTopology(quad_grid(2, 1), Topology.QUAD, Val(2))
# The second patch traverses the shared edge in the opposite direction, so that the faces of
# its products are rotated with respect to those of the first patch.
rotated_quads = Topology.MeshTopology(((1, 2, 3, 4), (3, 2, 5, 6)), Topology.QUAD)


struct NonTensorProductPatch <: Topology.AbstractPatch{1, 2, 2} end
struct NonTensorProductTopology <: Topology.AbstractTopology{1, 2, 1, NonTensorProductPatch}
    topological_patch::NonTensorProductPatch
end

############################################################################################
#                                     Argument errors                                      #
############################################################################################
@testset "Argument errors" begin
    @test_throws ArgumentError Topology.tensor_product_topology()
    @test_throws ArgumentError Topology.tensor_product_topology(())
    # Topologies are limited to manifold dimension 3.
    @test_throws ArgumentError Topology.tensor_product_topology(quads, quads)
    @test_throws ArgumentError Topology.tensor_product_topology(chain, chain, quads)
    hexes = Topology.MeshTopology(hex_grid(1, 1, 1), Topology.HEX, Val(1))
    @test_throws ArgumentError Topology.tensor_product_topology(hexes, segment)
    # The factors must be made of tensor-product patches.
    @test_throws ArgumentError Topology.tensor_product_topology(
        NonTensorProductTopology(NonTensorProductPatch()), segment
    )
end

############################################################################################
#                                      Explicit cases                                      #
############################################################################################
@testset "Explicit cases" verbose = true begin
    @testset "1D ⊗ 1D: patches and vertices" begin
        topology = Topology.tensor_product_topology(chain, segment)
        @test topology isa Topology.MeshTopology
        @test Topology.get_manifold_dim(topology) == 2
        @test Topology.get_num_patches(topology) == 2
        @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
        @test size(topology) == (6, 7, 2)
        # Vertex (i, j) of chain ⊗ segment has id i + 3(j - 1); the vertices of a patch are
        # listed in the local order of a quadrilateral.
        @test topology[3, 1] == [[1, 2, 5, 4], [2, 3, 6, 5]]
    end

    @testset "patch numbering: first factor varies fastest" begin
        topology = Topology.tensor_product_topology(chain, ring)
        # chain has 2 patches and ring has 3; the vertices of the product are numbered with
        # the vertices of chain varying fastest (3 per ring vertex).
        @test Topology.get_num_patches(topology) == 6
        for (patch_id, factor_patch_ids) in enumerate(CartesianIndices((2, 3)))
            chain_vertices = chain[2, 1][factor_patch_ids[1]]
            ring_vertices = ring[2, 1][factor_patch_ids[2]]
            expected = [
                chain_vertices[1] + 3 * (ring_vertices[1] - 1),
                chain_vertices[2] + 3 * (ring_vertices[1] - 1),
                chain_vertices[2] + 3 * (ring_vertices[2] - 1),
                chain_vertices[1] + 3 * (ring_vertices[2] - 1),
            ]
            @test topology[3, 1][patch_id] == expected
        end
    end

    @testset "one factor" begin
        @test Topology.tensor_product_topology(chain) == chain
        @test Topology.tensor_product_topology(quads) == quads
        @test Topology.tensor_product_topology((chain,)) == chain
    end
end

############################################################################################
#                                Equal to structured meshes                                #
############################################################################################
@testset "Equal to structured meshes" begin
    @test Topology.tensor_product_topology(chain, segment) ==
        Topology.MeshTopology(quad_grid(2, 1), Topology.QUAD, Val(2))
    @test Topology.tensor_product_topology(chain, chain) ==
        Topology.MeshTopology(quad_grid(2, 2), Topology.QUAD, Val(4))
    @test Topology.tensor_product_topology(chain, chain, segment) ==
        Topology.MeshTopology(hex_grid(2, 2, 1), Topology.HEX, Val(4))
    @test Topology.tensor_product_topology(quads, segment) ==
        Topology.MeshTopology(hex_grid(2, 1, 1), Topology.HEX, Val(2))
end

############################################################################################
#                                        Properties                                        #
############################################################################################
@testset "Properties of mixed products" verbose = true begin
    products = Dict(
        "1D ⊗ 1D" => (chain, flipped_chain),
        "1D ⊗ 2D" => (ring, quads),
        "2D ⊗ 1D" => (rotated_quads, chain),
        "1D ⊗ 1D ⊗ 1D" => (chain, ring, flipped_chain),
        "skeleton ⊗ 1D" => (Topology.SkeletonTopology(quads), segment),
    )
    @testset "$name" for (name, factors) in products
        topology = Topology.tensor_product_topology(factors)
        manifold_dim = Topology.get_manifold_dim(topology)
        @test manifold_dim == sum(Topology.get_manifold_dim, factors)
        @test Topology.get_num_patches(topology) == prod(Topology.get_num_patches, factors)
        @test size(topology, 1) == prod(f -> size(f, 1), factors)
        @test Topology.get_topological_patch(topology) ==
            Topology.tensor_product_patch(Val(manifold_dim))

        # Every patch is made of the vertices of the factor patches.
        factor_patches = CartesianIndices(map(Topology.get_num_patches, factors))
        vertex_ids = LinearIndices(map(f -> size(f, 1), factors))
        for (patch_id, factor_patch_ids) in enumerate(factor_patches)
            product_vertices = Set(topology[manifold_dim + 1, 1][patch_id])
            factor_vertices = map(
                (f, id) -> abs.(f[Topology.get_manifold_dim(f) + 1, 1][id]),
                factors,
                Tuple(factor_patch_ids),
            )
            expected = Set(
                vertex_ids[ids...] for ids in Iterators.product(factor_vertices...)
            )
            @test product_vertices == expected
        end

        # The number of objects per patch is that of the tensor-product patch.
        @test Topology.get_local_size(topology) ==
            size(Topology.tensor_product_patch(Val(manifold_dim)))
        for dim in 0:(manifold_dim - 1)
            @test all(
                length(row) == Topology.get_local_size(topology, dim + 1) for
                row in topology[manifold_dim + 1, dim + 1]
            )
        end

        # Containing objects are the transposed relation.
        for i in 1:(manifold_dim + 1), k in (i + 1):(manifold_dim + 1)
            transposed = [Int[] for _ in 1:size(topology, i)]
            for container_id in 1:size(topology, k),
                target_id in topology[k, i][container_id]

                push!(transposed[abs(target_id)], container_id)
            end
            @test sort!.(unique.(transposed)) == sort!.(unique.(topology[i, k]))
        end
    end

    @testset "nested products equal flat products" begin
        nested = Topology.tensor_product_topology(
            Topology.tensor_product_topology(chain, ring), flipped_chain
        )
        flat = Topology.tensor_product_topology(chain, ring, flipped_chain)
        @test nested == flat
    end

    @testset "neighbours" begin
        topology = Topology.tensor_product_topology(rotated_quads, flipped_chain)
        manifold_dim = Topology.get_manifold_dim(topology)
        for object_dim in 0:(manifold_dim - 1), include_local_patch in (false, true)
            all_neighbours = Topology.compute_neighbours(
                topology, object_dim, include_local_patch
            )
            @test size(all_neighbours) == (
                Topology.get_num_patches(topology),
                Topology.get_local_size(topology, object_dim + 1),
            )
        end
    end
end

end

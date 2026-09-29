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
    equivalent_mesh_topology(topology)

Return the [`Topology.MeshTopology`](@ref) built by `MeshCore` from the patch-to-vertex
relation of `topology`.

Both topologies share the numbering of their vertices and patches, but not that of the
objects in between, which each numbers its own way. The tests below therefore only compare
quantities that are independent of that numbering.
"""
function equivalent_mesh_topology(@nospecialize(topology))
    manifold_dim = Topology.get_manifold_dim(topology)
    patches = [
        Tuple(topology[manifold_dim + 1, 1][p]) for p in 1:size(topology, manifold_dim + 1)
    ]
    return Topology.MeshTopology(
        patches, Topology.get_topological_patch(topology), Val(length(patches))
    )
end

"""
    vertex_set(topology, object_dim, object_id)

Return the sorted global vertex ids of object `object_id` of dimension `object_dim`, which
identifies the object independently of how `topology` numbers it.
"""
function vertex_set(@nospecialize(topology), object_dim, object_id)
    object_dim == 0 && return [object_id]
    return sort(topology[object_dim + 1, 1][object_id])
end

"""
    vertex_sets(topology, object_dim)

Return the sorted list of the vertex sets of all objects of dimension `object_dim`.
"""
function vertex_sets(@nospecialize(topology), object_dim)
    return sort([
        vertex_set(topology, object_dim, id) for id in 1:size(topology, object_dim + 1)
    ])
end

"""
    test_against_mesh_topology(topology)

Check `topology` against the [`Topology.MeshTopology`](@ref) that `MeshCore` builds from its
patches: both must describe the same objects, the same local numbering within every patch,
the same neighbours, and the same boundaries and interfaces.
"""
function test_against_mesh_topology(@nospecialize(topology))
    reference = equivalent_mesh_topology(topology)
    manifold_dim = Topology.get_manifold_dim(topology)
    num_patches = size(topology, manifold_dim + 1)

    @testset "same objects" begin
        @test size(topology) == size(reference)
        for dim in 1:(manifold_dim - 1)
            @test vertex_sets(topology, dim) == vertex_sets(reference, dim)
        end
    end

    @testset "same local numbering" begin
        # The objects of each patch are listed in the local order of the patch: the vertices
        # of the object in each local slot are the ones the topological patch prescribes.
        for dim in 1:(manifold_dim - 1), patch_id in 1:num_patches
            for local_id in 1:Topology.get_local_size(topology, dim + 1)
                global_id = Topology.get_global_id(topology, patch_id, local_id, dim)
                local_vertices = Topology._object_vertices_in_patch(
                    topology, patch_id, local_id, dim
                )
                @test vertex_set(topology, dim, global_id) == sort(local_vertices)
                @test Topology.get_local_id(topology, patch_id, global_id, dim) == local_id
            end
        end
    end

    @testset "same neighbours" begin
        # Neighbours are described relative to the current patch, so they do not depend on
        # how the shared objects are numbered.
        for dim in 0:(manifold_dim - 1), patch_id in 1:num_patches
            for local_id in 1:Topology.get_local_size(topology, dim + 1)
                @test Topology.compute_neighbours(topology, patch_id, local_id, dim) ==
                    Topology.compute_neighbours(reference, patch_id, local_id, dim)

                # Relative to the global definition of the object, the rotation and
                # orientation depend on the global vertex order, which each topology chooses
                # its own way. Check the patches and local ids, and that the rotation and
                # orientation reproduce how each patch traverses the object.
                neighbours = Topology.compute_neighbours(
                    topology, patch_id, local_id, dim, true
                )
                reference_neighbours = Topology.compute_neighbours(
                    reference, patch_id, local_id, dim, true
                )
                @test neighbours[1:2, :] == reference_neighbours[1:2, :]
                dim == 0 && continue
                global_id = Topology.get_global_id(topology, patch_id, local_id, dim)
                global_vertices = topology[dim + 1, 1][global_id]
                for column in axes(neighbours, 2)
                    neighbour_id, neighbour_local_id, rotation, orientation = neighbours[
                        :, column
                    ]
                    @test reconstruct_vertices(global_vertices, rotation, orientation) ==
                        Topology._object_vertices_in_patch(
                        topology, neighbour_id, neighbour_local_id, dim
                    )
                end
            end
        end
    end

    @testset "same boundaries and interfaces" begin
        boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
        reference_boundaries, reference_interfaces = Topology.get_boundaries_and_interfaces(
            reference
        )
        to_vertex_sets(t, objects) =
            sort([(dim, vertex_set(t, dim, id)) for (dim, id) in objects])
        @test to_vertex_sets(topology, boundaries) ==
            to_vertex_sets(reference, reference_boundaries)
        @test to_vertex_sets(topology, interfaces) ==
            to_vertex_sets(reference, reference_interfaces)
        @test to_vertex_sets(topology, Topology.get_interfaces_on_boundary(topology)) ==
            to_vertex_sets(reference, Topology.get_interfaces_on_boundary(reference))
    end
end

"""
    test_structural_invariants(topology)

Check the properties that must hold for the rows of any tensor-product topology, beyond what
can be compared with `MeshCore`: every object lists the objects it contains in the local
order of the tensor-product patch of its dimension, and the relations listing contained and
containing objects are each other's transposes.
"""
function test_structural_invariants(@nospecialize(topology))
    manifold_dim = Topology.get_manifold_dim(topology)

    @testset "rows follow the local order of the object's patch" begin
        for container_dim in 1:manifold_dim
            patch = Topology.tensor_product_patch(Val(container_dim))
            for container_id in 1:size(topology, container_dim + 1)
                container_vertices = topology[container_dim + 1, 1][container_id]
                for target_dim in 0:(container_dim - 1)
                    row = topology[container_dim + 1, target_dim + 1][container_id]
                    @test length(row) == size(patch, target_dim + 1)
                    for (local_id, target_id) in enumerate(row)
                        # A local vertex is itself; other objects list their vertices.
                        local_vertices = if target_dim == 0
                            [local_id]
                        else
                            patch[target_dim + 1, 1][local_id]
                        end
                        @test vertex_set(topology, target_dim, target_id) ==
                            sort(container_vertices[local_vertices])
                    end
                end
            end
        end
    end

    @testset "containing objects are the transposed relation" begin
        for i in 1:(manifold_dim + 1), k in (i + 1):(manifold_dim + 1)
            transposed = [Int[] for _ in 1:size(topology, i)]
            for container_id in 1:size(topology, k),
                target_id in topology[k, i][container_id]

                push!(transposed[target_id], container_id)
            end
            @test collect(topology[i, k]) == sort!.(transposed)
        end
    end
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

############################################################################################
#                                     Argument errors                                      #
############################################################################################
struct NonTensorProductPatch <: Topology.AbstractPatch{1, 2, 2} end
struct NonTensorProductTopology <: Topology.AbstractTopology{1, 2, 1, NonTensorProductPatch}
    topological_patch::NonTensorProductPatch
end

@testset "Argument errors" begin
    @testset "construction" begin
        @test_throws ArgumentError Topology.TensorProductTopology()
        @test_throws ArgumentError Topology.TensorProductTopology(())
        # Topologies are limited to manifold dimension 3.
        @test_throws ArgumentError Topology.TensorProductTopology(quads, quads)
        @test_throws ArgumentError Topology.TensorProductTopology(chain, chain, quads)
        hexes = Topology.MeshTopology(hex_grid(1, 1, 1), Topology.HEX, Val(1))
        @test_throws ArgumentError Topology.TensorProductTopology(hexes, segment)
        # The factors must be made of tensor-product patches.
        @test_throws ArgumentError Topology.TensorProductTopology(
            NonTensorProductTopology(NonTensorProductPatch()), segment
        )
    end

    @testset "indexing" begin
        topology = Topology.TensorProductTopology(chain, segment)
        @test_throws BoundsError topology[0, 1]
        @test_throws BoundsError topology[1, 4]
        @test_throws BoundsError topology[3, 1][3]
        @test_throws BoundsError topology[1, 2][0]
        # Objects are not related to the objects of their own dimension.
        @test_throws BoundsError topology[2, 2][1]
    end

    @testset "factor ids" begin
        topology = Topology.TensorProductTopology(chain, segment)
        @test_throws BoundsError Topology.get_factor_object_ids(topology, 3, 1)
        @test_throws BoundsError Topology.get_factor_object_ids(topology, 1, 8)
        @test_throws BoundsError Topology.get_factor_object_ids(topology, 0, 0)
        # A 1D factor has no objects of dimension 2.
        @test_throws ArgumentError Topology.get_product_object_id(topology, (2, 0), (1, 1))
        @test_throws ArgumentError Topology.get_product_object_id(topology, (2, 1), (1, 1))
        @test_throws BoundsError Topology.get_product_object_id(topology, (1, 1), (3, 1))
    end
end

############################################################################################
#                                      Explicit cases                                      #
############################################################################################
@testset "two patches: (1 --> 2 --> 3) ⊗ (1 --> 2)" verbose=true begin
    # Two quadrilaterals side by side, with the vertices of the product numbered
    # lexicographically (the first factor varying fastest):
    #
    #   4 --- 5 --- 6          · -3- · -4- ·
    #   |     |     |          5     6     7
    #   1 --- 2 --- 3          · -1- · -2- ·
    #
    # The (edge ⊗ vertex) edges come first, then the (vertex ⊗ edge) ones.
    topology = Topology.TensorProductTopology(chain, segment)

    @test Topology.get_manifold_dim(topology) == 2
    @test Topology.get_incidence_relations_dim(topology) == 3
    @test Topology.get_num_patches(topology) == 2
    @test Topology.get_patch_type(topology) == typeof(Topology.QUAD)
    @test Topology.get_topological_patch(topology) == Topology.QUAD
    @test Topology.get_factor_topologies(topology) === (chain, segment)
    @test size(topology) == (6, 7, 2)
    @test @expected Dict(1 => 6, 2 => 7, 3 => 2) k -> size(topology, k)
    @test Topology.get_local_size(topology) == (4, 4, 1)
    @test lastindex(topology) == 3

    expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
        (1, 1) => Vector{Vector{Int}}(),
        (1, 2) => [[1, 5], [1, 2, 6], [2, 7], [3, 5], [3, 4, 6], [4, 7]],
        (1, 3) => [[1], [1, 2], [2], [1], [1, 2], [2]],
        (2, 1) => [[1, 2], [2, 3], [4, 5], [5, 6], [1, 4], [2, 5], [3, 6]],
        (2, 2) => Vector{Vector{Int}}(),
        (2, 3) => [[1], [2], [1], [2], [1], [1, 2], [2]],
        # Counter-clockwise, as for the topological patch.
        (3, 1) => [[1, 2, 5, 4], [2, 3, 6, 5]],
        # Local edges at the positions (-1, 0), (1, 0), (0, -1) and (0, 1).
        (3, 2) => [[5, 6, 1, 3], [6, 7, 2, 4]],
        (3, 3) => Vector{Vector{Int}}(),
    )
    @test @expected expected_ir k -> topology[k[1], k[2]]

    expected_factor_ids = Dict(
        # (object_dim, object_id) => (object_dims, factor_object_ids)
        (0, 1) => ((0, 0), (1, 1)),
        (0, 5) => ((0, 0), (2, 2)),
        (1, 2) => ((1, 0), (2, 1)),
        (1, 3) => ((1, 0), (1, 2)),
        (1, 5) => ((0, 1), (1, 1)),
        (1, 7) => ((0, 1), (3, 1)),
        (2, 2) => ((1, 1), (2, 1)),
    )
    @test @expected expected_factor_ids k -> Topology.get_factor_object_ids(topology, k...)
    @test @expected Dict(v => k[2] for (k, v) in expected_factor_ids) k ->
        Topology.get_product_object_id(topology, k...)

    expected_global_ids = Dict{NTuple{3, Int}, Int}(
        # (patch_id, local_id, local_dim)
        (1, 3, 0) => 5,
        (2, 1, 0) => 2,
        (1, 2, 1) => 6,
        (2, 1, 1) => 6,
        (2, 4, 1) => 4,
    )
    @test @expected expected_global_ids k -> Topology.get_global_id(topology, k...)
    @test Topology.get_global_id(topology, 6, 1, 1, 0) == 2 # 1st vertex of edge 6.
    @test Topology.get_global_id(topology, 6, 1, 2, 2) == 2 # 2nd patch of edge 6.

    expected_local_ids = Dict{NTuple{3, Int}, Int}(
        # (patch_id, global_object_id, object_dim)
        (1, 5, 0) => 3,
        (2, 5, 0) => 4,
        (1, 6, 1) => 2,
        (2, 6, 1) => 1,
        (2, 2, 1) => 3,
    )
    @test @expected expected_local_ids k -> Topology.get_local_id(topology, k...)

    expected_neighbours = Dict{Tuple{Int, Int, Int, Bool}, Matrix{Int}}(
        # (patch_id, local_id, object_dim, include_local_patch)
        (1, 2, 0, false) => [2; 1; 0; 1;;],
        (1, 3, 0, false) => [2; 4; 0; 1;;],
        (1, 1, 0, false) => Matrix{Int}(undef, 4, 0),
        (1, 2, 0, true) => [1 2; 2 1; 0 0; 1 1],
        (1, 2, 1, false) => [2; 1; 0; 1;;],
        (2, 1, 1, false) => [1; 2; 0; 1;;],
        (1, 1, 1, false) => Matrix{Int}(undef, 4, 0),
        (1, 2, 1, true) => [1 2; 2 1; 0 0; 1 1],
    )
    @test @expected expected_neighbours k -> Topology.compute_neighbours(topology, k...)

    boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
    @test boundaries ==
        [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 7), (0, 1), (0, 3), (0, 4), (0, 6)]
    @test interfaces == [(1, 6), (0, 2), (0, 5)]
    @test Topology.get_interfaces_on_boundary(topology) == [(0, 2), (0, 5)]
end

@testset "single periodic patch: (1 --> 1) ⊗ (1 --> 2)" verbose=true begin
    # A cylinder made of one patch, whose left and right edges are the same edge (the seam).
    periodic_patch = Topology.MeshTopology(((1, 1),), Topology.LINE)
    topology = Topology.TensorProductTopology(periodic_patch, segment)

    @test size(topology) == (2, 3, 1)
    expected_ir = Dict{NTuple{2, Int}, Vector{Vector{Int}}}(
        (3, 1) => [[1, 1, 2, 2]],
        (3, 2) => [[3, 3, 1, 2]],
        (2, 1) => [[1, 1], [2, 2], [1, 2]],
        (2, 3) => [[1], [1], [1, 1]],
    )
    @test @expected expected_ir k -> topology[k[1], k[2]]

    # The seam is shared by the patch with itself, so it is an interface.
    boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
    @test boundaries == [(1, 1), (1, 2)]
    @test interfaces == [(1, 3), (0, 1), (0, 2)]
end

############################################################################################
#                                  Numbering of objects                                    #
############################################################################################
@testset "Numbering" verbose=true begin
    products = Dict(
        "1D ⊗ 1D" => (chain, flipped_chain),
        "1D ⊗ 2D" => (ring, quads),
        "2D ⊗ 1D" => (rotated_quads, chain),
        "1D ⊗ 1D ⊗ 1D" => (chain, ring, flipped_chain),
    )
    @testset "$name" for (name, factors) in products
        topology = Topology.TensorProductTopology(factors)
        manifold_dim = Topology.get_manifold_dim(topology)

        # Patches are numbered as the elements of a TensorProduct, which is how a
        # TensorProductGeometry numbers them.
        factor_patches = CartesianIndices(map(Topology.get_num_patches, factors))
        @test Topology.get_num_patches(topology) == length(factor_patches)
        @test all(1:length(factor_patches)) do patch_id
            object_dims, factor_ids = Topology.get_factor_object_ids(
                topology, manifold_dim, patch_id
            )
            return object_dims == map(Topology.get_manifold_dim, factors) &&
                   factor_ids == Tuple(factor_patches[patch_id])
        end

        # Every object is the product of exactly one combination of factor objects.
        for dim in 0:manifold_dim
            combinations = [
                Topology.get_factor_object_ids(topology, dim, id) for
                id in 1:size(topology, dim + 1)
            ]
            @test allunique(combinations)
            @test all(sum(object_dims) == dim for (object_dims, _) in combinations)
            @test all(eachindex(combinations)) do id
                return Topology.get_product_object_id(topology, combinations[id]...) == id
            end
        end

        # The size of each dimension is the number of combinations of factor objects whose
        # dimensions add up to it.
        expected_size = ntuple(manifold_dim + 1) do dim_id
            return sum(
                Iterators.product(map(f -> 0:Topology.get_manifold_dim(f), factors)...)
            ) do dims
                sum(dims) == dim_id - 1 || return 0
                return prod(map((f, d) -> size(f, d + 1), factors, dims))
            end
        end
        @test size(topology) == expected_size

        # Every product patch is a tensor-product patch with the expected number of objects.
        @test Topology.get_local_size(topology) ==
            size(Topology.tensor_product_patch(Val(manifold_dim)))
    end

    @testset "local id tables cover each row exactly once" begin
        topology = Topology.TensorProductTopology(chain, ring, flipped_chain)
        for i in 2:4, k in 1:(i - 1)
            tables = topology.local_id_tables[i][k]
            for container_block_id in axes(tables, 1)
                local_ids = sort!(reduce(vcat, vec.(tables[container_block_id, :])))
                @test local_ids == 1:Topology._num_sub_objects(i - 1, k - 1)
            end
        end
    end
end

############################################################################################
#                                     Single factor                                        #
############################################################################################
@testset "Single factor" verbose=true begin
    # With a single factor, the product numbers all objects as the factor does, and lists
    # them in the same order. This checks that MeshTopology rows follow the local order of
    # the tensor-product patch of their dimension, which the product relies on.
    factors = Dict(
        "1D" => flipped_chain,
        "2D" => rotated_quads,
        "3D" => Topology.MeshTopology(hex_grid(2, 2, 1), Topology.HEX, Val(4)),
    )
    @testset "$name" for (name, factor) in factors
        topology = Topology.TensorProductTopology(factor)
        ir_dim = Topology.get_incidence_relations_dim(factor)
        @test size(topology) == size(factor)
        @test Topology.get_topological_patch(topology) ==
            Topology.get_topological_patch(factor)
        for i in 1:ir_dim, k in 1:ir_dim
            @test collect(topology[i, k]) == [abs.(row) for row in factor[i, k]]
        end
        # The product is a different type of topology than its factor.
        @test topology != factor
    end
end

############################################################################################
#                                 Comparison with MeshCore                                 #
############################################################################################
@testset "Compared with MeshTopology" verbose=true begin
    # Orientation mismatches (flipped chain, rotated quads) and periodicity (ring) are the
    # cases where the relative rotation and orientation of neighbours are non-trivial.
    products = Dict(
        "flipped chain ⊗ chain" => (flipped_chain, chain),
        "ring ⊗ flipped chain" => (ring, flipped_chain),
        "rotated quads ⊗ flipped chain" => (rotated_quads, flipped_chain),
        "flipped chain ⊗ rotated quads" => (flipped_chain, rotated_quads),
        "chain ⊗ ring ⊗ flipped chain" => (chain, ring, flipped_chain),
    )
    @testset "$name" verbose=true for (name, factors) in products
        topology = Topology.TensorProductTopology(factors)
        test_against_mesh_topology(topology)
        test_structural_invariants(topology)
    end

    @testset "nested products" verbose=true begin
        # The factors may themselves be tensor-product topologies. The objects are then
        # numbered differently, but the mesh is the same.
        nested = Topology.TensorProductTopology(
            Topology.TensorProductTopology(chain, ring), flipped_chain
        )
        flat = Topology.TensorProductTopology(chain, ring, flipped_chain)
        @test size(nested) == size(flat)
        @test nested[4, 1] == flat[4, 1]
        test_against_mesh_topology(nested)
        test_structural_invariants(nested)
    end

    @testset "skeleton factor" verbose=true begin
        # The edges of a 2D mesh, extruded.
        skeleton = Topology.SkeletonTopology(quads)
        topology = Topology.TensorProductTopology(skeleton, segment)
        test_against_mesh_topology(topology)
        test_structural_invariants(topology)
    end
end

############################################################################################
#                                      Whole mesh                                          #
############################################################################################
@testset "Whole-mesh neighbours" begin
    for topology in (
        Topology.TensorProductTopology(rotated_quads, flipped_chain),
        equivalent_mesh_topology(
            Topology.TensorProductTopology(rotated_quads, flipped_chain)
        ),
    )
        manifold_dim = Topology.get_manifold_dim(topology)
        for object_dim in 0:(manifold_dim - 1), include_local_patch in (false, true)
            all_neighbours = Topology.compute_neighbours(
                topology, object_dim, include_local_patch
            )
            @test size(all_neighbours) == (
                Topology.get_num_patches(topology),
                Topology.get_local_size(topology, object_dim + 1),
            )
            @test all(CartesianIndices(all_neighbours)) do index
                patch_id, local_id = Tuple(index)
                return all_neighbours[index] == Topology.compute_neighbours(
                    topology, patch_id, local_id, object_dim, include_local_patch
                )
            end
        end
    end
end

############################################################################################
#                                   Lazy evaluation                                        #
############################################################################################
@testset "Lazy evaluation" begin
    topology = Topology.TensorProductTopology(chain, segment)

    relation = topology[3, 1]
    @test relation isa Topology.TensorProductIncidenceRelation
    @test relation isa AbstractVector{Vector{Int}}
    @test size(relation) == (2,)
    @test eltype(relation) == Vector{Int}
    @test collect(relation) == [relation[1], relation[2]]
    # Rows are computed on request, so modifying one does not change the topology.
    row = relation[1]
    row[1] = 0
    @test relation[1] == [1, 2, 5, 4]
    # Objects are not related to the objects of their own dimension.
    @test isempty(topology[2, 2])
    @test size(topology[2, 2]) == (0,)

    # Only the structure of the product is stored, which does not depend on the size of the
    # factors.
    small = Topology.TensorProductTopology(
        Topology.MeshTopology(line_grid(2), Topology.LINE, Val(2)),
        Topology.MeshTopology(quad_grid(2, 2), Topology.QUAD, Val(4)),
    )
    large = Topology.TensorProductTopology(
        Topology.MeshTopology(line_grid(50), Topology.LINE, Val(50)),
        Topology.MeshTopology(quad_grid(20, 20), Topology.QUAD, Val(400)),
    )
    @test size(large) == (51 * 441, 50 * 441 + 51 * 840, 50 * 840 + 51 * 400, 50 * 400)
    own_size(t) = Base.summarysize(t) - Base.summarysize(t.factors)
    @test own_size(large) == own_size(small)
end

############################################################################################
#                                  Equality and hashing                                    #
############################################################################################
@testset "Equality and hashing" begin
    topology = Topology.TensorProductTopology(chain, segment)
    same = Topology.TensorProductTopology(
        Topology.MeshTopology(line_grid(2), Topology.LINE, Val(2)), segment
    )
    @test topology == same
    @test hash(topology) == hash(same)
    @test isequal(topology, same)
    @test topology != Topology.TensorProductTopology(segment, chain)
    @test topology != Topology.TensorProductTopology(chain, chain)
    # Equal to a MeshTopology of the same mesh would require equal numberings.
    @test topology != equivalent_mesh_topology(topology)
    @test length(Set([topology, same])) == 1
end

############################################################################################
#                                     Type stability                                       #
############################################################################################
@testset "Type stability" begin
    for topology in (
        Topology.TensorProductTopology(chain, segment),
        Topology.TensorProductTopology(rotated_quads, flipped_chain),
        Topology.TensorProductTopology(chain, ring, flipped_chain),
        Topology.TensorProductTopology(Topology.TensorProductTopology(chain, ring), chain),
    )
        manifold_dim = Topology.get_manifold_dim(topology)
        @test (@inferred topology[manifold_dim + 1, 1][1]) isa Vector{Int}
        @test (@inferred topology[1, manifold_dim + 1][1]) isa Vector{Int}
        @test (@inferred topology[2, 1][1]) isa Vector{Int}
        @test (@inferred size(topology)) isa NTuple{manifold_dim + 1, Int}
        @test (@inferred Topology.get_global_id(topology, 1, 2, 1)) isa Int
        @test (@inferred Topology.get_local_id(topology, 1, 1, 0)) isa Int
        @test (@inferred Topology.compute_neighbours(topology, 1, 2, 1)) isa Matrix{Int}
        @test (@inferred Topology.get_factor_object_ids(topology, 1, 1)) isa Tuple
        factors = Topology.get_factor_topologies(topology)
        @test (@inferred Topology.TensorProductTopology(factors)) == topology
    end
end

end

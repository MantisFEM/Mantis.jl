############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    MeshTopology{manifold_dim, ir_dim, num_patches, PT} <:
       AbstractTopology{manifold_dim, ir_dim, num_patches, PT}

Topological structure of a collection of patches (of equal shape) forming a mesh. See
[`AbstractTopology`](@ref) for more details about the type parameters.

# Constructors
- `MeshTopology(
        patches::Vector{NTuple{num_patch_vertices, Int}},
        topological_patch::AbstractPatch{1, 2, num_patch_vertices},
    ) where {num_patch_vertices}`: Constructor for 1D topologies.
- `MeshTopology(
        patches::Vector{NTuple{num_patch_vertices, Int}},
        topological_patch::AbstractPatch{2, 3, num_patch_vertices},
    ) where {num_patch_vertices}`: Constructor for 2D topologies.
- `MeshTopology(
        patches::Vector{NTuple{num_patch_vertices, Int}},
        topological_patch::AbstractPatch{3, 4, num_patch_vertices},
    ) where {num_patch_vertices}`: Constructor for 3D topologies.

# Fields
- `topological_patch::PT`: The [`AbstractPatch`](@ref) object out of which the mesh is made.
- `incidence_relations::NTuple{
        ir_dim, NTuple{ir_dim, Vector{Vector{Int}}}
    }`: The incidence relations between geometric objects of different dimensions.
- `num_geometric_objects::NTuple{ir_dim, Int}`: Total number of global
    geometric objects per topological dimension.
"""
struct MeshTopology{manifold_dim, ir_dim, num_patches, PT} <:
       AbstractTopology{manifold_dim, ir_dim, num_patches, PT}
    topological_patch::PT
    incidence_relations::NTuple{ir_dim, NTuple{ir_dim, Vector{Vector{Int}}}}
    num_geometric_objects::NTuple{ir_dim, Int}

    function MeshTopology(
        patches::NTuple{num_patches, NTuple{num_patch_vertices, Int}},
        topological_patch::AbstractPatch{manifold_dim, ir_dim, num_patch_vertices},
    ) where {num_patches, num_patch_vertices, manifold_dim, ir_dim}
        # We have to turn the patches into a vector for MeshCore
        patches_vec = [patch_i for patch_i in patches]
        return MeshTopology(patches_vec, topological_patch, Val(num_patches))
    end

    function MeshTopology(
        patches::Vector{NTuple{num_patch_vertices, Int}},
        topological_patch::AbstractPatch{1, 2, num_patch_vertices},
        ::Val{num_patches},
    ) where {num_patch_vertices, num_patches}
        num_vertices, patch2vertex, vertex2patch = _process_vertices(
            patches, topological_patch, num_patches
        )

        self_ir = Vector{Vector{Int}}()
        num_geometric_objects = (num_vertices, num_patches)
        incidence_relations = ((self_ir, vertex2patch._v), (patch2vertex._v, self_ir))

        return new{1, 2, num_patches, typeof(topological_patch)}(
            topological_patch, incidence_relations, num_geometric_objects
        )
    end

    function MeshTopology(
        patches::Vector{NTuple{num_patch_vertices, Int}},
        topological_patch::AbstractPatch{2, 3, num_patch_vertices},
        ::Val{num_patches},
    ) where {num_patch_vertices, num_patches}
        num_vertices, patch2vertex, vertex2patch = _process_vertices(
            patches, topological_patch, num_patches
        )

        num_faces, face2vertex, vertex2face, patch2face, face2patch = _process_facets(
            patch2vertex
        )

        self_ir = Vector{Vector{Int}}()
        num_geometric_objects = (num_vertices, num_faces, num_patches)
        incidence_relations = (
            (self_ir, vertex2face._v, vertex2patch._v),
            (face2vertex._v, self_ir, face2patch._v),
            (patch2vertex._v, patch2face._v, self_ir),
        )

        return new{2, 3, num_patches, typeof(topological_patch)}(
            topological_patch, incidence_relations, num_geometric_objects
        )
    end

    function MeshTopology(
        patches::Vector{NTuple{num_patch_vertices, Int}},
        topological_patch::AbstractPatch{3, 4, num_patch_vertices},
        ::Val{num_patches},
    ) where {num_patch_vertices, num_patches}
        num_vertices, patch2vertex, vertex2patch = _process_vertices(
            patches, topological_patch, num_patches
        )

        num_faces, face2vertex, vertex2face, patch2face, face2patch = _process_facets(
            patch2vertex
        )

        num_edges, edge2vertex, vertex2edge, patch2edge, edge2patch, face2edge, edge2face = _process_ridges(
            face2vertex, patch2vertex
        )

        self_ir = Vector{Vector{Int}}()
        num_geometric_objects = (num_vertices, num_edges, num_faces, num_patches)
        incidence_relations = (
            (self_ir, vertex2edge._v, vertex2face._v, vertex2patch._v),
            (edge2vertex._v, self_ir, edge2face._v, edge2patch._v),
            (face2vertex._v, face2edge._v, self_ir, face2patch._v),
            (patch2vertex._v, patch2edge._v, patch2face._v, self_ir),
        )

        return new{3, 4, num_patches, typeof(topological_patch)}(
            topological_patch, incidence_relations, num_geometric_objects
        )
    end
end

"""
    _process_vertices(patches, topological_patch)

Process the given `patches` of `topological_patch` to obtain the incidence relations between
vertices and patches. Used when constructing a [`MeshTopology`](@ref) with manifold_dim 1
or larger.

In table 1 of [Krysl2021](@cite) (for a 3D topology), this computes (3, 0) and (0, 3).
"""
function _process_vertices(patches, topological_patch, num_patches)
    if length(patches) != num_patches
        throw(
            ArgumentError(
                LazyString(
                    "The given number of patches: ",
                    num_patches,
                    ", should match the length of the provided vector of patch vertices: ",
                    length(patches),
                    ".",
                ),
            ),
        )
    end
    # First list all invalid vertex ids before throwing an error, so that the user is
    # immediately informed if multiple invalid ids are present.
    invalid_vertices = Vector{Tuple{Int, Int}}()
    num_vertices = 0
    for (patch_id, patch) in enumerate(patches)
        for vertex_id in patch
            num_vertices = max(num_vertices, vertex_id)
            if vertex_id < 1
                push!(invalid_vertices, (patch_id, vertex_id))
            end
        end
    end
    if !isempty(invalid_vertices)
        throw(
            ArgumentError(
                LazyString(
                    "All vertex ids must be positive. However, the following invalid ids ",
                    "were found (format: (patch_id, vertex_id)): ",
                    join(invalid_vertices, ", ", " and "),
                    ".",
                ),
            ),
        )
    end

    meshcore_patch = get_meshcore_patch(topological_patch)

    vertex_collection = MeshCore.ShapeColl(MeshCore.P1, num_vertices)
    patch_collection = MeshCore.ShapeColl(meshcore_patch, num_patches)

    patch2vertex = MeshCore.IncRel(patch_collection, vertex_collection, patches)
    vertex2patch = MeshCore.ir_transpose(patch2vertex)

    disconnected_vertices = findall(isempty, vertex2patch._v)
    if !isempty(disconnected_vertices)
        throw(
            ArgumentError(
                LazyString(
                    "All vertices must be consecutively numbered starting at 1. However, ",
                    "vertices ",
                    join(disconnected_vertices, ", ", " and "),
                    " are not connected to any patch and thus mising.",
                ),
            ),
        )
    end

    return num_vertices, patch2vertex, vertex2patch
end

"""
    _process_facets(patch2vertex)

Process the given `patch2vertex` relation to obtain the incidence relations between
facets (edges in 2D, faces in 3D) and vertices & patches. Used when constructing a
[`MeshTopology`](@ref) with manifold_dim 2 or larger.

In table 1 of [Krysl2021](@cite) (for a 3D topology), this computes (2, 0), (0, 2),
(3, 2), and (2, 3).
"""
function _process_facets(patch2vertex)
    face2vertex = MeshCore.ir_skeleton(patch2vertex)
    vertex2face = MeshCore.ir_transpose(face2vertex)
    patch2face = MeshCore.ir_bbyfacets(patch2vertex, face2vertex)
    face2patch = MeshCore.ir_transpose(patch2face)

    num_faces = MeshCore.nrelations(face2vertex)

    return num_faces, face2vertex, vertex2face, patch2face, face2patch
end

"""
    _process_ridges(face2vertex, patch2vertex)

Process the given `face2vertex` and `patch2vertex` relations to obtain the incidence
relations between ridges (edges in 3D) and vertices, faces & patches. Used when
constructing a [`MeshTopology`](@ref) with manifold_dim 3 or larger.

In table 1 of [Krysl2021](@cite) (for a 3D topology), this computes (1, 0), (0, 1),
(3, 1), (1, 3), (2, 1), and (1, 2).
"""
function _process_ridges(face2vertex, patch2vertex)
    edge2vertex = MeshCore.ir_skeleton(face2vertex)
    vertex2edge = MeshCore.ir_transpose(edge2vertex)
    patch2edge = MeshCore.ir_bbyridges(patch2vertex, edge2vertex)
    edge2patch = MeshCore.ir_transpose(patch2edge)
    face2edge = MeshCore.ir_bbyfacets(face2vertex, edge2vertex)
    edge2face = MeshCore.ir_transpose(face2edge)

    num_edges = MeshCore.nrelations(edge2vertex)

    return num_edges, edge2vertex, vertex2edge, patch2edge, edge2patch, face2edge, edge2face
end

############################################################################################
#                                        Equality                                          #
############################################################################################
# Ensure that two topologies are equal if their topological patches, sizes, and full
# incidence relations match. Because the incidence relations are vectors, the default
# isequal (whose fallback uses ===) would distinguish them. Setting up a hash with these
# quantities ensures their equality.
function Base.hash(topology::MeshTopology, h::UInt)
    h = hash(get_topological_patch(topology), h)
    h = hash(size(topology), h)
    return hash(topology.incidence_relations, h)
end

function Base.:(==)(a::MeshTopology, b::MeshTopology)
    return get_topological_patch(a) == get_topological_patch(b) &&
           size(a) == size(b) &&
           a.incidence_relations == b.incidence_relations
end

############################################################################################
#                                         Indexing                                         #
############################################################################################
Base.IndexStyle(::Type{<:MeshTopology}) = IndexLinear()
function Base.getindex(topology::MeshTopology, i::Int, k::Int)
    @boundscheck begin
        if i < 1 || i > get_incidence_relations_dim(topology)
            throw(BoundsError(topology.incidence_relations, i))
        end
        if k < 1 || k > get_incidence_relations_dim(topology)
            throw(BoundsError(topology.incidence_relations[i], k))
        end
    end
    @inbounds return topology.incidence_relations[i][k]
end

############################################################################################
#                                          Sizes                                           #
############################################################################################
Base.size(topology::MeshTopology) = topology.num_geometric_objects
Base.size(topology::MeshTopology, geometric_dim_id::Int) = size(topology)[geometric_dim_id]

############################################################################################
#                           Single TensorProductPatch Topologies                           #
############################################################################################
const SINGLE_PATCH_TOPOLOGY_1D = Topology.MeshTopology(((1, 2),), Topology.LINE)
const SINGLE_PATCH_TOPOLOGY_2D = Topology.MeshTopology(((1, 2, 3, 4),), Topology.QUAD)
const SINGLE_PATCH_TOPOLOGY_3D = Topology.MeshTopology(
    ((1, 2, 3, 4, 5, 6, 7, 8),), Topology.HEX
)

"""
    single_patch_tensorproduct_topology(::Val{manifold_dim})

Return the [`MeshTopology`](@ref) of a single patch of dimension `manifold_dim` using the
appropriate [`AbstractTensorProductPatch`](@ref).
"""
single_patch_tensorproduct_topology(::Val{1}) = SINGLE_PATCH_TOPOLOGY_1D
single_patch_tensorproduct_topology(::Val{2}) = SINGLE_PATCH_TOPOLOGY_2D
single_patch_tensorproduct_topology(::Val{3}) = SINGLE_PATCH_TOPOLOGY_3D
function single_patch_tensorproduct_topology(::Val{manifold_dim}) where {manifold_dim}
    return throw(
        ArgumentError(
            LazyString(
                "Topologies are limited to manifold dimension 1 to 3. Got ",
                manifold_dim,
                ".",
            ),
        ),
    )
end

############################################################################################
#                                 Tensor-product topologies                                #
############################################################################################
"""
    tensor_product_topology(factors::NTuple{num_factors, AbstractTopology})
    tensor_product_topology(factors::AbstractTopology...)

Return the [`MeshTopology`](@ref) of the tensor product of the topologies `factors`.

The patches of the product are the products of the patches of the factors. They are numbered
as the elements of a [`TensorProducts.TensorProduct`](@ref), i.e. patch `p` is the product
of the factor patches `Tuple(CartesianIndices(map(get_num_patches, factors))[p])`, with the
first factor varying fastest, as in a [`Geometry.TensorProductGeometry`](@ref). The vertices
are numbered in the same way. The patch type is the [`AbstractTensorProductPatch`](@ref) of
dimension `sum(map(get_manifold_dim, factors))`, and the vertices of every product patch are
listed in its local order. The remaining objects (edges, faces) are numbered, and their
global orientation is chosen, as for any other [`MeshTopology`](@ref).

All factors must be made of [`AbstractTensorProductPatch`](@ref)es, and the manifold
dimension of the product can be at most 3.
"""
function tensor_product_topology(
    factors::NTuple{num_factors, AbstractTopology}
) where {num_factors}
    _check_tensor_product_factors(factors)

    factor_dims = map(get_manifold_dim, factors)
    manifold_dim = sum(factor_dims)
    factor_num_patches = map(get_num_patches, factors)
    num_patches = prod(factor_num_patches)

    vmd = Val(manifold_dim)
    num_patches == 1 && return Topology.single_patch_tensorproduct_topology(vmd)

    patch = tensor_product_patch(vmd)
    num_patch_vertices = get_num_patch_vertices(patch)
    factor_patches = map(get_topological_patch, factors)
    factor_num_vertices = map(factor -> size(factor, 1), factors)
    vertex_ids = LinearIndices(factor_num_vertices)
    # Position of the first axis of each factor within the axes of the product patch.
    axis_offsets = ntuple(
        factor_id -> sum(factor_dims[1:(factor_id - 1)]; init=0), Val(num_factors)
    )

    # The local vertex of each factor patch that makes up each local vertex of the product
    # patch: the position of a product vertex is the concatenation of those of the factors.
    local_vertices = ntuple(num_patch_vertices) do local_vertex_id
        position = id_to_position(patch, 0, local_vertex_id)
        return ntuple(Val(num_factors)) do factor_id
            factor_position = ntuple(
                axis -> position[axis_offsets[factor_id] + axis], factor_dims[factor_id]
            )
            return position_to_id(factor_patches[factor_id], factor_position)
        end
    end

    factor_patch_vertices = map(
        (factor, factor_dim) -> factor[factor_dim + 1, 1], factors, factor_dims
    )
    patches = Vector{NTuple{num_patch_vertices, Int}}(undef, num_patches)
    for (patch_id, factor_patch_ids) in enumerate(CartesianIndices(factor_num_patches))
        factor_vertices = map(
            (rows, factor_patch_id) -> rows[factor_patch_id],
            factor_patch_vertices,
            Tuple(factor_patch_ids),
        )
        patches[patch_id] = map(local_vertices) do factor_local_vertices
            factor_vertex_ids = map(
                (vertices, local_vertex) -> abs(vertices[local_vertex]),
                factor_vertices,
                factor_local_vertices,
            )
            return vertex_ids[factor_vertex_ids...]
        end
    end

    return MeshTopology(patches, patch, Val(num_patches))
end

tensor_product_topology(factors::AbstractTopology...) = tensor_product_topology(factors)

"""
    _check_tensor_product_factors(factors::Tuple)

Throw an `ArgumentError` if `factors` is empty or if one of the factors is not made of
[`AbstractTensorProductPatch`](@ref)es, whose local numbering is needed to number the
objects of the product.
"""
function _check_tensor_product_factors(factors::Tuple)
    if isempty(factors)
        throw(
            ArgumentError("A tensor-product topology needs at least one factor topology.")
        )
    end
    for (factor_id, factor) in enumerate(factors)
        patch = get_topological_patch(factor)
        if !(patch isa AbstractTensorProductPatch)
            throw(
                ArgumentError(
                    LazyString(
                        "Factor ",
                        factor_id,
                        " is made of patches of type ",
                        typeof(patch),
                        ", but a tensor-product topology can only be formed from ",
                        "topologies made of tensor-product patches.",
                    ),
                ),
            )
        end
    end

    return nothing
end

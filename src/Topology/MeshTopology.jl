############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    MeshTopology{manifold_dim, incidence_relations_dim, num_patches, PT} <:
       AbstractTopology{manifold_dim, incidence_relations_dim, num_patches, PT}

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
        incidence_relations_dim, NTuple{incidence_relations_dim, Vector{Vector{Int}}}
    }`: The incidence relations between geometric objects of different dimensions.
- `num_geometric_objects::NTuple{incidence_relations_dim, Int}`: Total number of global
    geometric objects per topological dimension.
"""
struct MeshTopology{manifold_dim, incidence_relations_dim, num_patches, PT} <:
       AbstractTopology{manifold_dim, incidence_relations_dim, num_patches, PT}
    topological_patch::PT
    incidence_relations::NTuple{
        incidence_relations_dim, NTuple{incidence_relations_dim, Vector{Vector{Int}}}
    }
    num_geometric_objects::NTuple{incidence_relations_dim, Int}

    function MeshTopology(
        patches::Vector{NTuple{num_patch_vertices, Int}},
        topological_patch::AbstractPatch{1, 2, num_patch_vertices},
    ) where {num_patch_vertices}
        num_vertices, num_patches, patch2vertex, vertex2patch = _process_vertices(
            patches, topological_patch
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
    ) where {num_patch_vertices}
        num_vertices, num_patches, patch2vertex, vertex2patch = _process_vertices(
            patches, topological_patch
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
    ) where {num_patch_vertices}
        num_vertices, num_patches, patch2vertex, vertex2patch = _process_vertices(
            patches, topological_patch
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
function _process_vertices(patches, topological_patch)
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
    num_patches = length(patches)

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

    return num_vertices, num_patches, patch2vertex, vertex2patch
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

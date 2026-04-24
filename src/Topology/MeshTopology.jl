"""
    MeshTopology{manifold_dim, incidence_relations_dim, num_patches}

Represents the topological structure of a collection of patches (of equal shape) forming a mesh.

Each patch is considered an individual mesh patch at the global level. This structure
enables the computation of all incidence relations between geometric objects (vertices,
edges, faces, volumes), and the determination of topological neighbors. Supports 1D (lines),
2D (quads), and 3D (hexahedra) topologies.

# Fields
- `incidence_relations`: A nested tuple containing the incidence relations between
    geometric objects of different dimensions.
- `n_geometric_objects`: Total number of global geometric objects per topological dimension.
- `n_local_geometric_objects`: Number of local geometric objects per patch per dimension.
- `local_edge2vertex`: Local edge-to-vertex mapping, `local_edge2vertex[i,j]` contains the
    i-th vertex of the j-th edge, all at local level.
- `local_face2vertex`: Local face-to-vertex mapping, `local_face2vertex[i,j]` contains the
    i-th vertex of the j-th face, all at local level.

# Constructors
- `MeshTopology(patches::Vector{Vector{Int}})`: Builds the patch topology from a list of
    patch connectivities (vertex indices).
"""
struct MeshTopology{manifold_dim, incidence_relations_dim, num_patches} <:
       AbstractTopology{manifold_dim, incidence_relations_dim, num_patches}
    incidence_relations::NTuple{
        incidence_relations_dim, NTuple{incidence_relations_dim, Vector{Vector{Int}}}
    }  # incidence relations between the geometric objects
    n_geometric_objects::Vector{Int} # number of geometric objects in each dimension
    n_local_geometric_objects::Vector{Int} # number of local geometric objects in each dimension
    local_edge2vertex::Matrix{Int} # local edge to vertex incidence relation
    local_face2vertex::Matrix{Int} # local face to vertex incidence relation

    function MeshTopology(patches::Vector{Vector{Int}})
        # Determine the manifold dimension from the type of the patches
        # We consider only:
        # - line segments: 1D patches
        # - quadrilaterals: 2D patches
        # - hexahedra: 3D patches

        # Number of vertices in the first patch (assumed to be the same for all)
        # Defines the geometry of patches
        # 2: line segments
        # 4: quads
        # 8: hexahedra
        n_patch_vertices = length(patches[1])

        # Get the number of geometric objects of each dimension [#vertices, #edges, #facets]
        # The local incidence relations between
        #    i) edges and vertices
        #   ii) faces and vertices
        manifold_dim, patch_type, n_local_geometric_objects, local_edge2vertex, local_face2vertex = get_local_incidence_relations(
            n_patch_vertices
        )

        # Preallocate memory for the total number of geometric objects in each dimension
        n_geometric_objects = Vector{Int}(undef, manifold_dim + 1)

        # Preallocate memory for incidence relations
        incidence_relations_dim = manifold_dim + 1
        incidence_relations = ntuple(
            _ -> ntuple(_ -> Vector{Vector{Int}}(), incidence_relations_dim),
            incidence_relations_dim,
        )

        # The geometric objects present depend on the manifold dimension
        #   1D: vertices, lines [patches]
        #   2D: vertices, lines [facets], surfaces [patches]
        #   3D: vertices, lines [edges], surfaces [facets], volumes [patches]
        #
        # The steps to get all incidence relations are:
        # 1. Compute incidence relation between patches and vertices, i.e, (manifold_dim + 1, 1)
        # 2. Compute incidence relation between patches and facets, i.e, (manifold_dim + 1, manifold_dim)
        #    - In 1D this is not needed, so we skip it.
        #    - In 2D this is the incidence relation between patches and edges, i.e, (3, 2)
        #    - In 3D this is the incidence relation between patches and faces, i.e, (4, 3)
        # 3. Compute incidence relation between edges and vertices, i.e, (manifold_dim - 1, 1) (only in 3D)
        # 4. Compute incidence relation between patches and edges, i.e, (manifold_dim + 1, manifold_dim - 1) (only in 3D)
        # 5. Compute incidence relation between edges and patches, i.e, (manifold_dim - 1, manifold_dim + 1) (only in 3D)
        # 6. Compute incidence relation between facets and edges, i.e, (manifold_dim, manifold_dim - 1) (only in 3D)
        # 7. Compute incidence relation between edges and facets, i.e, (manifold_dim - 1, manifold_dim) (only in 3D)

        # Find the number of vertices by checking the maximum vertex id present in the definition of patches
        n_vertices = reduce(
            max, (vertex_id for patch in patches for vertex_id in patch); init=0
        )
        n_geometric_objects[1] = n_vertices  # number of vertices

        # Find the number of patches
        n_patches = length(patches)
        n_geometric_objects[manifold_dim + 1] = n_patches  # number of patches

        # We need to start by initializing the incicidence relation between n_manifold_dim geometrical objects
        # (patches) and the vertices. This is just the definition of patches that is given as input by the user.
        # We need to pass this into MeshCore to generate the incidence relation in the proper data structure so
        # that we can extract the other incidence relations from it.
        vertex_collection = MeshCore.ShapeColl(MeshCore.P1, n_vertices)  # generate the collection of vertices (this is just logical)
        patch_collection = MeshCore.ShapeColl(patch_type, n_patches)  # generate the collection of patches (this is just logical)
        patch2vertex = MeshCore.IncRel(patch_collection, vertex_collection, patches)  # the incidence relation between patches and vertices
        for patch_id in 1:n_patches
            push!(
                incidence_relations[manifold_dim + 1][1], collect(patch2vertex._v[patch_id])
            )
        end

        # Now we can compute the incidence relations between the vertices and the patches (1, manifold_dim + 1)
        vertex2patch = MeshCore.ir_transpose(patch2vertex)  # (1, manifold_dim + 1)
        for vertex_id in 1:n_vertices
            push!(
                incidence_relations[1][manifold_dim + 1],
                collect(vertex2patch._v[vertex_id]),
            )
        end

        if manifold_dim > 1
            # Compute the face incidence relations
            # First face to vertex (manifold_dim, 1)
            face2vertex = MeshCore.ir_skeleton(patch2vertex)  # (manifold_dim, 1)
            n_faces = MeshCore.nrelations(face2vertex)  # number of faces
            n_geometric_objects[manifold_dim] = n_faces  # number of faces

            for face_id in 1:n_faces
                push!(
                    incidence_relations[manifold_dim][1], collect(face2vertex._v[face_id])
                )
            end

            # Together with the vertex2face incidence relation (1, manifold_dim)
            vertex2face = MeshCore.ir_transpose(face2vertex)  # (1, manifold_dim)
            for vertex_id in 1:n_vertices
                push!(
                    incidence_relations[1][manifold_dim], collect(vertex2face._v[vertex_id])
                )
            end

            # Second the patch to face (manifold_dim + 1, manifold)
            patch2face = MeshCore.ir_bbyfacets(patch2vertex, face2vertex)  # (manifold_dim + 1, manifold)
            for patch_id in 1:n_patches
                push!(
                    incidence_relations[manifold_dim + 1][manifold_dim],
                    collect(patch2face._v[patch_id]),
                )
                if manifold_dim > 2
                    # Meshcore uses a different definition for the internal faces
                    # Due to this difference, internal faces 1, 4, and 5 need to flip their orientation
                    incidence_relations[manifold_dim + 1][manifold_dim][patch_id][1] *= -1
                    incidence_relations[manifold_dim + 1][manifold_dim][patch_id][4] *= -1
                    incidence_relations[manifold_dim + 1][manifold_dim][patch_id][5] *= -1
                end
            end

            # Third the face to patch (manifold_dim, manifold + 1)
            face2patch = MeshCore.ir_transpose(patch2face)  # (manifold_dim, manifold + 1)
            for face_id in 1:n_faces
                push!(
                    incidence_relations[manifold_dim][manifold_dim + 1],
                    collect(face2patch._v[face_id]),
                )
            end

            if manifold_dim > 2
                # Compute the edge incidence relations
                # First edge to vertex (manifold_dim - 1, 1)
                edge2vertex = MeshCore.ir_skeleton(face2vertex)  # (manifold_dim, 1)  --ir_skeleton--> (manifold_dim - 1, 1)
                n_edges = MeshCore.nrelations(edge2vertex)  # number of edges
                n_geometric_objects[manifold_dim - 1] = n_edges  # number of edges

                for edge_id in 1:n_edges
                    push!(
                        incidence_relations[manifold_dim - 1][1],
                        collect(edge2vertex._v[edge_id]),
                    )
                end

                # Together with the vertex2edge incidence relation (1, manifold_dim - 1)
                vertex2edge = MeshCore.ir_transpose(edge2vertex)  # (1, manifold_dim - 1)
                for vertex_id in 1:n_vertices
                    push!(
                        incidence_relations[1][manifold_dim - 1],
                        collect(vertex2edge._v[vertex_id]),
                    )
                end

                # Second the patch to edge (manifold_dim + 1, manifold_dim - 1)
                patch2edge = MeshCore.ir_bbyridges(patch2vertex, edge2vertex)  # (manifold_dim + 1, 1), (manifold_dim - 1, 1) --ir_bbydridges--> (manifold_dim + 1, manifold_dim - 1): (4, 1) + (2, 1) --ir_bbyridges--> (4, 2)
                for patch_id in 1:n_patches
                    push!(
                        incidence_relations[manifold_dim + 1][manifold_dim - 1],
                        collect(patch2edge._v[patch_id]),
                    )
                end

                # Third the edge to patch (manifold_dim - 1, manifold_dim + 1)
                edge2patch = MeshCore.ir_transpose(patch2edge)  # (manifold_dim + 1, manifold_dim - 1) --ir_transpose--> (manifold_dim - 1, manifold_dim + 1)
                for edge_id in 1:n_edges
                    push!(
                        incidence_relations[manifold_dim - 1][manifold_dim + 1],
                        collect(edge2patch._v[edge_id]),
                    )
                end

                # Fourth the face to edge (manifold_dim, manifold_dim - 1)
                face2edge = MeshCore.ir_bbyfacets(face2vertex, edge2vertex)  # (manifold_dim, 1), (manifold_dim - 1, 1) --ir_bbyridges--> (manifold_dim, manifold_dim - 1)
                for face_id in 1:n_faces
                    push!(
                        incidence_relations[manifold_dim][manifold_dim - 1],
                        collect(face2edge._v[face_id]),
                    )
                end

                # Fifth the edge to face (manifold_dim - 1, manifold_dim)
                edge2face = MeshCore.ir_transpose(face2edge)  # (manifold_dim, manifold_dim - 1) --ir_transpose--> (manifold_dim - 1, manifold_dim)
                for edge_id in 1:n_edges
                    push!(
                        incidence_relations[manifold_dim - 1][manifold_dim],
                        collect(edge2face._v[edge_id]),
                    )
                end
            end
        end

        return new{manifold_dim, incidence_relations_dim, n_patches}(
            incidence_relations,
            n_geometric_objects,
            n_local_geometric_objects,
            local_edge2vertex,
            local_face2vertex,
        )
    end
end

# Indexing.
Base.IndexStyle(::Type{<:MeshTopology}) = IndexLinear()
# i and k and geometric dimension indices, i.e., if geometric dimension is n
# then the index is (n + 1), this is because julia starts indices at 1 and vertices
# have dimension 0.
Base.getindex(topology::MeshTopology, i::Int, k::Int) = topology.incidence_relations[i][k]

# Sizes.
# Provide quick access to the number of geometric objects in each dimension
# (vertices, edges, patches) in 2D
# (vertices, edges, faces, patches) in 3D
Base.size(topology::MeshTopology) = topology.n_geometric_objects
# geometric_dim_id is the index associated to the geometric dimension. Geometric dimension n
# has index (n + 1), this is done to keep consistency with julia indices that start at 1 and
# not at 0 (vertices have geometric dimension 0).
Base.size(topology::MeshTopology, geometric_dim_id::Int) =
    topology.n_geometric_objects[geometric_dim_id]

"""
    get_local_size(topology::MeshTopology, geometric_dim_id::Int)
    get_local_size(topology::MeshTopology)

Return the number of local geometric objects per patch for a given geometric dimension.

All patches are assumed to have the same number of local geometric objects.

# Arguments
- `topology::MeshTopology`: The mesh topology.
- `geometric_dim_id::Int`: 1-based index of the geometric dimension.
    Dimension `n` corresponds to index `n + 1`
    (e.g. vertices have geometric dimension 0 and thus `geometric_dim_id = 1`).

# Returns
- `Int`: Number of local geometric objects of the given dimension per patch.
"""
function get_local_size(topology::MeshTopology, geometric_dim_id::Int)
    return topology.n_local_geometric_objects[geometric_dim_id]
end
function get_local_size(topology::MeshTopology)
    return topology.n_local_geometric_objects
end

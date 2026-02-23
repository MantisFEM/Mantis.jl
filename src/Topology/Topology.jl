module Topology

import MeshCore

const DEBUG = false  # set false for production

macro debug(ex)
    return :(DEBUG && $(esc(ex)))
end

const ID2POSITION_DICT = Dict(
    #----------------------------------------------------
    # 1D
    #----------------------------------------------------
    # Vertex numbering
    (1, 0, 1) => (-1,),
    (1, 0, 2) => (1,),
    # Edge numbering
    (1, 1, 1) => (0,),
    #----------------------------------------------------
    # 2D
    #----------------------------------------------------
    # Vertex numbering
    (2, 0, 1) => (-1, -1),
    (2, 0, 2) => (1, -1),
    (2, 0, 3) => (1, 1),
    (2, 0, 4) => (-1, 1),
    # Edge numbering
    (2, 1, 1) => (-1, 0),
    (2, 1, 2) => (1, 0),
    (2, 1, 3) => (0, -1),
    (2, 1, 4) => (0, 1),
    # Face numbering
    (2, 2, 1) => (0, 0),
    #----------------------------------------------------
    # 3D
    #----------------------------------------------------
    # Vertex numbering
    (3, 0, 1) => (-1, -1, -1),
    (3, 0, 2) => (1, -1, -1),
    (3, 0, 3) => (1, 1, -1),
    (3, 0, 4) => (-1, 1, -1),
    (3, 0, 5) => (-1, -1, 1),
    (3, 0, 6) => (1, -1, 1),
    (3, 0, 7) => (1, 1, 1),
    (3, 0, 8) => (-1, 1, 1),
    # Edge numbering
    (3, 1, 1) => (0, 1, 1),
    (3, 1, 2) => (0, -1, 1),
    (3, 1, 3) => (0, -1, -1),
    (3, 1, 4) => (0, 1, -1),
    (3, 1, 5) => (1, 0, 1),
    (3, 1, 6) => (-1, 0, 1),
    (3, 1, 7) => (-1, 0, -1),
    (3, 1, 8) => (1, 0, -1),
    (3, 1, 9) => (1, 1, 0),
    (3, 1, 10) => (-1, 1, 0),
    (3, 1, 11) => (-1, -1, 0),
    (3, 1, 12) => (1, -1, 0),
    # Face numbering
    (3, 2, 1) => (-1, 0, 0),
    (3, 2, 2) => (1, 0, 0),
    (3, 2, 3) => (0, -1, 0),
    (3, 2, 4) => (0, 1, 0),
    (3, 2, 5) => (0, 0, -1),
    (3, 2, 6) => (0, 0, 1),
    # Volume numbering
    (3, 3, 1) => (0, 0, 0),
)

const POSITION2ID_DICT = Dict(
    #----------------------------------------------------
    # 1D
    #----------------------------------------------------
    # Vertex numbering
    (-1,) => 1,
    (1,) => 2,
    # Edge numbering
    (0,) => 1,
    #----------------------------------------------------
    # 2D
    #----------------------------------------------------
    # Vertex numbering
    (-1, -1) => 1,
    (1, -1) => 2,
    (1, 1) => 3,
    (-1, 1) => 4,
    # Edge numbering
    (-1, 0) => 1,
    (1, 0) => 2,
    (0, -1) => 3,
    (0, 1) => 4,
    # Face numbering
    (0, 0) => 1,
    #----------------------------------------------------
    # 3D
    #----------------------------------------------------
    # Vertex numbering
    (-1, -1, -1) => 1,
    (1, -1, -1) => 2,
    (1, 1, -1) => 3,
    (-1, 1, -1) => 4,
    (-1, -1, 1) => 5,
    (1, -1, 1) => 6,
    (1, 1, 1) => 7,
    (-1, 1, 1) => 8,
    # Edge numbering
    (0, 1, 1) => 1,
    (0, -1, 1) => 2,
    (0, -1, -1) => 3,
    (0, 1, -1) => 4,
    (1, 0, 1) => 5,
    (-1, 0, 1) => 6,
    (-1, 0, -1) => 7,
    (1, 0, -1) => 8,
    (1, 1, 0) => 9,
    (-1, 1, 0) => 10,
    (-1, -1, 0) => 11,
    (1, -1, 0) => 12,
    # Face numbering
    (-1, 0, 0) => 1,
    (1, 0, 0) => 2,
    (0, -1, 0) => 3,
    (0, 1, 0) => 4,
    (0, 0, -1) => 5,
    (0, 0, 1) => 6,
    # Volume numbering
    (0, 0, 0) => 1,
)

"""
    id2position(manifold_dim::Int, object_dim::Int, object_local_id::Int)

Maps the local object ID (vertex, edge, face) in a reference patch to a logical position
tuple.

Supports dimensions 1 to 3.
Logical positions are defined based on assumption of tensor product patches (lines, quads,
hexahedra) and logical coordinate system is as follows:
    - -1: located at the leftmost or bottommost position (start of interval of that
        dimension)
    -  1: located at rightmost or topmost position (end of interval of that dimension)
    -  0: extended over that dimension

This means that the start vertex of a line segment is at (-1,), the end vertex is at (1,),
and the edge is at (0,).

For a quadrilateral, the vertices are at (-1, -1), (1, -1), (1, 1), (-1, 1), and the edges
are at (-1, 0), (1, 0), (0, -1), (0, 1) (left, right, bottom, top). The face is at (0, 0).

For hexahedra, the same logic follows, but now with one additional index.
"""
function id2position(manifold_dim::Int, object_dim::Int, object_local_id::Int)
    return ID2POSITION_DICT[(manifold_dim, object_dim, object_local_id)]
end

"""
    position2id(position::NTuple{D, Int}) where {D}

Maps a logical position (tuple of `Int`s) to its local object ID within the reference patch.
Inverse of `id2position`.

See [`id2position`](@ref) for the definition of logical positions.
"""
function position2id(position::NTuple{manifold_dim, Int}) where {manifold_dim}
    return POSITION2ID_DICT[position]
end

# To be added:
# function get_elements_at_face(mesh_topology, patch_id, face_id)
#     return element_ids, rotation, orientation
# end

# function get_elements_at_edge() end

# function get_elements_at_vertex() end

# struct TensorProductMeshTopology end

"""
    MeshTopology{manifold_dim, incidence_relations_dim, num_patches}

Represents the topological structure of a collection of patches forming a structured mesh.

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
struct MeshTopology{manifold_dim, incidence_relations_dim, num_patches}
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

        n_patch_vertices = length(patches[1])  # number of vertices in the first patch (assumed to be the same for all)

        # 1D manifold
        # Lines with vertices numbered as
        #
        # 1 --------- 2 --- ξ
        #
        if n_patch_vertices == 2
            manifold_dim = 1
            patch_type = MeshCore.L2

            n_local_geometric_objects = [2, 1]  # vertices, edges

            # Store the local face to vertex incidence relation for hexahedra
            # local_facet2vertex[i, j]: the i-th vertex of the j-th face
            local_edge2vertex = reshape(
                [
                    1
                    2
                ],
                :,
                1,
            )

            # Store the local face to vertex incidence relation for hexahedra
            # local_facet2vertex[i, j]: the i-th vertex of the j-th face
            local_face2vertex = zeros(Int, 1, 1)

            # 2D manifold
            # Quads with vertices numbered as
            #
            #   η
            #   |
            #   |
            #   4 ---------- 3
            #   |            |
            #   |            |
            #   |            |
            #   1 ---------- 2 --- ξ
            #
        elseif n_patch_vertices == 4
            manifold_dim = 2
            patch_type = MeshCore.Q4

            n_local_geometric_objects = [4, 4, 1]  # vertices, edges, facets

            # Store the local facet to vertex incidence relation for quads
            # local_facet2vertex[i, j]: the i-th vertex of the j-th facet (edge)
            local_edge2vertex = [
                1 2 1 4
                4 3 2 3
            ]

            # Store the local face to vertex incidence relation for hexahedra
            # local_facet2vertex[i, j]: the i-th vertex of the j-th face
            local_face2vertex = reshape(
                [
                    1
                    2
                    3
                    4
                ],
                :,
                1,
            )

            # 3D manifold
            # Hexahedra with vertices numbered as
            #
            #          ζ
            #          |
            #          |
            #          5 --------- 8
            #        / .         / .
            #      /   .       /   .
            #    6 --------- 7     .
            #    |     1 ----|---- 4 --- η
            #    |   .       |   .
            #    | .         | .
            #    2 --------- 3
            #   /
            # /
            # ξ
            #
        elseif n_patch_vertices == 8
            manifold_dim = 3
            patch_type = MeshCore.H8

            n_local_geometric_objects = [8, 12, 6, 1]  # vertices, edges, facets, volumes

            # Store the local edge to vertex incidence relation for hexahedra
            # local_edge2vertex[i, j]: the i-th vertex of the j-th edge
            local_edge2vertex = [
                8 5 1 4 6 5 1 2 3 4 1 2
                7 6 2 3 7 8 4 3 7 8 5 6
            ]

            # Store the local face to vertex incidence relation for hexahedra
            # local_facet2vertex[i, j]: the i-th vertex of the j-th face
            local_face2vertex = [
                1 2 1 4 1 5
                4 3 2 3 2 6
                8 7 6 7 3 7
                5 6 5 8 4 8
            ]

        else
            throw(
                ArgumentError(
                    LazyString(
                        "Unsupported patch type with ", num_patch_vertices, " vertices."
                    ),
                ),
            )
        end

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

function get_manifold_dim(
    ::MeshTopology{manifold_dim, incidence_relations_dim, num_patches}
) where {manifold_dim, incidence_relations_dim, num_patches}
    return manifold_dim
end
function get_incidence_relations_dim(
    ::MeshTopology{manifold_dim, incidence_relations_dim, num_patches}
) where {manifold_dim, incidence_relations_dim, num_patches}
    return incidence_relations_dim
end
function get_num_patches(
    ::MeshTopology{manifold_dim, incidence_relations_dim, num_patches}
) where {manifold_dim, incidence_relations_dim, num_patches}
    return num_patches
end

# Provide access to the incidence relation data as if it was a one-dimensional
# or two dimensional array.
Base.IndexStyle(::Type{<:MeshTopology}) = IndexLinear()
Base.getindex(mesh_topology::MeshTopology, i::Int, k::Int) =
    mesh_topology.incidence_relations[i][k]
Base.lastindex(mesh_topology::MeshTopology{manifold_dim}, d::Int=1) where {manifold_dim} =
    manifold_dim + 1

# Provide quick access to the number of geometric objects in each dimension
# (vertices, edges, patches) in 2D
# (vertices, edges, faces, patches) in 3D
Base.size(mesh_topology::MeshTopology) = mesh_topology.n_geometric_objects
Base.size(mesh_topology::MeshTopology, geometric_dim::Int) =
    mesh_topology.n_geometric_objects[geometric_dim]

function get_local_size(mesh_topology::MT) where {MT <: MeshTopology}
    # Get the (local, i.e., per patch, assumed all patches identical) number of geometric objects in each dimension
    return mesh_topology.n_local_geometric_objects
end

function get_local_size(mesh_topology::MT, geometric_dim::Int) where {MT <: MeshTopology}
    # Get the (local, i.e., per patch, assumed all patches identical) number of geometric objects for a given dimension
    return mesh_topology.n_local_geometric_objects[geometric_dim]
end

"""
    compute_face_neighbours(mesh_topology::MeshTopology{3,4}, patch_id::Int, face_local_id::Int)

Returns a `4 × N` matrix for the face `face_local_id` of `patch_id` containing information
    about its neighboring patches:
- Row 1: Neighboring patch ID
- Row 2: Local face ID in the neighbor patch
- Row 3: Rotation (number of vertices shifted), numbering of neighbour face dofs must be rotated
    clockwise 90 degrees as many times are rotation.
- Row 4: Orientation (+1 if aligned, -1 if reversed), indicates if the axis of the face in the neighbor patch
    is aligned with the axis of the face in the current patch, if not, the dof numbering must be transposed.

Only applicable to 3D hexahedral meshes.
"""
function compute_face_neighbours(
    mesh_topology::MT, patch_id::Int, face_local_id::Int
) where {MT <: MeshTopology{3, 4}}
    manifold_dim = 3  # we are in 3D
    patch_dimension = manifold_dim
    face_dimension = manifold_dim - 1
    vertex_dimension = 0

    # Determine the global face id and the
    # Get the faces of this patch
    @debug println("patch id: ", patch_id)
    patch_faces = mesh_topology[patch_dimension + 1, face_dimension + 1][patch_id]
    face_id = patch_faces[face_local_id]
    num_vertices_per_face = length(
        mesh_topology[face_dimension + 1, vertex_dimension + 1][1]
    ) # the number of vertices per face (assumed to be the same for all facets)

    @debug println("   face $face_local_id id: ", face_id)

    # Determine how many neighbours the face has
    face_patch_neighbours_ids = mesh_topology[face_dimension + 1, patch_dimension + 1][abs(
        face_id
    )]
    n_face_neighbours = length(face_patch_neighbours_ids) - 1

    # Initialize the face neighbours matrix
    # as an empty matrix if there are no neighbours
    # or as a matrix with 4 rows and n_face_neighbours columns
    if n_face_neighbours == 0
        @debug println("      no neighbours")
        face_neighbours = Matrix{Int}(undef, 4, 0)
        return face_neighbours
    else
        face_neighbours = zeros(Int, 4, n_face_neighbours)
    end

    # Populate the face neighbours matrix with neighbour
    for neighbour_patch_id in face_patch_neighbours_ids
        @debug println("      neighbour patch id: ", neighbour_patch_id)

        if neighbour_patch_id ≠ patch_id
            # Get the local id of the face in the neighbour patch
            neighbour_faces = mesh_topology[patch_dimension + 1, face_dimension + 1][neighbour_patch_id]
            for (neighbour_face_local_id, neighbour_face_id) in enumerate(neighbour_faces)
                @debug println("         neighbour face id: ", neighbour_face_id)

                if abs(neighbour_face_id) == abs(face_id)
                    # Store the global id of the neighbour patch
                    face_neighbours[1] = neighbour_patch_id

                    # Store the local id of the face in the neighbour patch
                    face_neighbours[2] = neighbour_face_local_id

                    # Store the orientation of the face relative to the neighbour patch

                    # Get the sequence of vertices of the face in the neighbour patch
                    neighbour_face_vertices_local_ids = mesh_topology.local_face2vertex[
                        :, neighbour_face_local_id
                    ]
                    neighbour_patch_vertices_ids = mesh_topology[
                        patch_dimension + 1, vertex_dimension + 1
                    ][neighbour_patch_id]
                    neighbour_face_vertices = neighbour_patch_vertices_ids[neighbour_face_vertices_local_ids]

                    # Get the sequence of vertices of the face in the current patch
                    face_vertices_local_idx = mesh_topology.local_face2vertex[
                        :, face_local_id
                    ]
                    patch_vertices_ids = mesh_topology[
                        patch_dimension + 1, vertex_dimension + 1
                    ][patch_id]
                    patch_face_vertices = patch_vertices_ids[face_vertices_local_idx]

                    @debug println(
                        "            neighbour face vertices: ", neighbour_face_vertices
                    )
                    @debug println("            patch face vertices: ", patch_face_vertices)

                    # Check the position of the first vertex of the face of the neighbour patch
                    # in the face of the current patch, this way we know the rotation between the two faces
                    base_vertex_neighbour_global_id = neighbour_face_vertices[1]
                    base_vertex_neighbour_local_id_in_face =
                        findfirst(
                            ==(base_vertex_neighbour_global_id), patch_face_vertices
                        ) - 1  # -1 because if the base vertex is located at vertex n of the neighbour then n-1 rotations are needed.
                    face_neighbours[3] = base_vertex_neighbour_local_id_in_face

                    # Then determine if the face is oriented in the same direction or not
                    # This is done by checking if the next vertex in the neighbour face is also
                    # the next vertex in the current face, or if it is the previous vertex
                    next_vertex_neighbour_global_id = neighbour_face_vertices[2]  # the second vertex of the facet in the neighbour patch
                    next_vertex_neighbour_local_id_in_face =
                        findfirst(
                            ==(next_vertex_neighbour_global_id), patch_face_vertices
                        ) - 1  # again, -1 to account for coinciding with the base vertex of the current face
                    if (base_vertex_neighbour_local_id_in_face == 0) && (
                        next_vertex_neighbour_local_id_in_face ==
                        (num_vertices_per_face - 1)
                    )
                        face_neighbours[4] = -1  # the face is oriented in the opposite direction
                    elseif (
                        base_vertex_neighbour_local_id_in_face ==
                        (num_vertices_per_face - 1)
                    ) && (next_vertex_neighbour_local_id_in_face == 0)
                        face_neighbours[4] = 1  # the face is oriented in the same direction
                    elseif next_vertex_neighbour_local_id_in_face >
                        base_vertex_neighbour_local_id_in_face
                        face_neighbours[4] = 1  # the face is oriented in the same direction
                    else
                        face_neighbours[4] = -1  # the face is oriented in the opposite direction
                    end

                    @debug println(
                        "            rotation: $(face_neighbours[3]); orientation: $(face_neighbours[4])\n",
                    )

                    break  # we do not need to check the other faces

                else
                    @debug println("            different face id")
                end
            end

        else
            @debug println("         same patch id\n")
        end
    end

    return face_neighbours
end

"""
    compute_face_neighbours(mesh_topology::MeshTopology{3,4})

Returns a matrix where each entry `[i,j]` contains the neighbor information of the `j`-th
face of patch `i`, as described in `compute_face_neighbours(mesh_topology, patch_id, face_local_id)`.
"""
function compute_face_neighbours(mesh_topology::MT) where {MT <: MeshTopology{3, 4}}
    manifold_dim = 3  # we are in 3D
    # Preallocate memory for the neighbours information
    n_local_faces = get_local_size(mesh_topology, manifold_dim)  # number of faces per patch
    n_total_patches = size(mesh_topology, manifold_dim + 1)
    face_neighbours = Matrix{Matrix{Int}}(undef, n_total_patches, n_local_faces)

    for patch_id in 1:n_total_patches
        @debug println("patch id: ", patch_id)

        # Loop over the faces of the patch and get the neighbours information
        for face_local_id in 1:n_local_faces
            @debug global_face_id = mesh_topology[manifold_dim + 1, manifold_dim][patch_id][face_local_id]
            @debug println("   face $face_local_id id: ", global_face_id)

            # Get the neighbours information for this face
            face_neighbours[patch_id, face_local_id] = compute_face_neighbours(
                mesh_topology, patch_id, face_local_id
            )
        end
    end
    return face_neighbours
end

"""
    compute_edge_neighbours(mesh_topology::MeshTopology{2,3}, patch_id::Int, edge_local_id::Int)

Returns a `4 × N` matrix describing the neighboring patches across edge `edge_local_id` of
`patch_id`. Each column encodes:
- Row 1: Neighboring patch ID.
- Row 2: Local edge ID in neighbor.
- Row 3: Always 0 (edges do not rotate), only orientation needs to be considered.
- Row 4: Orientation (+1 or -1), this indicates whether you need to apply `reverse` on edge numbering or not.

Only applicable to 2D quadrilateral meshes.
"""
function compute_edge_neighbours(
    mesh_topology::MT, patch_id::Int, edge_local_id::Int
) where {MT <: MeshTopology{2, 3}}
    manifold_dim = 2  # we are in 2D
    # Determine the global face id and the
    # Get the faces of this patch
    @debug println("patch id: ", patch_id)
    patch_edges = mesh_topology[manifold_dim + 1, manifold_dim][patch_id]
    edge_id = patch_edges[edge_local_id]
    num_vertices_per_edge = length(mesh_topology[manifold_dim, 1][1]) # the number of vertices per facet (assumed to be the same for all facets)

    @debug println("   edge $edge_local_id id: ", edge_id)

    # Determine how many neighbours the edge has
    n_edge_neighbours = length(mesh_topology[2, manifold_dim + 1][abs(edge_id)]) - 1
    if n_edge_neighbours == 0
        @debug println("      no neighbours")
        edge_neighbours = Matrix{Int}(undef, 4, 0)
        return edge_neighbours
    else
        edge_neighbours = zeros(Int, 4, n_edge_neighbours)
    end

    for neighbour_patch_id in mesh_topology[manifold_dim, manifold_dim + 1][abs(edge_id)]
        @debug println("      neighbour patch id: ", neighbour_patch_id)

        if neighbour_patch_id ≠ patch_id
            # Get the local id of the facet in the neighbour patch
            neighbour_edges = mesh_topology[manifold_dim + 1, manifold_dim][neighbour_patch_id]
            for (neighbour_edge_local_id, neighbour_edge_id) in enumerate(neighbour_edges)
                @debug println("         neighbour edge id: ", neighbour_edge_id)

                if abs(neighbour_edge_id) == abs(edge_id)
                    # Store the global id of the neighbour patch
                    edge_neighbours[1] = neighbour_patch_id

                    # Store the local id of the facet in the neighbour patch
                    edge_neighbours[2] = neighbour_edge_local_id

                    # Store the orientation of the facet relative to the neighbour patch

                    # Get the sequence of vertices of the facet in the current patch and its neighbour
                    neighbour_edge_vertices = mesh_topology[manifold_dim + 1, 1][neighbour_patch_id][mesh_topology.local_edge2vertex[
                        :, neighbour_edge_local_id
                    ]]
                    patch_edge_vertices = mesh_topology[manifold_dim + 1, 1][patch_id][mesh_topology.local_edge2vertex[
                        :, edge_local_id
                    ]]

                    @debug println(
                        "            neighbour edge vertices: ", neighbour_edge_vertices
                    )
                    @debug println("            patch edge vertices: ", patch_edge_vertices)

                    # Check the position of the first vertex of the edge in the neighbour patch
                    base_vertex_neighbour_global_id = neighbour_edge_vertices[1]
                    base_vertex_neighbour_local_id_in_edge =
                        findfirst(
                            ==(base_vertex_neighbour_global_id), patch_edge_vertices
                        ) - 1  # -1 because if the base vertex is located at vertex n of the neighbour then n-1 rotations are needed.
                    edge_neighbours[3] = 0  # There is no rotation since the facet is an edge, therefore there is only sign

                    if base_vertex_neighbour_local_id_in_edge == 0
                        edge_neighbours[4] = 1  # the edge is oriented in the same direction
                    else
                        edge_neighbours[4] = -1  # the edge is oriented in the opposite direction
                    end

                    @debug println(
                        "            rotation: $(edge_neighbours[3]); orientation: $(edge_neighbours[4])\n",
                    )

                    break  # we do not need to check the other facets

                else
                    @debug println("            different edge id")
                end
            end

        else
            @debug println("         same patch id\n")
        end
    end

    return edge_neighbours
end

"""
    compute_edge_neighbours(mesh_topology::MeshTopology{3,4}, patch_id::Int, edge_local_id::Int)

Same as the 2D version, but for 3D hexahedral meshes. Computes edge neighbors of a specific patch edge.
Returns a `4 × N` matrix describing the neighboring patches across edge `edge_local_id` of
`patch_id`. Each column encodes:
- Row 1: Neighboring patch ID.
- Row 2: Local edge ID in neighbor.
- Row 3: Always 0 (edges do not rotate), only orientation needs to be considered.
- Row 4: Orientation (+1 or -1).

Only applicable to 3D hexahedral meshes.
"""
function compute_edge_neighbours(
    mesh_topology::MT, patch_id::Int, edge_local_id::Int
) where {MT <: MeshTopology{3, 4}}
    manifold_dim = 3  # we are in 2D
    # Determine the global edge id
    # Get the edges of this patch
    @debug println("patch id: ", patch_id)
    patch_edges = mesh_topology[manifold_dim + 1, 2][patch_id]
    edge_id = patch_edges[edge_local_id]

    @debug println("   facet $edge_local_id id: ", edge_id)

    # Determine how many neighbours the edge has
    n_edge_neighbours = length(mesh_topology[2, manifold_dim + 1][abs(edge_id)]) - 1
    if n_edge_neighbours == 0
        @debug println("      no neighbours")
        edge_neighbours = Matrix{Int}(undef, 4, 0)
        return edge_neighbours
    else
        edge_neighbours = zeros(Int, 4, n_edge_neighbours)
    end

    for neighbour_patch_id in mesh_topology[2, manifold_dim + 1][abs(edge_id)]
        @debug println("      neighbour patch id: ", neighbour_patch_id)

        if neighbour_patch_id ≠ patch_id
            # Get the local id of the facet in the neighbour patch
            neighbour_edges = mesh_topology[manifold_dim + 1, 2][neighbour_patch_id]
            for (neighbour_edge_local_id, neighbour_edge_id) in enumerate(neighbour_edges)
                @debug println("         neighbour edge id: ", neighbour_edge_id)

                if abs(neighbour_edge_id) == abs(edge_id)
                    # Store the global id of the neighbour patch
                    edge_neighbours[1] = neighbour_patch_id

                    # Store the local id of the facet in the neighbour patch
                    edge_neighbours[2] = neighbour_edge_local_id

                    # Store the orientation of the facet relative to the neighbour patch

                    # Get the sequence of vertices of the facet in the current patch and its neighbour
                    neighbour_edge_vertices = mesh_topology[manifold_dim + 1, 1][neighbour_patch_id][mesh_topology.local_edge2vertex[
                        :, neighbour_edge_local_id
                    ]]
                    patch_edge_vertices = mesh_topology[manifold_dim + 1, 1][patch_id][mesh_topology.local_edge2vertex[
                        :, edge_local_id
                    ]]

                    @debug println(
                        "            neighbour edge vertices: ", neighbour_edge_vertices
                    )
                    @debug println("            patch edge vertices: ", patch_edge_vertices)

                    # Check the position of the first vertex of the facet in the neighbour patch
                    base_vertex_neighbour_global_id = neighbour_edge_vertices[1]
                    base_vertex_neighbour_local_id_in_edge =
                        findfirst(
                            ==(base_vertex_neighbour_global_id), patch_edge_vertices
                        ) - 1  # -1 because if the base vertex is located at vertex n of the neighbour then n-1 rotations are needed.
                    edge_neighbours[3] = 0  # There is no rotation since the facet is an edge, therefore there is only sign

                    if base_vertex_neighbour_local_id_in_edge == 0
                        edge_neighbours[4] = 1  # the edge is oriented in the same direction
                    else
                        edge_neighbours[4] = -1  # the edge is oriented in the opposite direction
                    end

                    @debug println(
                        "            rotation: $(edge_neighbours[3]); orientation: $(edge_neighbours[4])\n",
                    )

                    break  # we do not need to check the other edges

                else
                    @debug println("            different edge id")
                end
            end

        else
            @debug println("         same patch id\n")
        end
    end

    return edge_neighbours
end

"""
    compute_edge_neighbours(mesh_topology::MeshTopology)

Returns a matrix containing neighbor information for all edges of all patches.
    Each entry `[i,j]` corresponds to the result of `compute_edge_neighbours(mesh_topology, i, j)`.
"""
function compute_edge_neighbours(
    mesh_topology::MT
) where {
    manifold_dim,
    incidence_relations_dim,
    MT <: MeshTopology{manifold_dim, incidence_relations_dim},
}
    # Preallocate memory for the neighbours information
    n_local_edges = get_local_size(mesh_topology, 2)  # number of edges per patch
    n_total_patches = size(mesh_topology, manifold_dim + 1)
    edge_neighbours = Array{Matrix{Int}}(undef, n_total_patches, n_local_edges)

    for patch_id in 1:n_total_patches
        @debug println("patch id: ", patch_id)

        # Loop over the faces of the patch and get the neighbours information
        for edge_local_id in 1:n_local_edges
            @debug global_edge_id = mesh_topology[manifold_dim + 1, 2][patch_id][edge_local_id]
            @debug println("   face $edge_local_id id: ", global_edge_id)

            # Get the neighbours information for this face
            edge_neighbours[patch_id, edge_local_id] = compute_edge_neighbours(
                mesh_topology, patch_id, edge_local_id
            )
        end
    end
    return edge_neighbours
end

"""
    compute_vertex_neighbours(mesh_topology::MeshTopology, patch_id::Int, vertex_local_id::Int)

Returns a `4 × N` matrix of vertex neighbor data:
- Row 1: Neighboring patch ID.
- Row 2: Local vertex ID in neighbor.
- Row 3: Always 0 (no rotation for vertices).
- Row 4: Always 0 (no orientation.

Applicable to all supported topologies.
"""
function compute_vertex_neighbours(
    mesh_topology::MT, patch_id::Int, vertex_local_id::Int
) where {
    manifold_dim,
    incidence_relations_dim,
    MT <: MeshTopology{manifold_dim, incidence_relations_dim},
}
    # Determine the global face id and the
    # Get the faces of this patch
    @debug println("patch id: ", patch_id)
    patch_vertices = mesh_topology[manifold_dim + 1, 1][patch_id]
    vertex_id = patch_vertices[vertex_local_id]

    @debug println("   vertex $vertex_local_id id: ", vertex_id)

    # Determine how many neighbours the face has
    n_vertex_neighbours = length(mesh_topology[1, manifold_dim + 1][abs(vertex_id)]) - 1
    if n_vertex_neighbours == 0
        @debug println("      no neighbours")
        vertex_neighbours = Matrix{Int}(undef, 4, 0)
        return vertex_neighbours
    else
        vertex_neighbours = zeros(Int, 4, n_vertex_neighbours)
    end

    for neighbour_patch_id in mesh_topology[1, manifold_dim + 1][abs(vertex_id)]
        @debug println("      neighbour patch id: ", neighbour_patch_id)

        if neighbour_patch_id ≠ patch_id
            # Get the local id of the facet in the neighbour patch
            neighbour_vertices = mesh_topology[manifold_dim + 1, 1][neighbour_patch_id]
            for (neighbour_vertex_local_id, neighbour_vertex_id) in
                enumerate(neighbour_vertices)
                @debug println("         neighbour vertex id: ", neighbour_vertex_id)

                if abs(neighbour_vertex_id) == abs(vertex_id)
                    # Store the global id of the neighbour patch
                    vertex_neighbours[1] = neighbour_patch_id

                    # Store the local id of the facet in the neighbour patch
                    vertex_neighbours[2] = neighbour_vertex_local_id

                    # Store the orientation of the facet relative to the neighbour patch
                    # In 1D this is trivial, since no rotation is needed and there is no sign
                    vertex_neighbours[3] = 0  # there is no rotation since the facet is an vertex
                    vertex_neighbours[4] = 0  # there is also no sign

                    @debug println(
                        "            rotation: $(vertex_neighbours[3]); orientation: $(vertex_neighbours[4])\n",
                    )

                    break  # we do not need to check the other facets

                else
                    @debug println("            different vertex id")
                end
            end

        else
            @debug println("         same patch id\n")
        end
    end

    return vertex_neighbours
end

"""
    compute_vertex_neighbours(mesh_topology::MeshTopology)

Returns a matrix of vertex neighbor information for all vertices of all patches.
"""
function compute_vertex_neighbours(mesh_topology::MeshTopology)
    manifold_dim = get_manifold_dim(mesh_topology)

    # Preallocate memory for the neighbours information
    n_local_vertices = get_local_size(mesh_topology, 1)  # number of faces per patch
    n_total_patches = size(mesh_topology, manifold_dim + 1)
    vertex_neighbours = Matrix{Matrix{Int}}(undef, n_total_patches, n_local_vertices)

    @debug println("--------------------------------------------")

    for patch_id in 1:n_total_patches
        # Loop over the faces of the patch and get the neighbours information
        for vertex_local_id in 1:n_local_vertices            # Get the neighbours information for this face
            vertex_neighbours[patch_id, vertex_local_id] = compute_vertex_neighbours(
                mesh_topology, patch_id, vertex_local_id
            )
        end
    end
    @debug println("--------------------------------------------")
    return vertex_neighbours
end

end

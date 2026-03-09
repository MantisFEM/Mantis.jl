module Topology

import MeshCore

const DEBUG = false  # set false for production

macro debug(ex)
    return :(DEBUG && $(esc(ex)))
end

# To be added:
# struct TensorProductMeshTopology end

"""
    AbstractTopology{manifold_dim, incidence_relations_dim, num_patches}

Abstract type of all topologies that represent the topological structure of a collection of
patches (of equal shape) forming a mesh. This includes the skeleton mesh.

Each patch is considered an individual mesh element at the global level. These structures
enable the computation of all incidence relations between geometric objects (vertices,
edges, faces, volumes), and the determination of topological neighbors. Supports 1D (lines),
2D (quads), and 3D (hexahedra) topologies.
"""
abstract type AbstractTopology{manifold_dim, incidence_relations_dim, num_patches} end

# Type-based getters.
function get_manifold_dim(
    ::AbstractTopology{manifold_dim, incidence_relations_dim, num_patches}
) where {manifold_dim, incidence_relations_dim, num_patches}
    return manifold_dim
end
function get_incidence_relations_dim(
    ::AbstractTopology{manifold_dim, incidence_relations_dim, num_patches}
) where {manifold_dim, incidence_relations_dim, num_patches}
    return incidence_relations_dim
end
function get_num_patches(
    ::AbstractTopology{manifold_dim, incidence_relations_dim, num_patches}
) where {manifold_dim, incidence_relations_dim, num_patches}
    return num_patches
end

# Indexing.
function Base.lastindex(::AbstractTopology{manifold_dim}, d::Int=1) where {manifold_dim}
    return manifold_dim + 1
end

# Local numbering and connectivity.
"""
    get_local_incidence_relations(n_patch_vertices::Int)

Return the local incidence relations for a patch type determined by its number of vertices.

Supported patch types and their local vertex, edge, and face numbering conventions are:

Supported patch types are:
- `2` vertices: 1D line element (L2)
- `4` vertices: 2D quadrilateral element (Q4)
- `8` vertices: 3D hexahedral element (H8)

Their local numbering is as given below:

---

**1D line element (L2, 2 vertices, 1 edge)**

Vertices:
```
1 --------- 2 --- ξ
```
Edge 1 is the element itself, oriented 1 → 2.

---

**2D quadrilateral element (Q4, 4 vertices, 4 edges, 1 face)**

Vertices:
```
  η
  |
  |
  4 ---------- 3
  |            |
  |            |
  |            |
  1 ---------- 2 --- ξ
```
Edges (arrows indicate orientation):
```
  η
  |
  |
  * ----e4----> *
  ^             ^
  |             |
  e1            e2
  |             |
  * ----e3----> * --- ξ
```
- Edge 1: 1 → 4 (left, bottom to top)
- Edge 2: 2 → 3 (right, bottom to top)
- Edge 3: 1 → 2 (bottom, left to right)
- Edge 4: 4 → 3 (top, left to right)

Face 1 is the element itself, with cyclic vertex order 1 → 2 → 3 → 4.

---

**3D hexahedral element (H8, 8 vertices, 12 edges, 6 faces)**

Vertices:
```
         ζ
         |
         |
         5 --------- 8
       / .         / |
     /   .       /   |
   6 --------- 7     |
   |     1 . . | . . 4 --- η
   |   .       |   /
   | .         | /
   2 --------- 3
  /
ξ
```
Edges (arrows indicate orientation):
```
         ζ
         |
         |
         * ---e6----> *
       / ↑          ↙ ↑
     e2  e11      e1  |
    ↙    .       /    e10
   * ---e5----> *     |
   ↑     *. e7 .↑. . →* --- η
   e12  .       |    /
   |  e3       e9  e4
   | ↙          | ↙
   * ----e8---> *
  /
ξ
```
- Edge 1: 8 → 7
- Edge 2: 5 → 6
- Edge 3: 1 → 2
- Edge 4: 4 → 3
- Edge 5: 6 → 7
- Edge 6: 5 → 8
- Edge 7: 1 → 4
- Edge 8: 2 → 3
- Edge 9: 3 → 7
- Edge 10: 4 → 8
- Edge 11: 1 → 5
- Edge 12: 2 → 6

Faces (cyclic vertex order indicates orientation):

- Face 1 (f1, back face, not visible): 1 → 4 → 8 → 5 (ξ = min)
- Face 2 (f2, front face): 2 → 3 → 7 → 6 (ξ = max)
- Face 3 (f3, left face): 1 → 2 → 6 → 5 (η = min)
- Face 4 (f4, right face): 4 → 3 → 7 → 8 (η = max)
- Face 5 (f5, bottom face): 1 → 2 → 3 → 4 (ζ = min)
- Face 6 (f6, top face): 5 → 6 → 7 → 8 (ζ = max)

---

# Arguments
- `n_patch_vertices::Int`: Number of vertices in the patch.

# Returns
A tuple `(manifold_dim, patch_type, n_local_geometric_objects, local_edge2vertex, local_face2vertex)` where:
- `manifold_dim::Int`: Topological dimension of the patch.
- `patch_type`: The corresponding `MeshCore` patch type (e.g. `MeshCore.L2`, `MeshCore.Q4`, `MeshCore.H8`).
- `n_local_geometric_objects::Vector{Int}`: Number of local geometric objects per dimension,
  ordered as `[n_vertices, n_edges, n_faces, n_volumes]`, truncated to the relevant dimensions.
- `local_edge2vertex::Matrix{Int}`: A `2 × n_edges` matrix where column `j` contains the
  local vertex indices of the two endpoints of the `j`-th local edge.
- `local_face2vertex::Matrix{Int}`: A `4 × n_faces` matrix where column `j` contains the
  local vertex indices of the `j`-th local face, in cyclic order. For 1D and 2D patches this
  matrix is a placeholder and should not be used.

# Throws
- `ArgumentError`: If `n_patch_vertices` does not correspond to a supported patch type.
"""
function get_local_incidence_relations(n_patch_vertices::Int)
    if n_patch_vertices == 2
        # 1D Line
        manifold_dim = 1
        patch_type = MeshCore.L2

        n_local_geometric_objects = [2, 1]  # vertices, edges

        # Store the local face to vertex incidence relation for hexahedra
        # local_facet2vertex[i, j]: the i-th vertex of the j-th face
        local_edge2vertex = reshape([1, 2], :, 1)

        # Store the local face to vertex incidence relation for hexahedra
        # local_facet2vertex[i, j]: the i-th vertex of the j-th face
        local_face2vertex = zeros(Int, 1, 1)

    elseif n_patch_vertices == 4
        # 2D Quad
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
        local_face2vertex = reshape([1, 2, 3, 4], :, 1)

    elseif n_patch_vertices == 8
        # 3D Hexahedra
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
                LazyString("Unsupported patch type with ", num_patch_vertices, " vertices.")
            ),
        )
    end

    return manifold_dim,
    patch_type, n_local_geometric_objects, local_edge2vertex,
    local_face2vertex
end

# Position/id conversions.
"""
    ID2POSITION_DICT

Dictionary mapping a local geometric object identifier to its reference-element position.

Keys are tuples `(manifold_dim, object_dim, local_id)` where:
- `manifold_dim::Int`: Topological dimension of the containing element (1, 2, or 3).
- `object_dim::Int`: Topological dimension of the geometric object
  (0 = vertex, 1 = edge, 2 = face, 3 = volume).
- `local_id::Int`: Local index of the geometric object within the element.

Values are tuples of length `manifold_dim`, with each component being `-1`, `0`, or `+1`,
encoding the position of the object along each reference axis (ξ, η, ζ):
- `-1`: object is located at the minimum of that axis.
- `+1`: object is located at the maximum of that axis.
- `0`: object  extends along that axis (e.g. an edge aligned with it).

# Examples

**1D element:**
- Vertex 1: key `(1, 0, 1)` → position `(-1,)` (at ξ = -1)
- Vertex 2: key `(1, 0, 2)` → position `(1,)` (at ξ = +1)
- Edge 1:   key `(1, 1, 1)` → position `(0,)` (extends along ξ)

**2D element:**
- Vertex 1: key `(2, 0, 1)` → position `(-1, -1)` (at ξ = -1, η = -1)
- Edge 1:   key `(2, 1, 1)` → position `(-1, 0)` (at ξ = -1, extends along η)
- Edge 3:   key `(2, 1, 3)` → position `(0, -1)` (extends along ξ, at η = -1)
- Face 1:   key `(2, 2, 1)` → position `(0, 0)` (extends along the two axis ξ and η)

**3D element:**
- Vertex 1: key `(3, 0, 1)` → position `(-1, -1, -1)` (at ξ = -1, η = -1, ζ = -1)
- Edge 4:   key `(3, 1, 4)` → position `(0, 1, -1)` (extends along ξ, at η = +1, ζ = -1)
- Face 1:   key `(3, 2, 1)` → position `(-1, 0, 0)` (at ξ = -1, extends along η and ζ)
- Volume 1: key `(3, 3, 1)` → position `(0, 0, 0)` (extends along the three axis, ξ, η, and ζ)

# Notes
- Supported element types are line segments (1D), quadrilaterals (2D), and hexahedra (3D).
- The position tuple has a unique entry for each geometric object in the reference element,
  and can therefore be used as an alternative identifier to the local index.
- See also [`get_local_incidence_relations`](@ref) for the local vertex numbering conventions
  that determine the correspondence between local indices and positions.
"""
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

Return the reference-element position tuple of a geometric object identified by its
local index.

This is a lookup into [`ID2POSITION_DICT`](@ref). The position tuple encodes the location
of the object in the reference element along each axis (ξ, η, ζ) using the following
conventions:
- `-1`: object is located at the minimum of that axis.
- `+1`: object is located at the maximum of that axis.
- `0`: object extends along that axis.

The returned tuple has length `manifold_dim`, one component per reference axis.

# Arguments
- `manifold_dim::Int`: Topological dimension of the containing element (1, 2, or 3).
- `object_dim::Int`: Topological dimension of the geometric object
  (0 = vertex, 1 = edge, 2 = face, 3 = volume).
- `object_local_id::Int`: Local index of the geometric object within the element.

# Returns
- `NTuple{manifold_dim, Int}`: Position tuple of the geometric object in the reference element.

# Examples

**1D element:**
- `id2position(1, 0, 1)` → `(-1,)` (vertex at ξ = -1)
- `id2position(1, 0, 2)` → `(1,)`  (vertex at ξ = +1)
- `id2position(1, 1, 1)` → `(0,)`  (edge extending along ξ)

**2D element:**
- `id2position(2, 0, 1)` → `(-1, -1)` (vertex at ξ = -1, η = -1)
- `id2position(2, 1, 1)` → `(-1, 0)`  (edge at ξ = -1, extending along η)
- `id2position(2, 1, 3)` → `(0, -1)`  (edge extending along ξ, at η = -1)
- `id2position(2, 2, 1)` → `(0, 0)`   (face, i.e. the element itself)

**3D element:**
- `id2position(3, 0, 1)` → `(-1, -1, -1)` (vertex at ξ = -1, η = -1, ζ = -1)
- `id2position(3, 1, 4)` → `(0, 1, -1)`   (edge extending along ξ, at η = +1, ζ = -1)
- `id2position(3, 2, 1)` → `(-1, 0, 0)`   (face at ξ = -1, extending along η and ζ)
- `id2position(3, 3, 1)` → `(0, 0, 0)`    (volume, i.e. the element itself)

# Notes
- Supported element types are line segments (1D), quadrilaterals (2D), and hexahedra (3D).
- See [`ID2POSITION_DICT`](@ref) for the full mapping.
- See [`get_local_incidence_relations`](@ref) for the local vertex numbering conventions.
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

# Local/global conversions.
"""
    get_global_id(
        topology::AbstractTopology,
        container_id::Int,
        container_dim::Int,
        local_id::Int,
        local_dim::Int,
    )

Return the global index of the geometric object of dimension `local_dim` with local
index `local_id` within the geometric object of dimension `container_dim` and global
index `container_id`.

# Arguments
- `topology::AbstractTopology`: The mesh topology.
- `container_id::Int`: Global index of the containing geometric object.
- `container_dim::Int`: Topological dimension of the containing object.
- `local_id::Int`: Local index of the target object within the container.
- `local_dim::Int`: Topological dimension of the target object.

# Returns
- `Int`: Global index of the target geometric object.

# Example
To obtain the global index of the 2nd vertex (dimension 0) of the 5th edge (dimension 1):
```julia
get_global_id(topology, 5, 1, 2, 0)
```
"""
function get_global_id(
    topology::AbstractTopology,
    container_id::Int,
    container_dim::Int,
    local_id::Int,
    local_dim::Int,
)
    # Definitions
    #   - container: is the object of dimension container_dim and global index container_id
    #   - local_objec: is the object of dimension local_dim and local index in container local_id

    # Get the list of objects of dimension local_dim in container
    container_local_objects = topology[container_dim + 1, local_dim + 1][container_id]

    # Extract the global index of the local object
    global_id = container_local_objects[local_id]

    return global_id
end

"""
    get_global_id(
        topology::AbstractTopology{manifold_dim},
        patch_id::Int,
        local_object_id::Int,
        local_object_dim::Int,
    ) where {manifold_dim}

Convenience method for [`get_global_id(::AbstractTopology, ::Int, ::Int, ::Int, ::Int)`](@ref)
where the container is a patch, i.e. the containing dimension is fixed to `manifold_dim`.

Return the global index of the geometric object of dimension `local_object_dim` with local
index `local_object_id` within patch `patch_id`.

# Arguments
- `topology::AbstractTopology{manifold_dim}`: The mesh topology.
- `patch_id::Int`: Global index of the patch.
- `local_object_id::Int`: Local index of the target object within the patch.
- `local_object_dim::Int`: Topological dimension of the target object.

# Returns
- `Int`: Global index of the target geometric object.

# Example
To obtain the global index of the 2nd vertex (dimension 0) of the 5th patch:
```julia
get_global_id(topology, 5, 2, 0)
```
"""
function get_global_id(
    topology::AbstractTopology{manifold_dim}, patch_id, local_object_id, local_object_dim
) where {manifold_dim}
    return get_global_id(
        topology, patch_id, manifold_dim, local_object_id, local_object_dim
    )
end

"""
    get_local_id(
        topology::AbstractTopology,
        patch_id::Int,
        global_object_id::Int,
        object_dim::Int,
    )

Return the local index of the geometric object of dimension `object_dim` with global
index `global_object_id` within patch `patch_id`, or `nothing` if the object is not
found in that patch.

# Notes
- The search is performed on absolute values of the stored indices, as sign is used
  to encode orientation information.

# Arguments
- `topology::AbstractTopology`: The mesh topology.
- `patch_id::Int`: Global index of the patch.
- `global_object_id::Int`: Global index of the target geometric object.
- `object_dim::Int`: Topological dimension of the target object.

# Returns
- `Int`: Local index of the target object within the patch, or `0` if not found.

# Example
To obtain the local index of the edge (dimension 1) with global index 7 within patch 3:
```julia
get_local_id(topology, 3, 7, 1)
```
"""
function get_local_id(topology::AbstractTopology, patch_id, global_object_id, object_dim)
    # First get the local id with no sign
    local_object_id = findfirst(
        x -> abs(x) == global_object_id,
        topology[get_manifold_dim(topology) + 1, object_dim + 1][patch_id],
    )

    # Then get the sign
    local_object_id =
        local_object_id * sign(
            topology[get_manifold_dim(topology) + 1, object_dim + 1][patch_id][local_object_id],
        )

    return local_object_id
end

# Neighbour information
"""
    compute_face_neighbours(
        mesh_topology::AbstractTopology{3,4},
        patch_id::Int,
        face_local_id::Int;
        include_local_patch::Bool = false,
    )

Return a `4 × N` matrix describing the neighbouring patches of the face
`face_local_id` of patch `patch_id`.

The columns of the returned matrix correspond to each of the neighbouring patches,
and the rows encode local identification of face and orientation with respect to
original patch (`patch_id`).

If `include_local_patch=true`, the current patch is also included
in the output. In this case the **orientation** and **rotation** (see below) is with
respect to the definition of the face (as given in the mesh topology).

# Notes
- Only applicable to 3D hexahedral meshes.
- Assumes consistent local face numbering on each patch.

# Arguments
- `mesh_topology::AbstractTopology{3,4}`: A 3D mesh topology with hexahedral patches.
- `patch_id::Int`: Global identifier of the patch.
- `face_local_id::Int`: Local face index of the patch.
- `include_local_patch::Bool=false`: Whether to include the current patch
  in the neighbour list.

# Returns
- `Matrix{Int}` of size `4 × N`, where `N` is the number of neighbouring patches.
   If `include_local_patch` is `true` then it contains `N+1` columns, i.e. the total
   number of patches that share this face, each row contains:
    1. **Neighbour patch ID**
    2. **Local face ID in the neighbour patch**
    3. **Rotation**: number of 90° clockwise vertex shifts required to align
    neighbour face DoF ordering with the current face
    4. **Orientation**: `+1` if aligned, `-1` if reversed, meaning that a transpose is required
    in the degree-of-freedom (DoF) numbering to ensure consistent orientation between DoFs of
    original patch (or edge definition if `include_local_patch = true`) and neighbours.

    **Note**: The sequence of vertices that define a face dictates its orientation.
"""
function compute_face_neighbours(
    mesh_topology::MT, patch_id::Int, face_local_id::Int; include_local_patch::Bool=false
) where {MT <: AbstractTopology{3, 4}}
    manifold_dim = 3  # we are in 3D
    patch_dimension = manifold_dim
    face_dimension = manifold_dim - 1
    vertex_dimension = 0

    # Determine the global face id and the
    # Get the faces of this patch
    @debug println("patch id: ", patch_id)
    # patch_faces = mesh_topology[patch_dimension + 1, face_dimension + 1][patch_id]
    # face_id = patch_faces[face_local_id]
    face_id = get_global_id(
        mesh_topology, patch_id, patch_dimension, face_local_id, face_dimension
    )

    num_vertices_per_face = length(
        mesh_topology[face_dimension + 1, vertex_dimension + 1][1]
    ) # the number of vertices per face (assumed to be the same for all facets)

    @debug println("   face $face_local_id id: ", face_id)

    # Determine how many neighbours the face has
    face_patch_neighbours_ids = mesh_topology[face_dimension + 1, patch_dimension + 1][abs(
        face_id
    )]

    if include_local_patch
        # All patches sharing the face are included
        n_face_neighbours = length(face_patch_neighbours_ids)
    else
        # The original patch, i.e., the one with `patch_id` is excluded
        n_face_neighbours = length(face_patch_neighbours_ids) - 1
    end

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
    for (k_neighbour, neighbour_patch_id) in enumerate(face_patch_neighbours_ids)
        @debug println("      neighbour patch id: ", neighbour_patch_id)

        if include_local_patch || (neighbour_patch_id ≠ patch_id)
            # Get the local id of the face in the neighbour patch
            neighbour_faces = mesh_topology[patch_dimension + 1, face_dimension + 1][neighbour_patch_id]
            for (neighbour_face_local_id, neighbour_face_id) in enumerate(neighbour_faces)
                @debug println("         neighbour face id: ", neighbour_face_id)

                if abs(neighbour_face_id) == abs(face_id)
                    # Store the global id of the neighbour patch
                    face_neighbours[1, k_neighbour] = neighbour_patch_id

                    # Store the local id of the face in the neighbour patch
                    face_neighbours[2, k_neighbour] = neighbour_face_local_id

                    # Store the orientation of the face relative to the neighbour patch

                    # Get the sequence of vertices of the face in the neighbour patch
                    neighbour_face_vertices_local_ids = mesh_topology.local_face2vertex[
                        :, neighbour_face_local_id
                    ]
                    neighbour_patch_vertices_ids = mesh_topology[
                        patch_dimension + 1, vertex_dimension + 1
                    ][neighbour_patch_id]
                    neighbour_face_vertices = neighbour_patch_vertices_ids[neighbour_face_vertices_local_ids]

                    if include_local_patch
                        # Get the sequence of vertices that define the face
                        patch_face_vertices = mesh_topology[
                            face_dimension + 1, vertex_dimension + 1
                        ][abs(face_id)]

                    else
                        # Get the sequence of vertices of the face in the current (original) patch
                        face_vertices_local_idx = mesh_topology.local_face2vertex[
                            :, face_local_id
                        ]
                        patch_vertices_ids = mesh_topology[
                            patch_dimension + 1, vertex_dimension + 1
                        ][patch_id]
                        patch_face_vertices = patch_vertices_ids[face_vertices_local_idx]
                    end
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
                    face_neighbours[3, k_neighbour] = base_vertex_neighbour_local_id_in_face

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
                        face_neighbours[4, k_neighbour] = -1  # the face is oriented in the opposite direction
                    elseif (
                        base_vertex_neighbour_local_id_in_face ==
                        (num_vertices_per_face - 1)
                    ) && (next_vertex_neighbour_local_id_in_face == 0)
                        face_neighbours[4, k_neighbour] = 1  # the face is oriented in the same direction
                    elseif next_vertex_neighbour_local_id_in_face >
                        base_vertex_neighbour_local_id_in_face
                        face_neighbours[4, k_neighbour] = 1  # the face is oriented in the same direction
                    else
                        face_neighbours[4, k_neighbour] = -1  # the face is oriented in the opposite direction
                    end

                    @debug println(
                        "            rotation: $(face_neighbours[3, k_neighbour]); orientation: $(face_neighbours[4, k_neighbour])\n",
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
    compute_face_neighbours(
        mesh_topology::AbstractTopology{3,4};
        include_local_patch::Bool = false,
    )

Return a matrix collecting the face-neighbour information for all patches
and all faces in the mesh.

The returned object is indexed as `[i, j]`, where entry `(i, j)` contains
the neighbour information for the `j`-th face of patch `i`, in the same
format as returned by [`compute_face_neighbours(::AbstractTopology{3,4}, ::Int, ::Int)`](@ref)

If `include_local_patch=true`, the current patch is also included
in the neighbour data. In this case the **orientation** and **rotation** (see below) is with
respect to the definition of the face (as given in the mesh topology).

# Notes
- Only applicable to 3D hexahedral meshes.
- Assumes consistent local face numbering across patches.

# Arguments
- `mesh_topology::AbstractTopology{3,4}`: A 3D mesh topology with hexahedral patches.
- `include_local_patch::Bool=false`: Whether to include the patch itself
  in the neighbour information for each face.

# Returns
- A matrix whose entry `(i, j)` contains the `4 × N` neighbour matrix
  associated with the `j`-th face of patch `i`. The neighbour matrix has the
  format described in [`compute_face_neighbours(::AbstractTopology{3,4}, ::Int, ::Int)`](@ref).
"""
function compute_face_neighbours(
    mesh_topology::MT; include_local_patch::Bool=false
) where {MT <: AbstractTopology{3, 4}}
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
                mesh_topology,
                patch_id,
                face_local_id;
                include_local_patch=include_local_patch,
            )
        end
    end
    return face_neighbours
end

"""
    compute_edge_neighbours(
        mesh_topology::AbstractTopology{2,3},
        patch_id::Int,
        edge_local_id::Int;
        include_local_patch::Bool = false,
    )

Return a `4 × N` matrix describing the neighbouring patches across
edge `edge_local_id` of patch `patch_id`.

Each column corresponds to one neighbouring patch.

If `include_local_patch=true`, the current patch will also be included
in the neighbour list.

# Notes
- Only applicable to 2D quadrilateral meshes.
- Assumes consistent local edge numbering across patches.

# Arguments
- `mesh_topology::MeshTopology{2,3}`: A 2D mesh topology with quadrilateral patches.
- `patch_id::Int`: Global identifier of the patch.
- `edge_local_id::Int`: Local edge index of the patch.
- `include_local_patch::Bool=false`: Whether to include the current patch
  in the neighbour information.

# Returns
- `Matrix{Int}` of size `4 × N`, where `N` is the number of neighbouring patches with
    each rown being
    1. **Neighbour patch ID**
    2. **Local edge ID in the neighbour patch**
    3. **Rotation**: always `0` (edges do not rotate; only orientation matters)
    4. **Orientation**: `+1` if aligned, `-1` if reversed.
    A value of `-1` indicates that the edge degree-of-freedom (DoF)
    numbering must be reversed to ensure consistent orientation.
"""
function compute_edge_neighbours(
    mesh_topology::MT, patch_id::Int, edge_local_id::Int; include_local_patch::Bool=false
) where {MT <: AbstractTopology{2, 3}}
    manifold_dim = 2  # we are in 2D
    patch_dimension = manifold_dim
    edge_dimension = manifold_dim - 1
    vertex_dimension = 0

    # Determine the global face id and the
    # Get the faces of this patch
    @debug println("patch id: ", patch_id)
    # patch_edges = mesh_topology[manifold_dim + 1, manifold_dim][patch_id]
    # edge_id = patch_edges[edge_local_id]
    edge_id = get_global_id(
        mesh_topology, patch_id, patch_dimension, edge_local_id, edge_dimension
    )

    num_vertices_per_edge = length(mesh_topology[manifold_dim, 1][1]) # the number of vertices per facet (assumed to be the same for all facets)

    @debug println("   edge $edge_local_id id: ", edge_id)

    # Determine how many neighbours the edge has
    edge_patch_neighbours_ids = mesh_topology[edge_dimension + 1, patch_dimension + 1][abs(
        edge_id
    )]

    if include_local_patch
        # All patches sharing the face are included
        n_edge_neighbours = length(edge_patch_neighbours_ids)
    else
        # The original patch, i.e., the one with `patch_id` is excluded
        n_edge_neighbours = length(edge_patch_neighbours_ids) - 1
    end

    if n_edge_neighbours == 0
        @debug println("      no neighbours")
        edge_neighbours = Matrix{Int}(undef, 4, 0)
        return edge_neighbours
    else
        edge_neighbours = zeros(Int, 4, n_edge_neighbours)
    end

    for (k_neighbour, neighbour_patch_id) in enumerate(edge_patch_neighbours_ids)
        @debug println("      neighbour patch id: ", neighbour_patch_id)

        if include_local_patch || (neighbour_patch_id ≠ patch_id)
            # Get the local id of the facet in the neighbour patch
            neighbour_edges = mesh_topology[manifold_dim + 1, manifold_dim][neighbour_patch_id]
            for (neighbour_edge_local_id, neighbour_edge_id) in enumerate(neighbour_edges)
                @debug println("         neighbour edge id: ", neighbour_edge_id)

                if abs(neighbour_edge_id) == abs(edge_id)
                    # Store the global id of the neighbour patch
                    edge_neighbours[1, k_neighbour] = neighbour_patch_id

                    # Store the local id of the facet in the neighbour patch
                    edge_neighbours[2, k_neighbour] = neighbour_edge_local_id

                    # Store the orientation of the facet relative to the neighbour patch

                    # Get the sequence of vertices of the facet in the current patch and its neighbour
                    neighbour_edge_vertices = mesh_topology[manifold_dim + 1, 1][neighbour_patch_id][mesh_topology.local_edge2vertex[
                        :, neighbour_edge_local_id
                    ]]

                    if include_local_patch
                        # Get the sequence of vertices that define the edge
                        patch_edge_vertices = mesh_topology[
                            edge_dimension + 1, vertex_dimension + 1
                        ][edge_id]

                    else
                        # Get the sequence of vertices of the edge in the current (original) patch
                        edge_vertices_local_idx = mesh_topology.local_edge2vertex[
                            :, edge_local_id
                        ]
                        patch_vertices_ids = mesh_topology[
                            patch_dimension + 1, vertex_dimension + 1
                        ][patch_id]
                        patch_edge_vertices = patch_vertices_ids[edge_vertices_local_idx]
                    end

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
                    edge_neighbours[3, k_neighbour] = 0  # There is no rotation since the facet is an edge, therefore there is only sign

                    if base_vertex_neighbour_local_id_in_edge == 0
                        edge_neighbours[4, k_neighbour] = 1  # the edge is oriented in the same direction
                    else
                        edge_neighbours[4, k_neighbour] = -1  # the edge is oriented in the opposite direction
                    end

                    @debug println(
                        "            rotation: $(edge_neighbours[3, k_neighbour]); orientation: $(edge_neighbours[4, k_neighbour])\n",
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
   compute_edge_neighbours(
        mesh_topology::AbstractTopology{3,4},
        patch_id::Int,
        edge_local_id::Int;
        include_local_patch::Bool = false,
    )

Return a `4 × N` matrix describing the neighbouring patches across
edge `edge_local_id` of patch `patch_id`.

Each column corresponds to one neighbouring patch.

If `include_local_patch=true`, the current patch will also be included
in the neighbour list.

# Notes
- Only applicable to 3D hexahedral meshes.
- Assumes consistent local edge numbering across patches.

# Arguments
- `mesh_topology::MeshTopology{3,4}`: A 3D mesh topology with hexahedral patches.
- `patch_id::Int`: Global identifier of the patch.
- `edge_local_id::Int`: Local edge index of the patch.
- `include_local_patch::Bool=false`: Whether to include the current patch
  in the neighbour information.

# Returns
- `Matrix{Int}` of size `4 × N`, where `N` is the number of neighbouring patches with
    each rown being
    1. **Neighbour patch ID**
    2. **Local edge ID in the neighbour patch**
    3. **Rotation**: always `0` (edges do not rotate; only orientation matters)
    4. **Orientation**: `+1` if aligned, `-1` if reversed.
    A value of `-1` indicates that the edge degree-of-freedom (DoF)
    numbering must be reversed to ensure consistent orientation.
"""
function compute_edge_neighbours(
    mesh_topology::MT, patch_id::Int, edge_local_id::Int; include_local_patch::Bool=false
) where {MT <: AbstractTopology{3, 4}}
    manifold_dim = 3  # we are in 2D
    patch_dimension = manifold_dim
    edge_dimension = manifold_dim - 2

    # Determine the global edge id
    # Get the edges of this patch
    @debug println("patch id: ", patch_id)
    # patch_edges = mesh_topology[manifold_dim + 1, 2][patch_id]
    # edge_id = patch_edges[edge_local_id]
    edge_id = get_global_id(
        mesh_topology, patch_id, patch_dimension, edge_local_id, edge_dimension
    )

    @debug println("   facet $edge_local_id id: ", edge_id)

    # Determine how many neighbours the edge has
    edge_patch_neighbours_ids = mesh_topology[edge_dimension + 1, patch_dimension + 1][abs(
        edge_id
    )]

    if include_local_patch
        # All patches sharing the face are included
        n_edge_neighbours = length(edge_patch_neighbours_ids)
    else
        # The original patch, i.e., the one with `patch_id` is excluded
        n_edge_neighbours = length(edge_patch_neighbours_ids) - 1
    end

    if n_edge_neighbours == 0
        @debug println("      no neighbours")
        edge_neighbours = Matrix{Int}(undef, 4, 0)
        return edge_neighbours
    else
        edge_neighbours = zeros(Int, 4, n_edge_neighbours)
    end

    for (k_neighbour, neighbour_patch_id) in enumerate(edge_patch_neighbours_ids)
        @debug println("      neighbour patch id: ", neighbour_patch_id)

        if include_local_patch || (neighbour_patch_id ≠ patch_id)
            # Get the local id of the facet in the neighbour patch
            neighbour_edges = mesh_topology[manifold_dim + 1, 2][neighbour_patch_id]
            for (neighbour_edge_local_id, neighbour_edge_id) in enumerate(neighbour_edges)
                @debug println("         neighbour edge id: ", neighbour_edge_id)

                if abs(neighbour_edge_id) == abs(edge_id)
                    # Store the global id of the neighbour patch
                    edge_neighbours[1, k_neighbour] = neighbour_patch_id

                    # Store the local id of the facet in the neighbour patch
                    edge_neighbours[2, k_neighbour] = neighbour_edge_local_id

                    # Store the orientation of the facet relative to the neighbour patch

                    # Get the sequence of vertices of the facet in the current patch and its neighbour
                    neighbour_edge_vertices = mesh_topology[manifold_dim + 1, 1][neighbour_patch_id][mesh_topology.local_edge2vertex[
                        :, neighbour_edge_local_id
                    ]]

                    if include_local_patch
                        # Get the sequence of vertices that define the edge
                        patch_edge_vertices = mesh_topology[
                            edge_dimension + 1, vertex_dimension + 1
                        ][edge_id]

                    else
                        # Get the sequence of vertices of the edge in the current (original) patch
                        edge_vertices_local_idx = mesh_topology.local_edge2vertex[
                            :, edge_local_id
                        ]
                        patch_vertices_ids = mesh_topology[
                            patch_dimension + 1, vertex_dimension + 1
                        ][patch_id]
                        patch_edge_vertices = patch_vertices_ids[edge_vertices_local_idx]
                    end

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
                    edge_neighbours[3, k_neighbour] = 0  # There is no rotation since the facet is an edge, therefore there is only sign

                    if base_vertex_neighbour_local_id_in_edge == 0
                        edge_neighbours[4, k_neighbour] = 1  # the edge is oriented in the same direction
                    else
                        edge_neighbours[4, k_neighbour] = -1  # the edge is oriented in the opposite direction
                    end

                    @debug println(
                        "            rotation: $(edge_neighbours[3, k_neighbour]); orientation: $(edge_neighbours[4, k_neighbour])\n",
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
    compute_edge_neighbours(
        mesh_topology::AbstractTopology{3,4};
        include_local_patch::Bool = false,
    )

Return a matrix collecting the edge-neighbour information for all patches
and all edges in the mesh.

The returned object is indexed as `[i, j]`, where entry `(i, j)` contains
the neighbour information for the `j`-th edge of patch `i`, in the same
format as returned by [`compute_edge_neighbours(::AbstractTopology{3,4}, ::Int, ::Int)`](@ref).

If `include_local_patch=true`, the current patch is also included
in the neighbour data. In this case the **orientation** (see below) is with
respect to the definition of the edge (as given in the mesh topology).

# Notes
- Only applicable to 3D hexahedral meshes.
- Assumes consistent local edge numbering across patches.

# Arguments
- `mesh_topology::AbstractTopology{3,4}`: A 3D mesh topology with hexahedral patches.
- `include_local_patch::Bool=false`: Whether to include the patch itself
  in the neighbour information for each edge.

# Returns
- A matrix whose entry `(i, j)` contains the `4 × N` neighbour matrix
  associated with the `j`-th edge of patch `i`. The neighbour matrix has the
  format described in [`compute_edge_neighbours(::AbstractTopology{3,4}, ::Int, ::Int)`](@ref).
"""
function compute_edge_neighbours(
    mesh_topology::MT
) where {
    manifold_dim,
    incidence_relations_dim,
    MT <: AbstractTopology{manifold_dim, incidence_relations_dim},
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
                mesh_topology,
                patch_id,
                edge_local_id;
                include_local_patch=include_local_patch,
            )
        end
    end
    return edge_neighbours
end

"""
    compute_vertex_neighbours(
        mesh_topology::MeshTopology,
        patch_id::Int,
        vertex_local_id::Int;
        include_local_patch::Bool = false,
    )

Return a `4 × N` matrix describing the neighbouring patches sharing
vertex `vertex_local_id` of patch `patch_id`.

Each column corresponds to one neighbouring patch.

If `include_local_patch=true`, the current patch is also included
in the neighbour list.

# Notes
- Applicable to all supported mesh topologies.
- Assumes consistent local vertex numbering across patches.

# Arguments
- `mesh_topology::MeshTopology`: A mesh topology of any supported type.
- `patch_id::Int`: Global identifier of the patch.
- `vertex_local_id::Int`: Local vertex index of the patch.
- `include_local_patch::Bool=false`: Whether to include the current patch
  in the neighbour list.

# Returns
- `Matrix{Int}` of size `4 × N`, where `N` is the number of neighbouring patches.
  If `include_local_patch` is `true` then it contains `N+1` columns, i.e. the total
  number of patches that share this vertex. Each row contains:
    1. **Neighbour patch ID**
    2. **Local vertex ID in the neighbour patch**
    3. **Rotation**: always `0` (vertices have no rotation required to match DoFs)
    4. **Orientation**: always `1` (vertices have no orientation required to match DoFs)
"""
function compute_vertex_neighbours(
    mesh_topology::MT, patch_id::Int, vertex_local_id::Int; include_local_patch::Bool=false
) where {
    manifold_dim,
    incidence_relations_dim,
    MT <: AbstractTopology{manifold_dim, incidence_relations_dim},
}
    patch_dimension = manifold_dim
    vertex_dimension = 0

    # Determine the global vertex id
    @debug println("patch id: ", patch_id)
    # patch_vertices = mesh_topology[manifold_dim + 1, 1][patch_id]
    # vertex_id = patch_vertices[vertex_local_id]
    vertex_id = get_global_id(
        mesh_topology, patch_id, patch_dimension, vertex_local_id, vertex_dimension
    )

    @debug println("   vertex $vertex_local_id id: ", vertex_id)

    # Determine how many neighbours the vertex has
    vertex_patch_neighbours_ids = mesh_topology[vertex_dimension + 1, patch_dimension + 1][abs(
        vertex_id
    )]
    if include_local_patch
        # All patches sharing the face are included
        n_vertex_neighbours = length(vertex_patch_neighbours_ids)
    else
        # Exclude the patch used to identify to vertex, i.e., the original patch
        n_vertex_neighbours = length(vertex_patch_neighbours_ids) - 1
    end

    if n_vertex_neighbours == 0
        @debug println("      no neighbours")
        vertex_neighbours = Matrix{Int}(undef, 4, 0)
        return vertex_neighbours
    else
        vertex_neighbours = zeros(Int, 4, n_vertex_neighbours)
    end

    for (k_neighbour, neighbour_patch_id) in enumerate(vertex_patch_neighbours_ids)
        @debug println("      neighbour patch id: ", neighbour_patch_id)

        if include_local_patch || (neighbour_patch_id ≠ patch_id)
            # Get the local id of the facet in the neighbour patch
            neighbour_vertices = mesh_topology[manifold_dim + 1, 1][neighbour_patch_id]
            for (neighbour_vertex_local_id, neighbour_vertex_id) in
                enumerate(neighbour_vertices)
                @debug println("         neighbour vertex id: ", neighbour_vertex_id)

                if abs(neighbour_vertex_id) == abs(vertex_id)
                    # Store the global id of the neighbour patch
                    vertex_neighbours[1, k_neighbour] = neighbour_patch_id

                    # Store the local id of the facet in the neighbour patch
                    vertex_neighbours[2, k_neighbour] = neighbour_vertex_local_id

                    # Store the orientation of the facet relative to the neighbour patch
                    # In 1D this is trivial, since no rotation is needed and there is no sign
                    vertex_neighbours[3, k_neighbour] = 0  # there is no rotation since the facet is a vertex
                    vertex_neighbours[4, k_neighbour] = 1  # there is also no sign (use 1 to signal that, for consistency)

                    @debug println(
                        "            rotation: $(vertex_neighbours[3, k_neighbour]); orientation: $(vertex_neighbours[4, k_neighbour])\n",
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
    compute_vertex_neighbours(
        mesh_topology::AbstractTopology;
        include_local_patch::Bool = false
    )

Return a matrix collecting the vertex-neighbour information for all patches
and all vertices in the mesh.

The returned object is indexed as `[i, j]`, where entry `(i, j)` contains
the neighbour information for the `j`-th vertex of patch `i`, in the same
format as returned by [`compute_vertex_neighbours(::AbstractTopology, ::Int, ::Int)`](@ref).

If `include_local_patch=true`, the current patch is also included
in the neighbour data.

# Notes
- Applicable to all supported mesh topologies.
- Assumes consistent local vertex numbering across patches.

# Arguments
- `mesh_topology::AbstractTopology`: A mesh topology of any supported type.
- `include_local_patch::Bool=false`: Whether to include the patch itself
  in the neighbour information for each vertex.

# Returns
- A matrix whose entry `(i, j)` contains the `4 × N` neighbour matrix
  associated with the `j`-th vertex of patch `i`. The neighbour matrix has the
  format described in [`compute_vertex_neighbours(::AbstractTopology, ::Int, ::Int)`](@ref).
"""
function compute_vertex_neighbours(mesh_topology::AbstractTopology)
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

include("MeshTopology.jl")
include("SkeletonTopology.jl")

end

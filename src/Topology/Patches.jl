############################################################################################
#                                      Abstract Types                                      #
############################################################################################
abstract type AbstractPatch{manifold_dim, incidence_relations_dim, num_patch_vertices} end
abstract type AbstractTensorProductPatch{
    manifold_dim, incidence_relations_dim, num_patch_vertices
} <: AbstractPatch{manifold_dim, incidence_relations_dim, num_patch_vertices} end

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



      get_local_incidence_relations(n_patch_vertices::Int)

  Return the local incidence relations for a patch type determined by its number of vertices.

  Supported patch types and their local vertex, edge, and face numbering conventions are:

  Supported patch types are:
  - `2` vertices: 1D line element (L2)
  - `4` vertices: 2D quadrilateral element (Q4)
  - `8` vertices: 3D hexahedral element (H8)

  # Arguments
  - `num_patch_vertices::Int`: Number of vertices in the patch.

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
  - `ArgumentError`: If `num_patch_vertices` does not correspond to a supported patch type.


  # # Indexing.
  # i and k and geometric dimension indices, i.e., if geometric dimension is n
  # then the index is (n + 1), this is because julia starts indices at 1 and vertices
  # have dimension 0.

  # Sizes.
  # Provide quick access to the number of local geometric objects in each dimension
  # (vertices, edges) in 1D
  # (vertices, edges, surfaces) in 2D
  # (vertices, edges, surfaces, volumes) in 3D
  # geometric_dim_id is the index associated to the geometric dimension. Geometric dimension n
  # has index (n + 1), this is done to keep consistency with julia indices that start at 1 and
  # not at 0 (vertices have geometric dimension 0).


"""

############################################################################################
#                                        Structures                                        #
############################################################################################
"""
**1D line element (L2, 2 vertices, 1 edge)**

Vertices:
```
1 --------- 2 --- ξ
```
Edge 1 is the element itself, oriented 1 → 2.
"""
struct Line{MCT} <: AbstractTensorProductPatch{1, 2, 2}
    MeshCorePatch::MCT
    incidence_relations::NTuple{2, NTuple{2, Vector{Vector{Int}}}}
    n_geometric_objects::NTuple{2, Int}
    map_id_to_position::Dict{NTuple{3, Int}, NTuple{1, Int}}
    map_position_to_id::Dict{NTuple{1, Int}, Int}
    dof_placement::Dict{NTuple{3, Int}, Int}

    function Line()
        incidence_relations = (
            (
                Vector{Vector{Int}}(),  # [1][1]: vertex to vertex
                [[1], [1]],             # [1][2]: vertex to edge
            ), (
                [[1, 2]],               # [2][1]: edge to vertex
                Vector{Vector{Int}}(),  # [2][2]: edge to edge
            )
        )

        n_geometric_objects = (2, 1)

        map_id_to_position = Dict(
            # Vertex numbering
            (1, 0, 1) => (-1,),
            (1, 0, 2) => (1,),
            # Edge numbering
            (1, 1, 1) => (0,),
        )

        map_position_to_id = Dict(
            # Vertex numbering
            (-1,) => 1,
            (1,) => 2,
            # Edge numbering
            (0,) => 1,
        )

        dof_placement = Dict(
            # Vertex numbering
            (1, 0, 1) => 1,
            (1, 0, 2) => 3,
            # Edge numbering
            (1, 1, 1) => 2,
        )

        return new{typeof(MeshCore.L2)}(
            MeshCore.L2,
            incidence_relations,
            n_geometric_objects,
            map_id_to_position,
            map_position_to_id,
            dof_placement,
        )
    end
end

"""
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
"""
struct Quad{MCT} <: AbstractTensorProductPatch{2, 3, 4}
    MeshCorePatch::MCT
    incidence_relations::NTuple{3, NTuple{3, Vector{Vector{Int}}}}
    n_geometric_objects::NTuple{3, Int}
    map_id_to_position::Dict{NTuple{3, Int}, NTuple{2, Int}}
    map_position_to_id::Dict{NTuple{2, Int}, Int}
    dof_placement::Dict{NTuple{3, Int}, Int}

    function Quad()
        incidence_relations = (
            (
                Vector{Vector{Int}}(),            # [1][1]: vertex to vertex
                [[1, 3], [2, 3], [2, 4], [1, 4]], # [1][2]: vertex to edge
                [[1], [1], [1], [1]],             # [1][3]: vertex to face
            ),
            (
                [[1, 4], [2, 3], [1, 2], [4, 3]],  # [2][1]: edge to vertex
                Vector{Vector{Int}}(),             # [2][2]: edge to edge
                [[1], [1], [1], [1]],              # [2][3]: edge to face
            ),
            (
                [[1, 2, 3, 4]],         # [3][1]: face to vertex
                [[1, 2, 3, 4]],         # [3][2]: face to edge
                Vector{Vector{Int}}(),  # [3][3]: face to face
            ),
        )

        n_geometric_objects = (4, 4, 1)

        map_id_to_position = Dict(
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
        )

        map_position_to_id = Dict(
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
        )

        dof_placement = Dict(
            # Vertex numbering
            (2, 0, 1) => 1,
            (2, 0, 2) => 3,
            (2, 0, 3) => 9,
            (2, 0, 4) => 7,
            # Edge numbering
            (2, 1, 1) => 4,
            (2, 1, 2) => 6,
            (2, 1, 3) => 2,
            (2, 1, 4) => 8,
            # Face numbering
            (2, 2, 1) => 5,
        )

        return new{typeof(MeshCore.Q4)}(
            MeshCore.Q4,
            incidence_relations,
            n_geometric_objects,
            map_id_to_position,
            map_position_to_id,
            dof_placement,
        )
    end
end

"""
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

Faces (cyclic vertex order indicates orientation, pointing outwards):

- Face 1 (f1, back face, not visible): 1 → 5 → 8 → 4 (ξ = min)
- Face 2 (f2, front face): 2 → 3 → 7 → 6 (ξ = max)
- Face 3 (f3, left face): 1 → 2 → 6 → 5 (η = min)
- Face 4 (f4, right face): 4 → 8 → 7 → 3 (η = max)
- Face 5 (f5, bottom face): 1 → 4 → 3 → 2 (ζ = min)
- Face 6 (f6, top face): 5 → 6 → 7 → 8 (ζ = max)
"""
struct Hex{MCT} <: AbstractTensorProductPatch{3, 4, 8}
    MeshCorePatch::MCT
    incidence_relations::NTuple{4, NTuple{4, Vector{Vector{Int}}}}
    n_geometric_objects::NTuple{4, Int}
    map_id_to_position::Dict{NTuple{3, Int}, NTuple{3, Int}}
    map_position_to_id::Dict{NTuple{3, Int}, Int}
    dof_placement::Dict{NTuple{3, Int}, Int}

    function Hex()
        incidence_relations = (
            (
                Vector{Vector{Int}}(),  # [1][1]: vertex to vertex
                [
                    [3, 7, 11],
                    [3, 8, 12],
                    [4, 8, 9],
                    [4, 7, 10],
                    [2, 6, 11],
                    [2, 5, 12],
                    [1, 5, 9],
                    [1, 6, 10],
                ],              # [1][2]: vertex to edge
                [
                    [1, 3, 5],
                    [2, 3, 5],
                    [2, 5, 4],
                    [1, 4, 5],
                    [1, 3, 6],
                    [2, 3, 6],
                    [2, 4, 6],
                    [1, 4, 6],
                ],            # [1][3]: vertex to face
                [[1], [1], [1], [1], [1], [1], [1], [1]],  # [1][4]: vertex to volume
            ),
            (
                [
                    [8, 7],
                    [5, 6],
                    [1, 2],
                    [4, 3],
                    [6, 7],
                    [5, 8],
                    [1, 4],
                    [2, 3],
                    [3, 7],
                    [4, 8],
                    [1, 5],
                    [2, 6],
                ],                      # [2][1]: edge to vertex
                Vector{Vector{Int}}(),  # [2][2]: edge to edge
                [
                    [4, 6],
                    [3, 6],
                    [3, 5],
                    [4, 5],
                    [2, 6],
                    [1, 6],
                    [1, 5],
                    [2, 5],
                    [2, 4],
                    [1, 4],
                    [1, 3],
                    [2, 3],
                ],            # [2][3]: edge to face
                [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], # [2][4]: edge to volume
            ),
            (
                [
                    [1, 5, 8, 4],
                    [2, 3, 7, 6],
                    [1, 2, 6, 5],
                    [4, 8, 7, 3],
                    [1, 4, 3, 2],
                    [5, 6, 7, 8],
                ],               # [3][1]: face to vertex
                [
                    [11, 10, 7, 6],
                    [12, 9, 8, 5],
                    [11, 12, 3, 2],
                    [10, 9, 4, 1],
                    [7, 8, 3, 4],
                    [6, 5, 2, 1],
                ],                                          # [3][2]: face to edge
                Vector{Vector{Int}}(),                      # [3][3]: face to face
                [[1], [1], [1], [1], [1], [1]],             # [3][4]: face to volume
            ),
            (
                [[1, 2, 3, 4, 5, 6, 7, 8]],                 # [4][1]: volume to vertex
                [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],  # [4][2]: volume to edge
                [[1, 2, 3, 4, 5, 6]],                       # [4][3]: volume to face
                Vector{Vector{Int}}(),                      # [4][4]: volume to volume
            ),
        )

        n_geometric_objects = (8, 12, 6, 1)

        map_id_to_position = Dict(
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

        map_position_to_id = Dict(
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

        dof_placement = Dict(
            # Vertex numbering
            (3, 0, 1) => 1,
            (3, 0, 2) => 3,
            (3, 0, 3) => 9,
            (3, 0, 4) => 7,
            (3, 0, 5) => 19,
            (3, 0, 6) => 21,
            (3, 0, 7) => 27,
            (3, 0, 8) => 25,
            # Edge numbering
            (3, 1, 1) => 26,
            (3, 1, 2) => 20,
            (3, 1, 3) => 2,
            (3, 1, 4) => 8,
            (3, 1, 5) => 24,
            (3, 1, 6) => 22,
            (3, 1, 7) => 4,
            (3, 1, 8) => 6,
            (3, 1, 9) => 18,
            (3, 1, 10) => 16,
            (3, 1, 11) => 10,
            (3, 1, 12) => 12,
            # Face numbering
            (3, 2, 1) => 13,
            (3, 2, 2) => 15,
            (3, 2, 3) => 11,
            (3, 2, 4) => 17,
            (3, 2, 5) => 5,
            (3, 2, 6) => 23,
            # Volume numbering
            (3, 3, 1) => 14,
        )

        return new{typeof(MeshCore.H8)}(
            MeshCore.H8,
            incidence_relations,
            n_geometric_objects,
            map_id_to_position,
            map_position_to_id,
            dof_placement,
        )
    end
end

const LINE = Line()
const QUAD = Quad()
const HEX = Hex()

############################################################################################
#                                         Getters                                          #
############################################################################################
get_meshcore_patch(patch::AbstractTensorProductPatch) = patch.MeshCorePatch
get_incidence_relations(patch::AbstractTensorProductPatch) = patch.incidence_relations
get_dof_placement(patch::AbstractTensorProductPatch) = patch.dof_placement

############################################################################################
#                                         Indexing                                         #
############################################################################################
Base.getindex(patch::AbstractTensorProductPatch, i::Int, k::Int) =
    get_incidence_relations(patch)[i][k]

############################################################################################
#                                          Sizes                                           #
############################################################################################
Base.size(patch::AbstractTensorProductPatch) = patch.n_geometric_objects
Base.size(patch::AbstractTensorProductPatch, geometric_dim_id::Int) =
    size(patch)[geometric_dim_id]

############################################################################################
#                                   ID <-> position/dofs                                   #
############################################################################################
function id2position(
    patch::AbstractTensorProductPatch{manifold_dim}, object_dim::Int, object_local_id::Int
) where {manifold_dim}
    if object_local_id <= 0
        return throw(
            ArgumentError(
                LazyString(
                    "Local object ids must be at least 1. Got ", object_local_id, "."
                ),
            ),
        )
    end
    return patch.map_id_to_position[(manifold_dim, object_dim, object_local_id)]
end

function position2id(
    patch::AbstractTensorProductPatch{manifold_dim}, position::NTuple{manifold_dim, Int}
) where {manifold_dim}
    return patch.map_position_to_id[position]
end

# DOF placement per patch.
function id_to_dof_division(
    patch::AbstractTensorProductPatch{manifold_dim}, object_dim::Int, object_local_id::Int
) where {manifold_dim}
    dof_placement = get_dof_placement(patch)
    return dof_placement[(manifold_dim, object_dim, abs(object_local_id))]
end

############################################################################################
#                                     Skeleton Patches                                     #
############################################################################################
"""
    get_skeleton_patch(patch::AbstractPatch)

Return the [`AbstractPatch`](@ref) that makes up the patches of the skeleton of the `patch`.
Return `nothing` if there is no skeleton patch.
"""
get_skeleton_patch(patch::AbstractPatch) = throw(MethodError(get_skeleton_patch, (patch,)))
get_skeleton_patch(::Line) = nothing
get_skeleton_patch(::Quad) = LINE
get_skeleton_patch(::Hex) = QUAD

module Topology

import MeshCore

include("Patches.jl")

############################################################################################
#                                      Abstract Types                                      #
############################################################################################
"""
    AbstractTopology{manifold_dim, incidence_relations_dim, num_patches, PT}

Abstract type of all topologies that represent the topological structure of a collection of
patches (of equal shape) forming a mesh. This includes the skeleton mesh.

Each patch is considered an individual mesh element at the global level. These structures
enable the computation of all incidence relations between geometric objects (vertices,
edges, faces, volumes), and the determination of topological neighbours. Supports 1D
(lines), 2D (quads), and 3D (hexahedra) topologies.

The geometric objects present depend on the manifold dimension:
- 1D: vertices, lines (patches);
- 2D: vertices, lines (facets), surfaces (patches);
- 3D: vertices, lines (edges), surfaces (facets), volumes (patches).

A relation between objects of dimensions `n` and `m` is indexed as `topology[n + 1, m + 1]`,
since Julia indices start at 1 while vertices have dimension 0.
"""
abstract type AbstractTopology{manifold_dim, incidence_relations_dim, num_patches, PT} end

############################################################################################
#                                         Getters                                          #
############################################################################################
function get_manifold_dim(::AbstractTopology{manifold_dim}) where {manifold_dim}
    return manifold_dim
end
function get_incidence_relations_dim(
    ::AbstractTopology{manifold_dim, incidence_relations_dim}
) where {manifold_dim, incidence_relations_dim}
    return incidence_relations_dim
end
function get_num_patches(
    ::AbstractTopology{manifold_dim, incidence_relations_dim, num_patches}
) where {manifold_dim, incidence_relations_dim, num_patches}
    return num_patches
end
function get_patch_type(
    ::AbstractTopology{manifold_dim, incidence_relations_dim, num_patches, PT}
) where {manifold_dim, incidence_relations_dim, num_patches, PT}
    return PT
end

"""
    get_topological_patch(topology::AbstractTopology)

Return the [`AbstractPatch`](@ref) object that each patch in this topology is made of.
"""
get_topological_patch(topology::AbstractTopology) = topology.topological_patch

############################################################################################
#                                         Indexing                                         #
############################################################################################
function Base.lastindex(::AbstractTopology{manifold_dim}, d::Int=1) where {manifold_dim}
    return manifold_dim + 1
end

############################################################################################
#                                          Sizes                                           #
############################################################################################
"""
    get_local_size(topology::AbstractTopology)
    get_local_size(topology::AbstractTopology, geometric_dim_id::Int)

Return the number of local geometric objects of a patch, which is the same for all patches.
If a `geometric_dim_id` is given, return only the number of objects of dimension
`geometric_dim_id - 1` (e.g. `geometric_dim_id = 1` corresponds to vertices, which have
geometric dimension 0).
"""
get_local_size(topology::AbstractTopology) = size(get_topological_patch(topology))
function get_local_size(topology::AbstractTopology, geometric_dim_id::Int)
    return size(get_topological_patch(topology), geometric_dim_id)
end

############################################################################################
#                                Boundaries and interfaces                                 #
############################################################################################
"""
    get_boundaries_and_interfaces(topology::AbstractTopology)

Return the tuple `(boundaries, interfaces)` classifying every geometric object of dimension
lower than the manifold dimension of `topology`.

Both are vectors of `(object_dim, global_object_id)` tuples: an object is a *boundary* object
when it belongs to exactly one patch, and an *interface* object when it is shared by several.
Objects are listed from the highest dimension down to vertices.
"""
function get_boundaries_and_interfaces(
    topology::AbstractTopology{manifold_dim}
) where {manifold_dim}
    boundaries = Tuple{Int, Int}[]
    interfaces = Tuple{Int, Int}[]
    for dim in (manifold_dim - 1):-1:0
        # Check which patches (of dimension manifold_dim) the current bounding objects
        # belong to.
        object2patch = topology[dim + 1, manifold_dim + 1]
        for object_id in eachindex(object2patch)
            # If it is not shared, this object has only 1 patch in its list.
            if length(object2patch[object_id]) == 1
                push!(boundaries, (dim, object_id))
            else
                push!(interfaces, (dim, object_id))
            end
        end
    end

    return boundaries, interfaces
end

"""
    get_interfaces_on_boundary(topology::AbstractTopology)

Return the vector of `(object_dim, global_object_id)` tuples identifying the interface
objects of `topology` that bound at least one boundary object, e.g. in 2D the vertices that
are shared by several patches and that lie on a boundary edge.

See [`get_boundaries_and_interfaces`](@ref) for the classification into boundaries and
interfaces.
"""
function get_interfaces_on_boundary(topology::AbstractTopology)
    boundaries, interfaces = get_boundaries_and_interfaces(topology)
    # The lookup below is done once per containing object, so use a set rather than the
    # vector returned above.
    boundary_set = Set(boundaries)

    interfaces_on_boundary = Tuple{Int, Int}[]
    for (dim, global_id) in interfaces
        # Check the objects of dimension dim + 1 that the current interface bounds.
        containing_objects = topology[dim + 1, dim + 2][global_id]
        for containing_id in containing_objects
            if (dim + 1, abs(containing_id)) in boundary_set
                push!(interfaces_on_boundary, (dim, global_id))
                break
            end
        end
    end

    return interfaces_on_boundary
end

############################################################################################
#                                 Local/global conversions                                 #
############################################################################################
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

    if container_dim == local_dim
        return throw(ArgumentError("Container_dim and local_dim cannot be the same."))
    end
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
function get_global_id(topology::AbstractTopology, patch_id, local_id, local_dim)
    return get_global_id(
        topology, patch_id, get_manifold_dim(topology), local_id, local_dim
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
index `global_object_id` within patch `patch_id`.

The returned index is negative when the patch traverses the object in the opposite direction
to its global definition. Both `global_object_id` and the stored indices are matched on their
absolute values, since the sign encodes orientation rather than identity.

# Arguments
- `topology::AbstractTopology`: The mesh topology.
- `patch_id::Int`: Global index of the patch.
- `global_object_id::Int`: Global index of the target geometric object.
- `object_dim::Int`: Topological dimension of the target object, which must be smaller than
  the manifold dimension.

# Returns
- `Int`: Signed local index of the target object within the patch.

# Throws
- `ArgumentError`: If `object_dim` is the manifold dimension, or if the object does not
  belong to the patch.

# Example
To obtain the local index of the edge (dimension 1) with global index 7 within patch 3:
```julia
get_local_id(topology, 3, 7, 1)
```
"""
function get_local_id(topology::AbstractTopology, patch_id, global_object_id, object_dim)
    manifold_dim = get_manifold_dim(topology)
    if object_dim == manifold_dim
        throw(
            ArgumentError(
                LazyString(
                    "Mantis.Topology.get_local_id: objects of dimension ",
                    object_dim,
                    " are the patches themselves, so they have no local id within a patch.",
                ),
            ),
        )
    end
    incidence_relation = topology[manifold_dim + 1, object_dim + 1][patch_id]
    # First get the local id while ignoring the sign.
    local_object_id = findfirst(
        object_id -> abs(object_id) == abs(global_object_id), incidence_relation
    )

    if isnothing(local_object_id)
        throw(
            ArgumentError(
                LazyString(
                    "Mantis.Topology.get_local_id: no local object id found for inputs:",
                    " patch_id ",
                    patch_id,
                    ", global_object_id ",
                    global_object_id,
                    ", object_dim ",
                    object_dim,
                    ". The available global topological objects on this patch are ",
                    incidence_relation,
                    ".",
                ),
            ),
        )
    end

    # Then get the sign
    local_object_id = local_object_id * sign(incidence_relation[local_object_id])

    return local_object_id
end

############################################################################################
#                                   Neighbour information                                  #
############################################################################################
# All neighbour queries share the same shape: given a geometric object of a patch, find the
# other patches that contain that same object, and describe how each of them traverses it
# relative to a reference traversal. The dimension-specific entry points below therefore all
# forward to `_compute_object_neighbours`.

# Reference/neighbour vertex sequences are unused for vertices; this avoids allocating an
# empty vector on every such query.
const NO_VERTICES = Int[]

"""
    _object_vertices_in_patch(topology, patch_id, object_local_id, object_dim)

Return the global vertex ids of the local object with dimension `object_dim` and local index
`object_local_id` of patch `patch_id`, ordered as prescribed by the topological patch.
"""
function _object_vertices_in_patch(
    topology::AbstractTopology{manifold_dim},
    patch_id::Int,
    object_local_id::Int,
    object_dim::Int,
) where {manifold_dim}
    local_vertex_ids = get_topological_patch(topology)[object_dim + 1, 1][object_local_id]
    patch_vertex_ids = topology[manifold_dim + 1, 1][patch_id]
    return patch_vertex_ids[local_vertex_ids]
end

"""
    _reference_vertices(
        topology, patch_id, object_local_id, object_dim, object_id, include_local_patch
    )

Return the vertex sequence that neighbours are compared against when computing rotation and
orientation.

When `include_local_patch` is `false` the reference is how patch `patch_id` traverses the
object, so that the result describes each neighbour relative to the current patch. When it is
`true` the current patch is itself reported as a neighbour, so the reference is instead the
global definition of the object as stored in `topology`.
"""
function _reference_vertices(
    topology::AbstractTopology,
    patch_id::Int,
    object_local_id::Int,
    object_dim::Int,
    object_id::Int,
    include_local_patch::Bool,
)
    object_dim == 0 && return NO_VERTICES
    include_local_patch && return topology[object_dim + 1, 1][object_id]
    return _object_vertices_in_patch(topology, patch_id, object_local_id, object_dim)
end

"""
    _cyclic_position(vertex_id, reference_vertices)

Return the 0-based position of `vertex_id` within `reference_vertices`.
"""
function _cyclic_position(vertex_id::Int, reference_vertices::AbstractVector{Int})
    position = findfirst(==(vertex_id), reference_vertices)
    if isnothing(position)
        throw(
            ArgumentError(
                LazyString(
                    "Mantis.Topology: vertex ",
                    vertex_id,
                    " of a neighbouring patch is not part of the shared object with",
                    " vertices ",
                    reference_vertices,
                    ". The topology is inconsistent.",
                ),
            ),
        )
    end
    return position - 1
end

"""
    _rotation_and_orientation(neighbour_vertices, reference_vertices, object_dim)

Return the `(rotation, orientation)` pair describing how `neighbour_vertices` traverses an
object of dimension `object_dim` relative to `reference_vertices`.

`rotation` is the number of positions by which the neighbour's vertex sequence is shifted
with respect to the reference, and `orientation` is `1` when both traverse the object in the
same cyclic direction and `-1` otherwise.
"""
function _rotation_and_orientation(
    neighbour_vertices::AbstractVector{Int},
    reference_vertices::AbstractVector{Int},
    object_dim::Int,
)
    # Vertices have neither rotation nor orientation.
    object_dim == 0 && return (0, 1)

    base_position = _cyclic_position(neighbour_vertices[1], reference_vertices)

    # An edge has only two vertices, so the two cyclic directions coincide and the traversal
    # is fully determined by which vertex comes first.
    object_dim == 1 && return (0, base_position == 0 ? 1 : -1)

    num_vertices = length(reference_vertices)
    next_position = _cyclic_position(neighbour_vertices[2], reference_vertices)
    is_aligned = next_position == mod(base_position + 1, num_vertices)

    return (base_position, is_aligned ? 1 : -1)
end

"""
    _compute_object_neighbours(
        topology, patch_id, object_local_id, object_dim, include_local_patch
    )

Return the `4 × N` neighbour matrix of the local object with dimension `object_dim` and
local index `object_local_id` of patch `patch_id`. See [`compute_neighbours`](@ref) for the
meaning of the rows.
"""
function _compute_object_neighbours(
    topology::AbstractTopology{manifold_dim},
    patch_id::Int,
    object_local_id::Int,
    object_dim::Int,
    include_local_patch::Bool,
) where {manifold_dim}
    # The sign of a global id encodes orientation, which is recomputed below, so drop it.
    object_id = abs(get_global_id(topology, patch_id, object_local_id, object_dim))
    neighbour_patch_ids = topology[object_dim + 1, manifold_dim + 1][object_id]

    # Counting (rather than subtracting one) also covers patches that touch the same object
    # more than once, as happens in periodic meshes with a single patch.
    num_neighbours = if include_local_patch
        length(neighbour_patch_ids)
    else
        count(!=(patch_id), neighbour_patch_ids)
    end
    num_neighbours == 0 && return Matrix{Int}(undef, 4, 0)

    reference_vertices = _reference_vertices(
        topology, patch_id, object_local_id, object_dim, object_id, include_local_patch
    )

    neighbours = Matrix{Int}(undef, 4, num_neighbours)
    column = 1
    for neighbour_patch_id in neighbour_patch_ids
        (include_local_patch || neighbour_patch_id != patch_id) || continue

        neighbour_local_id = abs(
            get_local_id(topology, neighbour_patch_id, object_id, object_dim)
        )
        neighbour_vertices = if object_dim == 0
            NO_VERTICES
        else
            _object_vertices_in_patch(
                topology, neighbour_patch_id, neighbour_local_id, object_dim
            )
        end
        rotation, orientation = _rotation_and_orientation(
            neighbour_vertices, reference_vertices, object_dim
        )

        neighbours[1, column] = neighbour_patch_id
        neighbours[2, column] = neighbour_local_id
        neighbours[3, column] = rotation
        neighbours[4, column] = orientation
        column += 1
    end

    return neighbours
end

"""
    _compute_all_neighbours(topology, object_dim, include_local_patch)

Return a `num_patches × num_local_objects` matrix collecting the neighbour matrices of every
local object of dimension `object_dim` of every patch.
"""
function _compute_all_neighbours(
    topology::AbstractTopology{manifold_dim}, object_dim::Int, include_local_patch::Bool
) where {manifold_dim}
    num_local_objects = get_local_size(topology, object_dim + 1)
    num_patches = size(topology, manifold_dim + 1)

    neighbours = Matrix{Matrix{Int}}(undef, num_patches, num_local_objects)
    for object_local_id in 1:num_local_objects, patch_id in 1:num_patches
        neighbours[patch_id, object_local_id] = _compute_object_neighbours(
            topology, patch_id, object_local_id, object_dim, include_local_patch
        )
    end

    return neighbours
end

"""
    compute_vertex_neighbours(
        topology::AbstractTopology,
        patch_id::Int,
        vertex_local_id::Int;
        include_local_patch::Bool=false,
    )

Return a `4 × N` matrix describing the patches that share the vertex with local index
`vertex_local_id` of patch `patch_id`. See [`compute_neighbours`](@ref) for the meaning of
the rows and of `include_local_patch`.

Rows 3 and 4 are always `0` and `1`, since a vertex has neither rotation nor orientation.
"""
function compute_vertex_neighbours(
    topology::AbstractTopology,
    patch_id::Int,
    vertex_local_id::Int;
    include_local_patch::Bool=false,
)
    return _compute_object_neighbours(
        topology, patch_id, vertex_local_id, 0, include_local_patch
    )
end

"""
    compute_vertex_neighbours(topology::AbstractTopology; include_local_patch::Bool=false)

Return a `num_patches × num_local_vertices` matrix whose entry `[i, j]` holds the neighbour
matrix of the `j`-th vertex of patch `i`, as returned by
[`compute_vertex_neighbours(::AbstractTopology, ::Int, ::Int)`](@ref).
"""
function compute_vertex_neighbours(
    topology::AbstractTopology; include_local_patch::Bool=false
)
    return _compute_all_neighbours(topology, 0, include_local_patch)
end

"""
    compute_edge_neighbours(
        topology::AbstractTopology,
        patch_id::Int,
        edge_local_id::Int;
        include_local_patch::Bool=false,
    )

Return a `4 × N` matrix describing the patches that share the edge with local index
`edge_local_id` of patch `patch_id`. See [`compute_neighbours`](@ref) for the meaning of the
rows and of `include_local_patch`.

Row 3 is always `0`, since an edge has only two vertices and hence no rotation; row 4 is `-1`
when the neighbour traverses the edge in the opposite direction, meaning that its edge
degrees of freedom must be reversed to match.

Requires a topology of manifold dimension at least 2; in 1D the edges are the patches
themselves.
"""
function compute_edge_neighbours(
    topology::AbstractTopology{manifold_dim},
    patch_id::Int,
    edge_local_id::Int;
    include_local_patch::Bool=false,
) where {manifold_dim}
    _check_object_dim(manifold_dim, 1)
    return _compute_object_neighbours(
        topology, patch_id, edge_local_id, 1, include_local_patch
    )
end

"""
    compute_edge_neighbours(topology::AbstractTopology; include_local_patch::Bool=false)

Return a `num_patches × num_local_edges` matrix whose entry `[i, j]` holds the neighbour
matrix of the `j`-th edge of patch `i`, as returned by
[`compute_edge_neighbours(::AbstractTopology, ::Int, ::Int)`](@ref).
"""
function compute_edge_neighbours(
    topology::AbstractTopology{manifold_dim}; include_local_patch::Bool=false
) where {manifold_dim}
    _check_object_dim(manifold_dim, 1)
    return _compute_all_neighbours(topology, 1, include_local_patch)
end

"""
    compute_face_neighbours(
        topology::AbstractTopology,
        patch_id::Int,
        face_local_id::Int;
        include_local_patch::Bool=false,
    )

Return a `4 × N` matrix describing the patches that share the face with local index
`face_local_id` of patch `patch_id`. See [`compute_neighbours`](@ref) for the meaning of the
rows and of `include_local_patch`.

Requires a topology of manifold dimension 3; in 2D the faces are the patches themselves.
"""
function compute_face_neighbours(
    topology::AbstractTopology{manifold_dim},
    patch_id::Int,
    face_local_id::Int;
    include_local_patch::Bool=false,
) where {manifold_dim}
    _check_object_dim(manifold_dim, 2)
    return _compute_object_neighbours(
        topology, patch_id, face_local_id, 2, include_local_patch
    )
end

"""
    compute_face_neighbours(topology::AbstractTopology; include_local_patch::Bool=false)

Return a `num_patches × num_local_faces` matrix whose entry `[i, j]` holds the neighbour
matrix of the `j`-th face of patch `i`, as returned by
[`compute_face_neighbours(::AbstractTopology, ::Int, ::Int)`](@ref).
"""
function compute_face_neighbours(
    topology::AbstractTopology{manifold_dim}; include_local_patch::Bool=false
) where {manifold_dim}
    _check_object_dim(manifold_dim, 2)
    return _compute_all_neighbours(topology, 2, include_local_patch)
end

"""
    _check_object_dim(manifold_dim::Int, object_dim::Int)

Throw an `ArgumentError` unless objects of dimension `object_dim` are shared between the
patches of a topology of manifold dimension `manifold_dim`, i.e. unless
`0 ≤ object_dim < manifold_dim ≤ 3`.
"""
function _check_object_dim(manifold_dim::Int, object_dim::Int)
    if !(0 ≤ object_dim < manifold_dim ≤ 3)
        throw(
            ArgumentError(
                LazyString(
                    "Mantis.Topology: objects of dimension ",
                    object_dim,
                    " are not shared between the patches of a topology of manifold",
                    " dimension ",
                    manifold_dim,
                    "; the dimension of a shared object must satisfy",
                    " 0 <= object_dim < manifold_dim <= 3.",
                ),
            ),
        )
    end
    return nothing
end

"""
    compute_neighbours(
        topology::AbstractTopology,
        patch_id::Int,
        local_object_id::Int,
        local_object_dim::Int;
        include_local_patch::Bool=false,
    )

Return a `4 × N` matrix describing the patches that share the geometric object of dimension
`local_object_dim` with local index `local_object_id` of patch `patch_id`.

Each column describes one such patch, with the rows holding

1. the neighbour patch id;
2. the local id of the shared object in the neighbour patch (always positive);
3. the **rotation**, i.e. the number of positions by which the neighbour's vertex sequence of
   the shared object is cyclically shifted with respect to the reference one. It is always
   `0` for vertices and edges, and is the number of 90° shifts for the quadrilateral faces of
   a hexahedral mesh;
4. the **orientation**, which is `1` when the neighbour traverses the shared object in the
   same cyclic direction as the reference and `-1` otherwise. It is always `1` for vertices.
   For edges, `-1` means that the edge degrees of freedom must be reversed to match; for
   faces, that they must be transposed.

Rotation and orientation are measured relative to a reference traversal of the shared object.
By default, this reference is the traversal prescribed by the current patch `patch_id`, which
is itself excluded from the result. If `include_local_patch` is `true`, the current patch is
included as one of the neighbours and the reference becomes the global definition of the
shared object, so that all patches (including the current one) are described in a common
frame.

# Notes
- Applicable to all supported topologies, with `0 ≤ local_object_dim < manifold_dim`.
- Assumes consistent local object numbering across patches.
"""
function compute_neighbours(
    topology::AbstractTopology{manifold_dim},
    patch_id::Int,
    local_object_id::Int,
    local_object_dim::Int;
    include_local_patch::Bool=false,
) where {manifold_dim}
    _check_object_dim(manifold_dim, local_object_dim)
    return _compute_object_neighbours(
        topology, patch_id, local_object_id, local_object_dim, include_local_patch
    )
end

"""
    compute_neighbours(
        topology::AbstractTopology, local_object_dim::Int; include_local_patch::Bool=false
    )

Return a `num_patches × num_local_objects` matrix whose entry `[i, j]` holds the neighbour
matrix of the `j`-th object of dimension `local_object_dim` of patch `i`, as returned by
[`compute_neighbours(::AbstractTopology, ::Int, ::Int, ::Int)`](@ref).
"""
function compute_neighbours(
    topology::AbstractTopology{manifold_dim},
    local_object_dim::Int;
    include_local_patch::Bool=false,
) where {manifold_dim}
    _check_object_dim(manifold_dim, local_object_dim)
    return _compute_all_neighbours(topology, local_object_dim, include_local_patch)
end

include("MeshTopology.jl")
include("SkeletonTopology.jl")

end

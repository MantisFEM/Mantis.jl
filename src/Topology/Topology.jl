"""
    module Topology

Provides connectivity information and mesh numbering supporting unstructured patch layouts.

The `Topology` module relies on `MeshCore` to create the numbering and the incidence
relations between geometric objects (vertices, edges, faces, volumes). The `Topology`
module uses this information to compute neighbour information and to provide easy acces to
this information.

`Mantis` has 2 layers of mesh information:
- The unstructured patch layer.
- The structured element layer.
So, a multi-patch mesh consists of an unstructured collection of patches. Each of these
patches consists of a structured collection of elements. The `Topology` module handles the
connectivity information of the unstructured patch layer, independently of what happens
within each patch.
"""
module Topology

import MeshCore

############################################################################################
#                                      Abstract Types                                      #
############################################################################################
"""
    AbstractPatch{manifold_dim, ir_dim, num_patch_vertices}

Patch types that can be used to construct an [`AbstractTopology`](@ref).

The type parameters represent the following data:
- `manifold_dim`: The dimension of the patch (e.g. 1D, 2D, etc.).
- `ir_dim`: The number of objects between which incidence_relations can be created. Will
    always be `manifold_dim` + 1 (e.g. 3 objects in 2D: vertices, edges, and faces).
- `num_patch_vertices`: The total number of vertices that make up a patch (e.g. 8 for a
    cube in 3D).
"""
abstract type AbstractPatch{manifold_dim, ir_dim, num_patch_vertices} end

"""
    AbstractTensorProductPatch{
        manifold_dim, ir_dim, num_patch_vertices
    } <: AbstractPatch{manifold_dim, ir_dim, num_patch_vertices}

Tensorial patch types that can be used to construct an [`AbstractTopology`](@ref). That is,
it represents line, square, and cuboidal patches.

Due to the tensorial nature of these patches, they also support a position based id next to
the interger ids. These two id types can always be converted. See [`id_to_position`](@ref)
and [`position_to_id`](@ref) for the details.

The type parameters are as defined in [`AbstractPatch`](@ref).
"""
abstract type AbstractTensorProductPatch{manifold_dim, ir_dim, num_patch_vertices} <:
              AbstractPatch{manifold_dim, ir_dim, num_patch_vertices} end

"""
    AbstractTopology{
        manifold_dim, ir_dim, num_patches, PT <: AbstractPatch{manifold_dim, ir_dim}
    }

Topology of an unstructured collection of patches (of equal shape) forming a mesh.

The geometric objects present depend on the manifold dimension:
- 1D: vertices, lines (patches);
- 2D: vertices, lines (facets), surfaces (patches);
- 3D: vertices, lines (edges), surfaces (facets), volumes (patches).

The connectivity relation or incidence relation between objects of dimensions `n` and `m`
can be accessed using indexing as `topology[n + 1, m + 1]` (with `topology` an instance of
`AbstractTopology`).
"""
abstract type AbstractTopology{
    manifold_dim, ir_dim, num_patches, PT <: AbstractPatch{manifold_dim, ir_dim}
} end

############################################################################################
#                                         Getters                                          #
############################################################################################
get_manifold_dim(::AbstractPatch{manifold_dim}) where {manifold_dim} = manifold_dim

function get_incidence_relations_dim(
    ::AbstractPatch{manifold_dim, ir_dim}
) where {manifold_dim, ir_dim}
    return ir_dim
end

function get_num_patch_vertices(
    ::AbstractPatch{manifold_dim, ir_dim, num_patch_vertices}
) where {manifold_dim, ir_dim, num_patch_vertices}
    return num_patch_vertices
end

get_manifold_dim(::AbstractTopology{manifold_dim}) where {manifold_dim} = manifold_dim

function get_incidence_relations_dim(
    ::AbstractTopology{manifold_dim, ir_dim}
) where {manifold_dim, ir_dim}
    return ir_dim
end

function get_num_patches(
    ::AbstractTopology{manifold_dim, ir_dim, num_patches}
) where {manifold_dim, ir_dim, num_patches}
    return num_patches
end

function get_patch_type(
    ::AbstractTopology{manifold_dim, ir_dim, num_patches, PT}
) where {manifold_dim, ir_dim, num_patches, PT}
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

Both are vectors of `(object_dim, global_object_id)` tuples: an object is a *boundary*
object when it belongs to exactly one patch, and an *interface* object when it is shared by
several. Objects are listed from the highest dimension down to vertices.
"""
function get_boundaries_and_interfaces(
    topology::AbstractTopology{manifold_dim}
) where {manifold_dim}
    boundaries = Tuple{Int, Int}[]
    interfaces = Tuple{Int, Int}[]
    for dim in (manifold_dim - 1):-1:0
        # Check which patches (of dimension manifold_dim) the current bounding objects
        # belong to.
        object_to_patch = topology[dim + 1, manifold_dim + 1]
        for object_id in eachindex(object_to_patch)
            # If it is not shared, this object has only 1 patch in its list.
            if length(object_to_patch[object_id]) == 1
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
    get_global_id(topology::AbstractTopology, patch_id, local_id, local_dim)
    get_global_id(
        topology::AbstractTopology,
        container_id::Int,
        container_dim::Int,
        local_id::Int,
        local_dim::Int,
    )

Return the global id of the given `local_id` of dimension `local_dim` of patch `patch_id`
(e.g, the 3rd (`local_id`=3) vertex (`local_dim`=0) of patch 4 (`patch_id`=4)). Instead of
the `patch_id`, one can also specicfy the objects of other dimensions (e.g. (e.g, the 3rd
(`local_id`=3) vertex (`local_dim`=0) of edge (`container_dim`=1) 5 (`container_id`=5)).

# Example
```jldoctest
julia> using Mantis

julia> topology = MeshTopology([(1, 2, 3, 4), (2, 5, 6, 3)], Topology.QUAD);

julia> get_global_id(topology, 5, 1, 2, 0) # 2nd vertex (dim 0) of edge 5 (dim 1).
3

julia> get_global_id(topology, 2, 4, 0) # 4th vertex (dim 0) of patch 2.
3
```
"""
function get_global_id(
    topology::AbstractTopology,
    container_id::Int,
    container_dim::Int,
    local_id::Int,
    local_dim::Int,
)
    if container_dim == local_dim
        return throw(ArgumentError("Container_dim and local_dim cannot be the same."))
    end

    container_local_objects = topology[container_dim + 1, local_dim + 1][container_id]
    global_id = container_local_objects[local_id]

    return global_id
end

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
    _check_object_dim(manifold_dim::Int, object_dim::Int)

Return `true` if `0 ≤ object_dim < manifold_dim ≤ 3`. Throw an `ArgumentError` if not.
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

    return true
end

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
    compute_neighbours(
        topology::AbstractTopology,
        patch_id::Int,
        object_local_id::Int,
        object_dim::Int,
        include_local_patch::Bool=false,
    )

Return a `4 × N` matrix describing the patches that share the geometric object of dimension
`object_dim` with local index `object_local_id` of patch `patch_id`.

Each column describes one such patch, with the rows holding

1. the neighbour patch id;
2. the local id of the shared object in the neighbour patch (always positive);
3. the **rotation**, i.e. the number of positions by which the neighbour's vertex sequence
    of the shared object is cyclically shifted with respect to the reference one. It is
    always `0` for vertices and edges, and is the number of 90° shifts for the
    quadrilateral faces of a hexahedral mesh;
4. the **orientation**, which is `1` when the neighbour traverses the shared object in the
   same cyclic direction as the reference and `-1` otherwise. It is always `1` for vertices.
   For edges, `-1` means that the edge degrees of freedom must be reversed to match; for
   faces, that they must be transposed.

Rotation and orientation are measured relative to a reference traversal of the shared
object. By default this reference is the traversal prescribed by the current patch
`patch_id`, which is itself excluded from the result. If `include_local_patch` is `true`,
the current patch is included as one of the neighbours and the reference becomes the global
definition of the shared object, so that all patches (including the current one) are
described in a common frame.
"""
function compute_neighbours(
    topology::AbstractTopology{manifold_dim},
    patch_id::Int,
    object_local_id::Int,
    object_dim::Int,
    include_local_patch::Bool=false,
) where {manifold_dim}
    _check_object_dim(manifold_dim, object_dim)

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
    compute_neighbours(
        topology::AbstractTopology, object_dim::Int, include_local_patch::Bool=false
    )

Return a `num_patches × num_local_objects` matrix collecting the neighbour matrices of every
local object of dimension `object_dim` of every patch. See
[`compute_neighbours(::AbstractTopology, ::Int, ::Int, ::Int)`](@ref).
"""
function compute_neighbours(
    topology::AbstractTopology{manifold_dim},
    object_dim::Int,
    include_local_patch::Bool=false,
) where {manifold_dim}
    _check_object_dim(manifold_dim, object_dim)

    num_local_objects = get_local_size(topology, object_dim + 1)
    num_patches = size(topology, manifold_dim + 1)

    neighbours = Matrix{Matrix{Int}}(undef, num_patches, num_local_objects)
    for object_local_id in 1:num_local_objects, patch_id in 1:num_patches
        neighbours[patch_id, object_local_id] = compute_neighbours(
            topology, patch_id, object_local_id, object_dim; include_local_patch
        )
    end

    return neighbours
end

include("Patches.jl")
include("MeshTopology.jl")
include("SkeletonTopology.jl")

end

"""
    SkeletonTopology{manifold_dim, incidence_relations_dim, num_patches}

Represents the topological structure of the skeleton of a mesh.

A skeleton of a (parent) mesh of `manifold_dim` is a mesh containing the collection of all
geometric objects of dimension `<= (manifold_dim - 1)` of the (parent) mesh and their
incidence relations. For this reason, a `SkeletonTopology` can be seen as `MeshTopology`.

Additionally, a `SkeletonTopology` contains information relating its geometric objects to
the ones in the parent mesh.


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
struct SkeletonTopology{
    manifold_dim, incidence_relations_dim, num_patches, parent_type <: MeshTopology
} <: AbstractTopology{manifold_dim, incidence_relations_dim, num_patches}
    parent_topology::parent_type

    function SkeletonTopology(
        parent_topology::MT
    ) where {
        manifold_dim_parent,
        incidence_relations_dim_parent,
        num_patches_parent,
        MT <: MeshTopology{
            manifold_dim_parent, incidence_relations_dim_parent, num_patches_parent
        },
    }
        # Extract geometric information for skeleton
        manifold_dim = manifold_dim_parent - 1
        incidence_relations_dim = manifold_dim + 1
        n_patches = size(parent_topology, manifold_dim + 1)

        return new{manifold_dim, incidence_relations_dim, n_patches, MT}(parent_topology)
    end
end

# Property getters.
function get_parent_topology(topology::SkeletonTopology)
    return topology.parent_topology
end

# Indexing.
Base.IndexStyle(::Type{<:SkeletonTopology}) = IndexLinear()

function Base.getindex(
    topology::SkeletonTopology{manifold_dim}, i::Int, k::Int
) where {manifold_dim}
    @boundscheck begin
        if !(1 ≤ i ≤ (manifold_dim + 1) && 1 ≤ k ≤ (manifold_dim + 1))
            throw(BoundsError(topology, (i, k)))
        end
    end

    @inbounds return get_parent_topology(topology)[i, k]
end

# Sizes.
function Base.size(topology::SkeletonTopology{manifold_dim}) where {manifold_dim}
    # Note that indexing up to and including manifold_dim + 1 uses the manifold dim of the
    # skeleton, which is always one less than the parent. There are always manifold_dim + 1
    # types of geometry objects in a topology, so using the skeleton_manifold_dim + 1 as
    # index for the parent_topology will give exactly the number of geometric objects of
    # the skeleton.
    return size(get_parent_topology(topology))[1:(manifold_dim + 1)]
end

# geometric_dim_id is the index associated to the geometric dimension. Geometric dimension n
# has index (n + 1), this is done to keep consistency with julia indices that start at 1 and
# not at 0 (vertices have geometric dimension 0, and thus geometric_dim_id 1).
function Base.size(
    topology::SkeletonTopology{manifold_dim}, geometric_dim_id::Int
) where {manifold_dim}
    @boundscheck begin
        if !(1 ≤ geometric_dim_id ≤ (manifold_dim + 1))
            throw(BoundsError(topology, geometric_dim_id))
        end
    end
    @inbounds return size(get_parent_topology(topology), geometric_dim_id)
end

"""
    get_local_size(
        topology::SkeletonTopology{manifold_dim}, geometric_dim_id::Int
    ) where {manifold_dim}
    get_local_size(topology::SkeletonTopology{manifold_dim}) where {manifold_dim}

Return the number of local geometric objects per patch for a given geometric dimension.

All patches are assumed to have the same number of local geometric objects.

# Arguments
- `topology::SkeletonTopology`: The skeleton topology.
- `geometric_dim_id::Int`: 1-based index of the geometric dimension.
    Dimension `n` corresponds to index `n + 1`
    (e.g. vertices have geometric dimension 0 and thus `geometric_dim_id = 1`).

# Returns
- `Int`: Number of local geometric objects of the given dimension per patch.
"""
function get_local_size(
    topology::SkeletonTopology{manifold_dim}, geometric_dim_id::Int
) where {manifold_dim}
    # Get the (local, i.e., per patch, assumed all patches identical) number of geometric objects for a given dimension
    @boundscheck begin
        if !(1 ≤ geometric_dim_id ≤ (manifold_dim + 1))
            throw(BoundsError(topology, geometric_dim_id))
        end
    end
    @inbounds return get_local_size(get_parent_topology(topology), geometric_dim_id)
end
function get_local_size(topology::SkeletonTopology{manifold_dim}) where {manifold_dim}
    return get_local_size(get_parent_topology(topology))[1:(manifold_dim + 1)]
end

# Parent information.
function get_patch_parents(
    topology::SkeletonTopology{manifold_dim}, patch_id
) where {manifold_dim}
    patch_dim = manifold_dim  # geometric dimension of the current patch
    parent_dim = patch_dim + 1  # geometric dimension of the parent patch containing current patch as
    # part of its boundary
    parent_topology = get_parent_topology(topology)

    # Get the list of parent geometric objects containing the current patch with id patch_id
    # for which the curretn patch is part of their boundary
    parent_patches_id = parent_topology[patch_dim + 1, parent_dim + 1][patch_id]

    # Get the local id on each of these parent patches, together with orientation relative
    # to the definition of the current patch
    parent_patch_id = parent_patches_id[1]  # pick one (the first) to identify the patch (we need one to start)
    local_patch_id = abs(
        get_local_id(parent_topology, parent_patch_id, patch_id, patch_dim)
    )
    patch_parents = compute_face_neighbours(
        parent_topology, parent_patch_id, local_patch_id; include_local_patch=true
    )

    return patch_parents
end

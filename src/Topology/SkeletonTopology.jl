############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    SkeletonTopology{
        manifold_dim, incidence_relations_dim, num_patches, PT, parent_type <: MeshTopology
    } <: AbstractTopology{manifold_dim, incidence_relations_dim, num_patches, PT}

Topological structure of the skeleton of a mesh.

A skeleton of a (parent) mesh of `manifold_dim` is a mesh containing the collection of all
geometric objects of dimension `<= (manifold_dim - 1)` of the (parent) mesh and their
incidence relations. It also relates its geometric objects to the ones in the parent mesh.

# Constructors
- `SkeletonTopology(
        parent_topology::MT
    ) where {
        manifold_dim_parent,
        incidence_relations_dim_parent,
        num_patches_parent,
        MT <: MeshTopology{
            manifold_dim_parent, incidence_relations_dim_parent, num_patches_parent
        },
    }`: general constructor.

# Fields
- `topological_patch::PT`: The [`AbstractPatch`](@ref) object out of which the mesh is made.
    This is directly obtained from the topological patch of the `parent_topology`.
- `parent_topology::parent_type`: The parent [`MeshTopology`](@ref).
"""
struct SkeletonTopology{
    manifold_dim, incidence_relations_dim, num_patches, PT, MT <: MeshTopology
} <: AbstractTopology{manifold_dim, incidence_relations_dim, num_patches, PT}
    topological_patch::PT
    parent_topology::MT

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
        if manifold_dim_parent < 2
            throw(
                ArgumentError(
                    LazyString(
                        "The skeleton of a topology of manifold dimension ",
                        manifold_dim_parent,
                        " is a set of points, which is not a topology; a skeleton requires",
                        " a parent of manifold dimension at least 2.",
                    ),
                ),
            )
        end

        topological_patch = get_skeleton_patch(get_topological_patch(parent_topology))
        manifold_dim = manifold_dim_parent - 1
        incidence_relations_dim = manifold_dim + 1
        num_patches = size(parent_topology, manifold_dim + 1)

        return new{
            manifold_dim,
            incidence_relations_dim,
            num_patches,
            typeof(topological_patch),
            MT,
        }(
            topological_patch, parent_topology
        )
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################
function get_parent_topology(topology::SkeletonTopology)
    return topology.parent_topology
end

############################################################################################
#                                         Indexing                                         #
############################################################################################
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

############################################################################################
#                                          Sizes                                           #
############################################################################################
function Base.size(topology::SkeletonTopology{manifold_dim}) where {manifold_dim}
    # Note that indexing up to and including manifold_dim + 1 uses the manifold dim of the
    # skeleton, which is always one less than the parent. There are always manifold_dim + 1
    # types of geometry objects in a topology.
    return size(get_parent_topology(topology))[1:(manifold_dim + 1)]
end

function Base.size(
    topology::SkeletonTopology{manifold_dim}, geometric_dim_id::Int
) where {manifold_dim}
    # This boundscheck is stricter than the one for the parent topology, since the skeleton
    # topolgy is lower-dimensional.
    @boundscheck begin
        if !(1 ≤ geometric_dim_id ≤ (manifold_dim + 1))
            throw(BoundsError(topology, geometric_dim_id))
        end
    end
    @inbounds return size(get_parent_topology(topology), geometric_dim_id)
end

############################################################################################
#                                    Parent information                                    #
############################################################################################
"""
    get_patch_parents(topology::SkeletonTopology, patch_id::Int)

Return the `4 × N` matrix describing the patches of the parent topology that the skeleton
patch `patch_id` bounds, in the format documented for [`compute_neighbours`](@ref).

Rotation and orientation are relative to the definition of the skeleton patch in the parent
topology, i.e. this is [`compute_neighbours`](@ref) with `include_local_patch = true`.
"""
function get_patch_parents(
    topology::SkeletonTopology{manifold_dim}, patch_id::Int
) where {manifold_dim}
    # Geometric dimension of the current (skeleton) patch, and of the parent patches that
    # it is part of the boundary of.
    patch_dim = manifold_dim
    parent_dim = patch_dim + 1
    parent_topology = get_parent_topology(topology)

    # Any parent patch containing the current one identifies it; pick the first.
    parent_patch_id = parent_topology[patch_dim + 1, parent_dim + 1][patch_id][1]
    local_patch_id = abs(
        get_local_id(parent_topology, parent_patch_id, patch_id, patch_dim)
    )

    return compute_neighbours(
        parent_topology,
        parent_patch_id,
        local_patch_id,
        patch_dim;
        include_local_patch=true,
    )
end

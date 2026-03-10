"""
    CartesianGeometry{manifold_dim, image_dim, num_patches, B, CI} <: AbstractGeometry{
        manifold_dim, image_dim, num_patches
    }

A structure representing a Cartesian grid geometry in `manifold_dim` dimensions. Can have
multiple patches, even though each patch is still a Cartesian grid. Note that the patches
are not required to have a matching grid.

# Fields
- `breakpoints::B`: A tuple of vectors defining the grid points in each dimension.
- `cart_num_elements::CI`: A (tuple of) `CartesianIndices` representing the indices of
    elements in the grid for each patch.

# Constructors
- `CartesianGeometry(
        breakpoints::B
    ) where {
        manifold_dim,
        num_patches,
        NT <: Number,
        B <: NTuple{num_patches, NTuple{manifold_dim, AbstractVector{NT}}},
    }`: General constructor.
- `CartesianGeometry(
        breakpoints::NTuple{manifold_dim, AbstractVector{NT}}
    ) where {manifold_dim, NT <: Number}`: Single-patch convenience constructor.
"""
struct SkeletonGeometry{manifold_dim, image_dim, num_patches, T, PG} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    topology::T
    parent_geometry::PG

    function SkeletonGeometry(
        parent_geometry::PG
    ) where {
        parent_manifold_dim,
        image_dim,
        parent_num_patches,
        PG <: AbstractGeometry{parent_manifold_dim, image_dim, parent_num_patches}
    }
        manifold_dim = parent_manifold_dim - 1
        topology = Topology.SkeletonTopology(Geometry.get_topology(parent_geometry)) 
        num_patches = Topology.get_num_patches(topology)  # TODO: this is type unstable, either we
                                                          # use vectors or we must add 
                                                          # (n_faces, n_patches)
                                                          # to the type parameters instead of only n_patches

        return new{
            manifold_dim, 
            image_dim, 
            num_patches, 
            typeof(topology), 
            PG
        }(
            topology, parent_geometry
        )
    end
end

function get_topology(geometry::SG) where {
    SG <: SkeletonGeometry
}
    return geometry.topology
end

function get_parent_geometry(geometry::SG) where {
    SG <: SkeletonGeometry
}
    return geometry.parent_geometry
end

# function get_num_elements(geometry::SkeletonGeometry)
#     return geometry.num_elements
# end

# function get_num_elements(geometry::SkeletonGeometry, patch_id::Int)
#     return get_num_elements_per_patch(geometry)[patch_id]
# end

function get_num_elements_per_patch(
    geometry::SkeletonGeometry{patch_dim, image_dim, num_patches}
) where {
    patch_dim,
    image_dim,
    num_patches
}
    
    return ntuple(num_patches) do patch_id
        # Get the current patch parents, i.e., the patches in the parent geometry that contain
        # the current patch. Since all parents must have the same elements on the current patch 
        # of the skeleton, we can just pick any parent patch. We pick the first one.
        parent_patch = Topology.get_patch_parents(get_topology(geometry), patch_id)[:, 1]

        # Now we get the number of elements on the local geometric object of dimension
        # image_dim that coincides with our current patch, since the current patch must have the
        # same element distribution.
        parent_patch_id = parent_patch[1]
        local_patch_id = parent_patch[2]
        display(get_elements(get_parent_geometry(geometry), parent_patch_id, local_patch_id, patch_dim))
        length(get_elements(get_parent_geometry(geometry), parent_patch_id, local_patch_id, patch_dim))
    end 
end

# function get_element_vertices(
#     geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int
# ) where {manifold_dim, image_dim, num_patches}
#     throw(MethodError(get_element_vertices, (geometry, element_id)))
# end
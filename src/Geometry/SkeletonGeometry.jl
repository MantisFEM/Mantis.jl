"""
    SkeletonGeometry{manifold_dim, image_dim, num_patches, T, PG} <:
           AbstractGeometry{manifold_dim, image_dim, num_patches}

Contains the geometry of the skeleton of its `parent_geometry`, as well as the `topology`
of the skeleton.

# Fields
- `topology::T`: Skeleton topology, see [`SkeletonTopology`](@ref)
- `parent_geometry::PG`: The provided geometry of which to obtain the skeleton.

# Constructors
- `SkeletonGeometry(
        parent_geometry::PG
    ) where {
        parent_manifold_dim,
        image_dim,
        parent_num_patches,
        PG <: AbstractGeometry{parent_manifold_dim, image_dim, parent_num_patches},
    }`: General constructor.
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
        PG <: AbstractGeometry{parent_manifold_dim, image_dim, parent_num_patches},
    }
        manifold_dim = parent_manifold_dim - 1
        topology = Topology.SkeletonTopology(Geometry.get_topology(parent_geometry))
        num_patches = Topology.get_num_patches(topology)

        return new{manifold_dim, image_dim, num_patches, typeof(topology), PG}(
            topology, parent_geometry
        )
    end
end

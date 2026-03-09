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

"""
    UnstructuredGeometry{manifold_dim, image_dim, num_patches, T, GP} <: AbstractGeometry{
        manifold_dim, image_dim, num_patches
    }

A geometry consisting of multiple patches, each with its own geometry.

!!! warning "Avoid heterogeneous inputs."
    While the constructors allow different types of geometries, it is strongly recommended
    to use only a few different types. Failing to do so can cause type instabilities and
    therefore a significant performance penalty.

# Fields
- `topology::T`: A [`Topology.MeshTopology`](@ref) specifying how the patches connect.
- `geometry_per_patch::NTuple{GP, num_patches}`: The geometries for each patch.
- `num_elements::Int`: The total number of elements in the geometry.
- `num_elements_per_patch::NTuple{num_patches, Int}`: The number of elments per patch.
"""
struct UnstructuredGeometry{manifold_dim, image_dim, num_patches, T, GT} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    topology::T
    geometry_per_patch::GT
    num_elements::Int
    num_elements_per_patch::NTuple{num_patches, Int}

    function UnstructuredGeometry(
        geometry_per_patch::GT, topology::T
    ) where {
        manifold_dim,
        image_dim,
        num_patches,
        incidence_relations_dim,
        GT <: NTuple{num_patches, AbstractGeometry{manifold_dim, image_dim, 1}},
        T <: Topology.MeshTopology{manifold_dim, incidence_relations_dim, num_patches},
    }
        num_elements_per_patch = ntuple(num_patches) do i
            get_num_elements(geometry_per_patch[i])
        end
        num_elements = sum(num_elements_per_patch)

        return new{manifold_dim, image_dim, num_patches, T, GT}(
            topology, geometry_per_patch, num_elements, num_elements_per_patch
        )
    end

    # A single patch has no connectivity of its own, so its topology is inherited from the
    # geometry it wraps.
    function UnstructuredGeometry(
        geometry_per_patch::GT
    ) where {
        manifold_dim,
        image_dim,
        GT <: NTuple{1, AbstractGeometry{manifold_dim, image_dim, 1}},
    }
        return UnstructuredGeometry(geometry_per_patch, get_topology(geometry_per_patch[1]))
    end

    # As for the other multi-patch geometries, the connectivity between patches does not
    # follow from the patch geometries themselves.
    function UnstructuredGeometry(
        ::NTuple{num_patches, AbstractGeometry{manifold_dim, image_dim, 1}}
    ) where {num_patches, manifold_dim, image_dim}
        throw(
            ArgumentError(
                LazyString(
                    "An unstructured geometry with ",
                    num_patches,
                    " patches needs an explicit topology, because the connectivity between",
                    " patches does not follow from the patch geometries. Construct one with",
                    " Topology.MeshTopology and pass it as",
                    " UnstructuredGeometry(geometry_per_patch, topology).",
                ),
            ),
        )
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_geometry_per_patch(geometry::UnstructuredGeometry) = geometry.geometry_per_patch

function get_geometry(geometry::UnstructuredGeometry, patch_id::Int)
    return geometry.geometry_per_patch[patch_id]
end

# Getters for numbers, sizes, shapes, lengths, etc.
function get_element_lengths(geometry::UnstructuredGeometry, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return get_element_lengths(get_geometry(geometry, patch_id), local_element_id)
end

function get_element_measure(geometry::UnstructuredGeometry, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return get_element_measure(get_geometry(geometry, patch_id), local_element_id)
end

function get_element_vertices(geometry::UnstructuredGeometry, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return get_element_vertices(get_geometry(geometry, patch_id), local_element_id)
end

############################################################################################
#                                        Evaluation                                        #
############################################################################################

function evaluate(
    geometry::UnstructuredGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return evaluate(get_geometry(geometry, patch_id), local_element_id, xi)
end

function jacobian(
    geometry::UnstructuredGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return jacobian(get_geometry(geometry, patch_id), local_element_id, xi)
end

function hessian(
    geometry::UnstructuredGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return hessian(get_geometry(geometry, patch_id), local_element_id, xi)
end

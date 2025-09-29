"""
    UnstructuredGeometry{manifold_dim, image_dim, num_patches, GP} <: AbstractGeometry{
        manifold_dim, image_dim, num_patches
    }

A geometry consisting of multiple patches, each with its own geometry.

# Fields
- `geometry_per_patch::NTuple{GP, num_patches}`: The geometries for each patch.
"""
struct UnstructuredGeometry{manifold_dim, image_dim, num_patches, GT} <: AbstractGeometry{
    manifold_dim, image_dim, num_patches
}
    geometry_per_patch::GT
    num_elements::Int
    num_elements_per_patch::NTuple{num_patches, Int}

    function UnstructuredGeometry(geometry_per_patch::GT) where {
        manifold_dim, num_patches, GT <: NTuple{num_patches, AbstractGeometry{manifold_dim, 1}}
    }
        num_elements_per_patch = ntuple(num_patches) do i
            get_num_elements(geometry_per_patch[i])
        end
        num_elements = sum(num_elements_per_patch)
        image_dim = get_image_dim(geometry_per_patch[1])
        for geo in geometry_per_patch
            if get_image_dim(geo) != image_dim
                throw(ArgumentError(
                    LazyString("All patches must have the same image dimension.")
                ))
            end
        end
        return new{manifold_dim, image_dim, num_patches, GT}(
            geometry_per_patch, num_elements, num_elements_per_patch, image_dim
        )
    end
end



# Getters and setters.
"""
    get_geometry_on_patch(MPGeo::UnstructuredGeometry, patch_id::Int)

Get the geometry on a specific patch.

# Arguments
- `geometry::UnstructuredGeometry`: The multi-patch geometry.
- `patch_id::Int`: The patch ID.

# Returns
- ` <: AbstractGeometry{manifold_dim, image_dim, 1}`: The geometry on the specified patch.
"""
function get_geometry_on_patch(geometry::UnstructuredGeometry, patch_id::Int)
    return geometry.geometry_per_patch[patch_id]
end



# Evaluation (and related) methods.
# All these methods first determine the patch on which the element resides, then call the
# corresponding method on that patch's geometry.
function evaluate(
    geometry::UnstructuredGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return evaluate(geometry.geometry_per_patch[patch_id], local_element_id, xi)
end

function jacobian(
    geometry::UnstructuredGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return jacobian(geometry.geometry_per_patch[patch_id], local_element_id, xi)
end

function hessian(
    geometry::UnstructuredGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return hessian(geometry.geometry_per_patch[patch_id], local_element_id, xi)
end


function get_element_lengths(geometry::UnstructuredGeometry, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return get_element_lengths(geometry.geometry_per_patch[patch_id], local_element_id)
end

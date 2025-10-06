"""
    module Geometry

Contains all geometry structure definitions and related methods.
"""
module Geometry

using LinearAlgebra
using StaticArrays

import ..FunctionSpaces
import ..Points

abstract type AbstractGeometry{manifold_dim, image_dim, num_patches} end

"""
    get_manifold_dim(::AbstractGeometry{manifold_dim, image_dim, num_patches})

Returns the dimensions of the domain manifold of a given geometry.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `::Int`: The domain manifold dimension.

# Notes
This method is used as a fallback if there isn't a more specific method to be used. The
latter should only be implemented explicitly if necessary.
"""
get_manifold_dim(
    ::AbstractGeometry{manifold_dim, image_dim, num_patches}
) where {manifold_dim, image_dim, num_patches} = manifold_dim

"""
    get_image_dim(::AbstractGeometry{manifold_dim, image_dim, num_patches})

Returns the dimensions of the image manifold of a given geometry.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `::Int`: The image manifold dimension.
"""
get_image_dim(
    ::AbstractGeometry{manifold_dim, image_dim, num_patches}
) where {manifold_dim, image_dim, num_patches} = image_dim

"""
    get_num_patches(::AbstractGeometry{manifold_dim, image_dim, num_patches})

Returns the number of patches in a given geometry.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `::Int`: The number of patches in the geometry.
"""
get_num_patches(
    ::AbstractGeometry{manifold_dim, image_dim, num_patches}
) where {manifold_dim, image_dim, num_patches} = num_patches

"""
    get_patch_id(geometry::AbstractGeometry, element_id::Int)

Get the ID of the patch to which the specified global element belongs.

# Arguments
- `geometry::AbstractGeometry`: The multi-patch geometry.
- `element_id::Int`: The global element ID.

# Returns
- `::Int`: ID of the patch to which the element belongs.
"""
function get_patch_id(geometry::AbstractGeometry, element_id::Int)
    cumulative_elements = cumsum(get_num_elements_per_patch(geometry))
    for ci in eachindex(cumulative_elements)
        if element_id <= cumulative_elements[ci]
            return ci
        end
    end
    throw(
        ArgumentError(
            LazyString(
                "Element ID ",
                element_id,
                " exceeds the total number of elements in the geometry.",
            )
        ),
    )
end

function get_patch_id(
    geometry::AbstractGeometry{manifold_dim, 1}, element_id::Int
) where {manifold_dim}
    return 1
end

"""
    get_patch_and_local_element_id(geometry::AbstractGeometry, element_id::Int)

Get the constituent patch ID and local element ID for the specified global element ID.

# Arguments
- `geometry::AbstractGeometry`: The multi-patch geometry.
- `element_id::Int`: The global element ID.

# Returns
- `patch_id::Int`: The patch ID
- `local_element_id::Int`: The local element ID.
"""
function get_patch_and_local_element_id(geometry::AbstractGeometry, element_id::Int)
    patch_id = get_patch_id(geometry, element_id)

    local_element_id = element_id
    for (i, num_elements_patch_i) in pairs(get_num_elements_per_patch(geometry))
        if i < patch_id
            local_element_id -= num_elements_patch_i
        else
            break
        end
    end

    return patch_id, local_element_id
end

function get_patch_and_local_element_id(
    geometry::AbstractGeometry{manifold_dim, 1}, element_id::Int
) where {manifold_dim}
    return 1, element_id
end

"""
    get_global_element_id(geometry::AbstractGeometry, patch_id::Int, local_element_id::Int)

Get the global element ID for the specified constituent patch ID and local element ID.

# Arguments
- `geometry::AbstractGeometry`: A (multi-)patch geometry.
- `patch_id::Int`: The constituent patch ID.
- `local_element_id::Int`: The local element ID.

# Returns
- `::Int`: The global element ID.
"""
function get_global_element_id(geometry::AbstractGeometry, patch_id::Int, local_element_id::Int)
    return sum(get_num_elements_per_patch(geometry)[begin:(patch_id - 1)]; init=0) +
           local_element_id
end

"""
    get_num_elements(geometry::AbstractGeometry)

Returns the number of elements in `geometry`.

# Arguments
- `geometry::AbstractGeometry`: The geometry being used.

# Returns
- `::Int`: The number of elements in the geometry.

# Notes
This method is used as a fallback if there isn't a more specific method to be used. The
latter should only be implemented explicitly if necessary.
"""
function get_num_elements(geometry::AbstractGeometry)
    return geometry.num_elements
end

"""
    get_num_elements_per_patch(geometry::AbstractGeometry)

Returns the number of elements in each patch of `geometry`.

# Arguments
- `geometry::AbstractGeometry`: The geometry being used.

# Returns
- `::Int`: The number of elements in the geometry.

# Notes
This method is used as a fallback if there isn't a more specific method to be used. The
latter should only be implemented explicitly if necessary.
"""
function get_num_elements_per_patch(geometry::AbstractGeometry)
    return geometry.num_elements_per_patch
end

"""
    get_geometry(geometry::AbstractGeometry, patch_id::Int)

Get (or create) the physical geometry on a specific patch.

# Arguments
- `geometry::AbstractGeometry`: The multi-patch geometry.
- `patch_id::Int`: The patch ID.

# Returns
- ` <: AbstractGeometry{manifold_dim, image_dim, 1}`: The geometry on the specified patch.
"""
function get_geometry(geometry::AbstractGeometry, patch_id::Int)
    throw(
        ArgumentError(
            LazyString("Method not defined for geometry of type ",typeof(geometry),".")
        )
    )
end

function get_parametric_geometry(geometry::AbstractGeometry)
    throw(
        ArgumentError(
            LazyString("Method not defined for geometry of type ",typeof(geometry),".")
        )
    )
end

"""
    get_parametric_geometry(geometry::AbstractGeometry, patch_id::Int)

Finds the parametric geometry of the patch given by `patch_id` in `geometry`. If no
`patch_id` is given, the parametric geometry of the whole geometry is returned.

# Arguments
- 'geometry::AbstractGeometry': The (physical) geometry being used.
- 'patch_id::Int': Index of the patch of which the parametric geometry is to be returned.

# Returns
- '<:CartesianGeometry': The parametric geometry associated with the specified patch.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function get_parametric_geometry(geometry::AbstractGeometry, patch_id::Int)
    throw(
        ArgumentError(
            LazyString("Method not defined for geometry of type ",typeof(geometry),".")
        )
    )
end

"""
    get_element_measure(geometry::AbstractGeometry, element_id::Int)

Computes the measure of the element given by `element_id` in `geometry`.

# Arguments
- 'geometry::AbstractGeometry': The geometry being used.
- 'element_id::Int': Index of the element being considered.

# Returns
- '::Float64': The measure of the element.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function get_element_measure(geometry::AbstractGeometry, element_id::Int)
    throw(
        ArgumentError(
            LazyString("Method not defined for geometry of type ",typeof(geometry),".")
        )
    )
end

"""
    get_element_lengths(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int,
    ) where {manifold_dim, image_dim, num_patches}

Computes the length, in each manifold dimension, of the element given by `element_id` in
`geometry`.

# Arguments
- 'geometry::AbstractGeometry': The geometry being used.
- 'element_id::Int': Index of the element being considered.

# Returns
- '::NTuple{manifold_dim, Float64}': The element's lengths.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function get_element_lengths(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int
) where {manifold_dim, image_dim, num_patches}
    throw(
        ArgumentError(
            LazyString("Method not defined for geometry of type ",typeof(geometry),".")
        )
    )
end

"""
    get_element_vertices(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int,
    ) where {manifold_dim, image_dim, num_patches}

Computes the vertices, in each manifold dimension, of the element given by `element_id` in
`geometry`.

# Arguments
- 'geometry::AbstractGeometry': The geometry being used.
- 'element_id::Int': Index of the element being considered.

# Returns
- '::NTuple{manifold_dim, Float64}': The element's vertices.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function get_element_vertices(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int
) where {manifold_dim, image_dim, num_patches}
    throw(
        ArgumentError(
            LazyString("Method not defined for geometry of type ",typeof(geometry),".")
        )
    )
end

"""
    evaluate(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Computes the evaluation of the physical points, mapped from the canonical points `xi`, of
the element identified by `element_id` of a given `geometry`.

# Arguments
- `geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being evaluated.
- `element_id::Int`: The identifier of the element where the evaluation takes place.
- `xi::NTuple{manifold_dim,Vector{Float64}}`: The points in canonical space used for
    evaluation.

# Returns
- `eval::Matrix{Float64}`: The geometry evaluatation based on `element_id` and `xi`. The
    dimensions of `eval` are `(num_eval_points, image_dim)`, where `num_eval_points` is the
    product of the number of evaluation points in `xi` in each dimension, and `image_dim` is
    the image manifold dimension of `geometry`.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function evaluate(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    throw(
        ArgumentError(
            LazyString("Method not defined for geometry of type ",typeof(geometry),".")
        )
    )
end

"""
    jacobian(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Computes the jacobian at the physical points, mapped from the canonical points `xi`, of the
element identified by `element_id` of a given `geometry`.

# Arguments
- `geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.
- `element_id::Int`: The identifier of the element where the evaluation takes place.
- `xi::NTuple{manifold_dim,Vector{Float64}}`: The points in canonical space used for
    evaluation.

# Returns
- `J::Matrix{Float64}`: The jacobian evaluatation based on `element_id` and `xi`. The
    dimensions of `J` are `(num_eval_points, image_dim, manifold_dim)`, where
    `num_eval_points` is the product of the number of evaluation points in `xi` in each
    dimension, and `image_dim` is the image manifold dimension of `geometry`.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function jacobian(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    throw(
        ArgumentError(
            LazyString("Method not defined for geometry of type ",typeof(geometry),".")
        )
    )
end

# core functionality
include("./CartesianGeometry.jl")
include("./FEGeometry.jl")
include("./MappedGeometry.jl")
include("./TensorProductGeometry.jl")
include("./HierarchicalGeometry.jl")
include("./UnstructuredGeometry.jl")
include("./Metric.jl")

# helper functions for convenience
include("./GeometryHelpers.jl")

end

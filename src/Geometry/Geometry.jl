"""
    module Geometry

Contains all geometry structure definitions and related methods.
"""
module Geometry

import LinearAlgebra

using ..FunctionSpaces

abstract type AbstractGeometry{manifold_dim} end
abstract type AbstractAnalyticalGeometry{manifold_dim} <: AbstractGeometry{manifold_dim} end
abstract type AbstractFEMGeometry{manifold_dim} <: AbstractGeometry{manifold_dim} end

"""
    get_manifold_dim(::AbstractGeometry{manifold_dim})

Returns the dimensions of the domain manifold of a given geometry.

# Arguments
- `::AbstractGeometry{manifold_dim}`: The geometry being used.

# Returns
- `::Int`: The domain manifold dimension.

# Notes
This method is used as a fallback if there isn't a more specific method to be used. The
latter should only be implemented explicitly if necessary.
"""
get_manifold_dim(::AbstractGeometry{manifold_dim}) where {manifold_dim} = manifold_dim

"""
    get_image_dim(::AbstractGeometry{manifold_dim})

Returns the dimensions of the image manifold of a given geometry.

# Arguments
- `::AbstractGeometry{manifold_dim}`: The geometry being used.

# Returns
- `::Int`: The image manifold dimension.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function get_image_dim(geometry::AbstractGeometry)
    throw(ArgumentError("Method not defined for geometry of type $(typeof(geometry))."))
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
    throw(ArgumentError("Method not defined for geometry of type $(typeof(geometry))."))
end

"""
    get_element_lengths(
        geometry::AbstractGeometry{manifold_dim}, element_id::Int,
    ) where {manifold_dim}

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
    geometry::AbstractGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    throw(ArgumentError("Method not defined for geometry of type $(typeof(geometry))."))
end

"""
    evaluate(
        geometry::AbstractGeometry{manifold_dim},
        element_id::Int,
        xi::NTuple{manifold_dim, Vector{Float64}},
    ) where {manifold_dim}

Computes the evaluation of the physical points, mapped from the canonical points `xi`, of
the element identified by `element_id` of a given `geometry`.

# Arguments
- `geometry::AbstractGeometry{manifold_dim}`: The geometry being evaluated.
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
    geometry::AbstractGeometry{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    throw(ArgumentError("Method not defined for geometry of type $(typeof(geometry))."))
end

"""
    jacobian(
        geometry::AbstractGeometry{manifold_dim},
        element_id::Int,
        xi::NTuple{manifold_dim, Vector{Float64}},
    ) where {manifold_dim}

Computes the jacobian at the physical points, mapped from the canonical points `xi`, of the
element identified by `element_id` of a given `geometry`.

# Arguments
- `geometry::AbstractGeometry{manifold_dim}`: The geometry being used.
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
    geometry::AbstractGeometry{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    throw(ArgumentError("Method not defined for geometry of type $(typeof(geometry))."))
end

# core functionality
include("./CartesianGeometry.jl")
include("./FEMGeometry.jl")
include("./MappedGeometry.jl")
include("./TensorProductGeometry.jl")
include("./NestedGeometry.jl")
include("./Metric.jl")

# helper functions for convenience
include("./GeometryHelpers.jl")

end

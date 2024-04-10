"""
This (sub-)module provides a collection of scalar function spaces.

The exported names are:
"""
module FiniteElementSpaces

import .. Mesh
import .. ElementSpaces
import SparseArrays

"""
    AbstractFiniteElementSpace

Supertype for all scalar function spaces.
"""
abstract type AbstractFiniteElementSpace{n} end

# Getters for the function spaces
get_n(f::AbstractFiniteElementSpace{n}) where {n} = n

"""
    struct ExtractionOperator

Structure to store extraction operators and coefficients.
"""
struct ExtractionOperator
    extraction_coefficients::Vector{Array{Float64}}
    basis_indices::Vector{Vector{Int}}
    num_elements::Int
    space_dim::Int
end

# Getters for extraction operators
function get_extraction(extraction_op::ExtractionOperator, element_id::Int)
    return @views extraction_op.extraction_coefficients[element_id], extraction_op.basis_indices[element_id]
end

function get_num_elements(extraction_op::ExtractionOperator)
    return extraction_op.num_elements
end

include("CompositeFunctionSpaces.jl")
include("UnivariateSplineSpaces.jl")
include("UnivariateSplineExtractions.jl")
include("KnotInsertion.jl")

"""
    create_bspline_space(patch::Mesh.Patch{n}, degree::Vector{Int}, regularity::NTuple{n, Vector{Int}}) where {n}

Create a tensor product space made of only univariate b-spline spaces.

# Arguments
- `patch::Patch{n}`: Patch on which the b-spline space should be defined.
- `degree::NTuple{n, Int}`: Polynomial degree per dimension.
- `regularity::NTuple{n, Vector{Int}}`: Regularity per dimension per breakpoint.

# Returns
- `TensorProductSpace{n}`: Tensor product space of univariate b-splines.
"""
function create_bspline_space(patch::Mesh.Patch{n}, degree::Vector{Int}, regularity::NTuple{n, Vector{Int}}) where {n}
    f_spaces = NTuple{n, BSplineSpace}(BSplineSpace(patch[i], degree[i], regularity[i]) for i in 1:1:n)
    return TensorProductSpace{n}(patch, f_spaces)
end



end
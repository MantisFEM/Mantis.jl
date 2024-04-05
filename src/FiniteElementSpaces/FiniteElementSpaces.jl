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
function get_dim(extraction_op::ExtractionOperator)
    return extraction_op.space_dim
end

function get_extraction(extraction_op::ExtractionOperator, element_id::Int)
    return @views extraction_op.extraction_coefficients[element_id], extraction_op.basis_indices[element_id]
end

function get_num_elements(extraction_op::ExtractionOperator)
    return extraction_op.num_elements
end

# univariate function spaces
include("UnivariateSplineSpaces.jl")
include("UnivariateSplineExtractions.jl")
# composite function spaces
include("UnstructuredSpaces.jl")
include("TensorProductSpaces.jl")

end
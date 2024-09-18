
@doc raw"""
    struct ExtractionOperator

Structure to store extraction operators and coefficients.

# Fields
- `extraction_coefficients::Vector{Matrix{Float64}}`: A vector of extraction coefficient matrices, where each matrix corresponds to an element.
- `basis_indices::Vector{Vector{Int}}`: A vector of basis index vectors, where each vector corresponds to the basis indices for an element.
- `num_elements::Int`: The number of elements.
- `space_dim::Int`: The dimension of the function space.
"""
struct ExtractionOperator
    extraction_coefficients::Vector{Matrix{Float64}}
    basis_indices::Vector{Vector{Int}}
    num_elements::Int
    space_dim::Int
end

@doc raw"""
    get_num_basis(extraction_op::ExtractionOperator)

Returns the dimension of the function space associated with the `ExtractionOperator`.

# Arguments
- `extraction_op::ExtractionOperator`: The `ExtractionOperator` object.

# Returns
The dimension of the function space.
"""
function get_num_basis(extraction_op::ExtractionOperator)
    return extraction_op.space_dim
end

@doc raw"""
    get_basis_indices(extraction_op::ExtractionOperator, element_id::Int)

Returns the basis indices supported on the given element.

# Arguments
- `extraction_op::ExtractionOperator`: The `ExtractionOperator` object.
- `element_id::Int`: The index of the element.

# Returns
- `::Vector{Int}`: The basis indices supported on the given element.
"""
function get_basis_indices(extraction_op::ExtractionOperator, element_id::Int)
    return extraction_op.basis_indices[element_id]
end

@doc raw"""
    get_extraction(extraction_op::ExtractionOperator, element_id::Int)

Returns the extraction coefficient matrix and basis indices for a specific element.

# Arguments
- `extraction_op::ExtractionOperator`: The `ExtractionOperator` object.
- `element_id::Int`: The index of the element.

# Returns
A tuple containing:
- The extraction coefficient matrix for the specified element.
- The basis indices for the specified element.
"""
function get_extraction(extraction_op::ExtractionOperator, element_id::Int)
    return @views extraction_op.extraction_coefficients[element_id], get_basis_indices(extraction_op, element_id)
end

@doc raw"""
    get_num_elements(extraction_op::ExtractionOperator)

Returns the number of elements in the `ExtractionOperator`.

# Arguments
- `extraction_op::ExtractionOperator`: The `ExtractionOperator` object.

# Returns
The number of elements.
"""
function get_num_elements(extraction_op::ExtractionOperator)
    return extraction_op.num_elements
end

"""
    Indices{num_components, TI, TJ}

Stores basis indices for an `ExtractionOperator`.

# Fields

  - `I::TI`: Global basis indices per element. Should be a vector-like type with integer
    elements.
  - `J::NTuple{num_components, TJ}`: 'Permutation' of the basis indices. This
    tells the `evaluate` function to which basis on an element the evaluations correspond.
"""
struct Indices{num_components, TI, TJ}
    I::TI
    J::NTuple{num_components, TJ}

    function Indices(I::TI, J::NTuple{num_components, TJ}) where {num_components, TI, TJ}
        if eltype(I) != Int
            throw(ArgumentError("The elements in Indices.I must be of type Int."))
        end
        if eltype(eltype(J)) != Int
            throw(ArgumentError("The elements in Indices.J must be of type Int."))
        end

        return new{num_components, TI, TJ}(I, J)
    end
end

get_num_components(::Indices{num_components, TI, TJ}) where {num_components, TI, TJ} =
    num_components
get_basis_indices(indices::Indices) = indices.I
get_basis_permutation(indices::Indices, component_id::Int) = indices.J[component_id]

"""
    ExtractionOperator{num_components, TE, TI, TJ}

Stores extraction coefficients and basis indices for a function space.

# Fields

  - `extraction_coefficients::Vector{NTuple{num_components, TE}}`: A vector of extraction
    coefficient matrices, where each matrix corresponds to an element. TE should be matrix-
    like.
  - `basis_indices::Vector{TI}`: A vector of `Indices`. See [`Indices`](@ref) for more
    details.
  - `num_elements::Int`: The number of elements.
  - `num_basis::Int`: The (total) dimension of the function space.
"""
struct ExtractionOperator{num_components, TE, TI, TJ}
    extraction_coefficients::Vector{NTuple{num_components, TE}}
    basis_indices::Vector{Indices{num_components, TI, TJ}}
    num_elements::Int
    num_basis::Int

    function ExtractionOperator(
        extraction_coefficients::Vector{NTuple{num_components, TE}},
        basis_indices::Vector{Indices{num_components, TI, TJ}},
        num_elements::Int,
        num_basis::Int,
    ) where {num_components, TE, TI, TJ}
        if length(extraction_coefficients) != num_elements
            throw(
                ArgumentError(
                    "Number of extraction coefficient matrices must match number of elements.",
                ),
            )
        end

        if length(basis_indices) != num_elements
            throw(
                ArgumentError(
                    "Number of basis index vectors must match number of elements."
                ),
            )
        end

        return new{num_components, TE, TI, TJ}(
            extraction_coefficients, basis_indices, num_elements, num_basis
        )
    end
end

get_num_components(
    ::ExtractionOperator{num_components, TE, TI, TJ}
) where {num_components, TE, TI, TJ} = num_components
get_EIJ_types(
    ::ExtractionOperator{num_components, TE, TI, TJ}
) where {num_components, TE, TI, TJ} = (TE, TI, TJ)
get_extraction_type(
    ::ExtractionOperator{num_components, TE, TI, TJ}
) where {num_components, TE, TI, TJ} = TE
get_index_type(
    ::ExtractionOperator{num_components, TE, TI, TJ}
) where {num_components, TE, TI, TJ} = TI
get_permutation_type(
    ::ExtractionOperator{num_components, TE, TI, TJ}
) where {num_components, TE, TI, TJ} = TJ

function _get_index_operator(extraction_op::ExtractionOperator, element_id::Int)
    return extraction_op.basis_indices[element_id]
end

function get_extraction_coefficients(
    extraction_op::ExtractionOperator, element_id::Int, component_id::Int=1
)
    return extraction_op.extraction_coefficients[element_id][component_id]
end

function get_basis_indices(extraction_op::ExtractionOperator, element_id::Int)
    return get_basis_indices(_get_index_operator(extraction_op, element_id))
end

function get_basis_permutation(
    extraction_op::ExtractionOperator, element_id::Int, component_id::Int=1
)
    return get_basis_permutation(
        _get_index_operator(extraction_op, element_id), component_id
    )
end

get_num_basis(extraction_op::ExtractionOperator) = extraction_op.num_basis

function get_num_basis(extraction_op::ExtractionOperator, element_id::Int)
    return length(get_basis_indices(extraction_op, element_id))
end

function get_extraction(
    extraction_op::ExtractionOperator, element_id::Int, component_id::Int=1
)
    return (
        get_extraction_coefficients(extraction_op, element_id, component_id),
        get_basis_permutation(extraction_op, element_id, component_id),
    )
end

get_num_elements(extraction_op::ExtractionOperator) = extraction_op.num_elements

"""
Functions and algorithms used for two scale relations.
"""

"""
    AbstractTwoScaleOperator

Supertype for all two scale relations.
"""
abstract type AbstractTwoScaleOperator{manifold_dim, num_components, num_patches} end

"""
    get_global_subdiv_matrix(operator::AbstractTwoScaleOperator)

Retrieve and return the global subdivision matrix associated with a two-scale operator.

# Arguments
- `operator::AbstractTwoScaleOperator`: The two-scale operator from which to
    retrieve the global subdivision matrix.

# Returns
- `::SparseArrays.SparseMatrixCSC{Float64, Int}`: The global subdivision matrix associated
    with the two-scale operator.
"""
get_global_subdiv_matrix(operator::AbstractTwoScaleOperator) = operator.global_subdiv_matrix

"""
    get_coarse_space(operator::TwoScaleOperator)

Retrieve and return the coarse space associated with a two-scale operator.

# Arguments
- `operator::TwoScaleOperator`: The two-scale operator from which to retrieve the
    coarse space.

# Returns
- `::AbstractFESpace`: The coarse space associated with the two-scale operator.
"""
get_parent_space(operator::AbstractTwoScaleOperator) = operator.parent_space

"""
    get_child_space(operator::TwoScaleOperator)
Retrieve and return the fine space associated with a two-scale operator.

# Arguments
- `operator::TwoScaleOperator`: The two-scale operator from which to retrieve the
    fine space.

# Returns
- `::AbstractFESpace`: The fine space associated with the two-scale operator.
"""
get_child_space(operator::AbstractTwoScaleOperator) = operator.child_space

"""
    get_element_ancestor(two_scale_operators, child_element_id::Int, child_level::Int, num_ancestor_levels::Int)

Compute and return the ancestor element ID for a given child element ID within a specified
number of ancestor levels.

# Arguments
- `two_scale_operators`: A collection of operators that define the parent-child
    relationships between elements at different levels.
- `child_element_id::Int`: The identifier of the child element w.r.t. `child_level`.
- `child_level::Int`: The level of the child element.
- `num_ancestor_levels::Int`: The number of ancestor levels to traverse.

# Returns
- `ancestor_id::Int`: The identifier of the ancestor element.
"""
function get_element_ancestor(
    two_scale_operators::Vector{TS},
    child_element_id::Int,
    child_level::Int,
    num_ancestor_levels::Int,
) where {TS <: AbstractTwoScaleOperator}
    ancestor_id = child_element_id
    for level in child_level:-1:(child_level - num_ancestor_levels + 1)
        ancestor_id = get_element_parent(two_scale_operators[level - 1], ancestor_id)
    end

    return ancestor_id
end

"""
    get_child_basis_coefficients(coarse_basis_coeffs::Vector{Float64}, operator::AbstractTwoScaleOperator)

Perform a change of basis using a two-scale operator from a coarse basis and return the
resulting fine basis coefficients.

# Arguments
- `coarse_basis_coeffs::Vector{Float64}`: A vector containing the coefficients of the
    coarse basis.
- `operator::T`: The two-scale operator that defines the subdivision process,
    where `T` is a subtype of `AbstractTwoScaleOperator`.

# Returns
- `::Vector{Float64}`: A vector containing the coefficients of the fine basis after
    subdivision.
"""
function get_child_basis_coefficients(
    parent_basis_coefficients::AbstractVector{Float64}, operator::AbstractTwoScaleOperator
)
    return get_global_subdiv_matrix(operator) * parent_basis_coefficients
end

"""
    get_local_subdiv_matrix(operator::AbstractTwoScaleOperator, coarse_element_id::Int, fine_element_id::Int)

Retrieve and return the local subdivision matrix for a given pair of coarse and fine
elements within a two-scale operator.

# Arguments
- `operator::AbstractTwoScaleOperator`: The two-scale operator that defines the
    subdivision process.
- `coarse_element_id::Int`: The identifier of the coarse element.
- `fine_element_id::Int`: The identifier of the fine element.

# Returns
- `::Matrix{Float64}`: The local subdivision matrix corresponding to the specified coarse
    and fine elements.
"""
function get_local_subdiv_matrix(
    operator::AbstractTwoScaleOperator, parent_element_id::Int, child_element_id::Int
)
    child_basis_indices = get_basis_indices(get_child_space(operator), child_element_id)
    parent_basis_indices = get_basis_indices(get_parent_space(operator), parent_element_id)

    return get_global_subdiv_matrix(operator)[child_basis_indices, parent_basis_indices]
end

function get_parent_child_relations(operator::AbstractTwoScaleOperator)
    return operator.parent_child_relations
end

function get_element_children(operator::AbstractTwoScaleOperator, parent::Int)
    return get_element_children(get_parent_child_relations(operator), parent)
end

function get_element_parent(operator::AbstractTwoScaleOperator, child::Int)
    return get_element_parent(get_parent_child_relations(operator), child)
end

function get_basis_children(operator::AbstractTwoScaleOperator, parent::Int)
    return get_basis_children(get_parent_child_relations(operator), parent)
end

function get_basis_parents(operator::AbstractTwoScaleOperator, child::Int)
    return get_basis_parents(get_parent_child_relations(operator), child)
end

function get_parent_to_children_elements(operator::AbstractTwoScaleOperator)
    return get_parent_to_children_elements(get_parent_child_relations(operator))
end

function get_child_to_parent_elements(operator::AbstractTwoScaleOperator)
    return get_child_to_parent_elements(get_parent_child_relations(operator))
end

function get_parent_to_children_basis(operator::AbstractTwoScaleOperator)
    return get_parent_to_children_basis(get_parent_child_relations(operator))
end

function get_child_to_parent_basis(operator::AbstractTwoScaleOperator)
    return get_child_to_parents_basis(get_parent_child_relations(operator))
end

# Includes for concrete two scale relations

include("ParentChildRelations.jl")
include("TwoScaleRelations.jl")
include("UnivariateTwoScaleRelations.jl")
include("UnivariateBSplineTwoScaleRelations.jl")
include("TensorProductTwoScaleRelations.jl")
include("DirectSumTwoScaleRelations.jl")
include("GTBSplineTwoScaleRelations.jl")
include("PolarSplineTwoScaleRelations.jl")

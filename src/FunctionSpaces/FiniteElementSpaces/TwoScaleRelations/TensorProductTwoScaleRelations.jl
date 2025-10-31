"""
    struct TensorProductTwoScaleOperator{manifold_dim, TP, TS} <: AbstractTwoScaleOperator{manifold_dim}

A structure representing a two-scale operator for tensor product spaces, defining the
relationships between coarse and fine tensor product spaces.

# Fields
- `coarse_space::TP`: The coarse tensor product space.
- `fine_space::TP`: The fine tensor product space.
- `global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}`: The global
    subdivision matrix for the tensor product space.
- `twoscale_operators::TS`: A tuple of two-scale operators for each constituent space.
"""
struct TensorProductTwoScaleOperator{
    manifold_dim, num_components, num_patches, num_spaces, TP, TS, R
} <: AbstractTwoScaleOperator{manifold_dim, num_components, num_patches}
    parent_space::TP
    child_space::TP
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    twoscale_operators::TS
    parent_child_relations::R

    function TensorProductTwoScaleOperator(
        parent_space::TP, child_space::TP, twoscale_operators::TS
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        num_spaces,
        T <: NTuple{num_spaces, AbstractFESpace},
        TP <: TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces, T},
        TS <: NTuple{num_spaces, AbstractTwoScaleOperator},
    }
        gm = kron(
            (twoscale_operators[space].global_subdiv_matrix for space in num_spaces:-1:1)...
        )
        let operator_ref = Ref{TensorProductTwoScaleOperator}()
            parent_child_relations = ParentChildRelations(
                parent -> get_element_children(operator_ref[], parent),
                child -> get_element_parent(operator_ref[], child),
                parent -> get_basis_children(operator_ref[], parent),
                child -> get_basis_parents(operator_ref[], child),
            )
            R = typeof(parent_child_relations)
            operator = new{manifold_dim, num_components, num_patches, num_spaces, TP, TS, R}(
                parent_space, child_space, gm, twoscale_operators, parent_child_relations
            )
            operator_ref[] = operator

            return operator
        end
    end
end

function get_constituent_twoscale_operators(operator::TensorProductTwoScaleOperator)
    return operator.twoscale_operators
end

# Basic getters per two scale operator

function get_constituent_element_children(
    operator::TensorProductTwoScaleOperator{
        manifold_dim, num_components, num_patches, num_spaces
    },
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_element_id = get_constituent_element_id(get_parent_space(operator), element_id)
    twoscale_operators = get_constituent_twoscale_operators(operator)
    const_children = ntuple(
        space -> get_element_children(twoscale_operators[space], const_element_id[space]),
        num_spaces,
    )

    return const_children
end

function get_constituent_element_parent(
    operator::TensorProductTwoScaleOperator{
        manifold_dim, num_components, num_patches, num_spaces
    },
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_element_id = get_constituent_element_id(get_child_space(operator), element_id)
    twoscale_operators = get_constituent_twoscale_operators(operator)
    const_parent = ntuple(
        space -> get_element_parent(twoscale_operators[space], const_element_id[space]),
        num_spaces,
    )

    return const_parent
end

function get_constituent_basis_children(
    operator::TensorProductTwoScaleOperator{
        manifold_dim, num_components, num_patches, num_spaces
    },
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_basis_id = get_constituent_basis_id(get_parent_space(operator), basis_id)
    twoscale_operators = get_constituent_twoscale_operators(operator)
    const_children = ntuple(
        space -> get_basis_children(twoscale_operators[space], const_basis_id[space]),
        num_spaces,
    )

    return const_children
end

function get_constituent_basis_parent(
    operator::TensorProductTwoScaleOperator{
        manifold_dim, num_components, num_patches, num_spaces
    },
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_basis_id = get_constituent_basis_id(get_child_space(operator), basis_id)
    twoscale_operators = get_constituent_twoscale_operators(operator)
    const_parent = ntuple(
        space -> get_basis_parents(twoscale_operators[space], const_basis_id[space]),
        num_spaces,
    )

    return const_parent
end

# Basis getters for TensorProductTwoScaleOperator

"""
    get_element_children(operator::TensorProductTwoScaleOperator, element_id::Int)

Retrieve and return the child element IDs for a given element ID within a tensor product
two-scale operator.

# Arguments
- `operator::TensorProductTwoScaleOperator`: The tensor product two-scale operator
    that defines the parent-child relationships between elements.
- `element_id::Int`: The identifier of the element whose children are to be retrieved.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child elements.
"""
function get_element_children(operator::TensorProductTwoScaleOperator, element_id::Int)
    const_element_children = get_constituent_element_children(operator, element_id)
    child_lin_num_elements = get_lin_num_elements(get_child_space(operator))
    element_children = Vector{Int}(undef, prod(length, const_element_children))
    for (child_count, const_child_id) in
        enumerate(Iterators.product(const_element_children...))
        element_children[child_count] = child_lin_num_elements[const_child_id...]
    end

    return element_children
end

"""
    get_element_parent(operator::TensorProductTwoScaleOperator, element_id::Int)

Retrieve and return the parent element ID for a given element ID within a tensor product
two-scale operator.

# Arguments
- `operator::TensorProductTwoScaleOperator`: The tensor product two-scale operator
    that defines the parent-child relationships between elements.
- `element_id::Int`: The identifier of the element whose parent is to be retrieved.

# Returns
- `::Int`: The identifier of the parent element.
"""
function get_element_parent(operator::TensorProductTwoScaleOperator, element_id::Int)
    const_element_parent = get_constituent_element_parent(operator, element_id)
    parent_lin_num_elements = get_lin_num_elements(get_parent_space(operator))

    return parent_lin_num_elements[const_element_parent...]
end

"""
    get_basis_children(operator::TensorProductTwoScaleOperator, basis_id::Int)

Retrieve and return the child basis function IDs for a given basis function ID within a
tensor product two-scale operator.

# Arguments
- `operator::TensorProductTwoScaleOperator`: The tensor product two-scale operator
    that defines the parent-child relationships between basis functions.
- `basis_id::Int`: The identifier of the basis function whose children are to be retrieved.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child basis functions.
"""
function get_basis_children(operator::TensorProductTwoScaleOperator, basis_id::Int)
    const_basis_children = get_constituent_basis_children(operator, basis_id)
    child_lin_num_basis = get_lin_num_basis(get_child_space(operator))
    basis_children = Vector{Int}(undef, prod(length, const_basis_children))
    for (child_count, const_child_id) in
        enumerate(Iterators.product(const_basis_children...))
        basis_children[child_count] = child_lin_num_basis[const_child_id...]
    end

    return basis_children
end

"""
    get_basis_parents(operator::TensorProductTwoScaleOperator, basis_id::Int)

Retrieve and return the parent basis functions for a given basis function ID within a
tensor product two-scale operator.

# Arguments
- `operator::TensorProductTwoScaleOperator`: The tensor product two-scale operator
    that defines the parent-child relationships between basis functions.
- `basis_id::Int`: The identifier of the basis function whose parent is to be retrieved.

# Returns
- `::Vector{Int}`: The identifier of the parent basis functions.
"""
function get_basis_parents(operator::TensorProductTwoScaleOperator, basis_id::Int)
    const_basis_parents = get_constituent_basis_parent(operator, basis_id)
    parent_lin_num_basis = get_lin_num_basis(get_parent_space(operator))
    basis_parents = Vector{Int}(undef, prod(length, const_basis_parents))
    for (parent_count, const_parent_id) in
        enumerate(Iterators.product(const_basis_parents...))
        basis_parents[parent_count] = parent_lin_num_basis[const_parent_id...]
    end

    return basis_parents
end

"""
    subdivide_space(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, T},
        nsubdivisions::NTuple{num_spaces, Int}
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        num_spaces,
        T <: NTuple{num_spaces, AbstractFESpace},
    }

Subdivide the finite element spaces within a tensor product space and return the resulting
finer tensor product space.

# Arguments
- `space::TensorProductSpace`: The tensor product space containing finite
    element spaces to be subdivided.
- `nsubdivisions::NTuple{num_spaces, Int}`: A tuple specifying the number of subdivisions
    for each finite element space.

# Returns
- `::TensorProductSpace`: The resulting finer tensor product space after subdivision.
"""
function subdivide_space(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    num_subdivisions::NTuple{num_spaces, Int},
) where {manifold_dim, num_components, num_patches, num_spaces}
    constituent_spaces = get_constituent_spaces(space)
    subdivided_spaces = ntuple(
        space -> subdivide_space(constituent_spaces[space], num_subdivisions[space]),
        num_spaces,
    )

    return TensorProductSpace(subdivided_spaces)
end

"""
    build_two_scale_operator(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, T},
        nsubdivisions::NTuple{num_spaces, Int}
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        num_spaces,
        T <: NTuple{num_spaces, AbstractFESpace},
    }

Build a two-scale operator for a tensor product space by subdividing each constituent
finite element space.

# Arguments
- `space::TensorProductSpace`: The tensor product space to be subdivided.
- `nsubdivisions::NTuple{num_spaces, Int}`: A tuple specifying the number of subdivisions
    for each constituent finite element space.

# Returns
- `::TensorProductTwoScaleOperator`: The two-scale operator.
- `::TensorProductSpace`: The resulting finer tensor product space after subdivision.
"""
function build_two_scale_operator(
    parent_space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    num_subdivisions::NTuple{num_spaces, Int},
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_parent_spaces = get_constituent_spaces(parent_space)
    twoscale_data = ntuple(
        space ->
            build_two_scale_operator(const_parent_spaces[space], num_subdivisions[space]),
        num_spaces,
    )
    twoscale_operators = ntuple(space -> twoscale_data[space][1], num_spaces)
    const_child_spaces = ntuple(space -> twoscale_data[space][2], num_spaces)
    child_space = TensorProductSpace(const_child_spaces)

    return TensorProductTwoScaleOperator(parent_space, child_space, twoscale_operators),
    child_space
end

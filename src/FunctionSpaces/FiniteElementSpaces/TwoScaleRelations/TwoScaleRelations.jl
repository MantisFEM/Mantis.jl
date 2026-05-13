"""
    struct TwoScaleOperator{manifold_dim, S} <: AbstractTwoScaleOperator{manifold_dim}

A structure representing a two-scale operator that dechilds the relationships between parent
and child finite element spaces.

# Fields
- `parent_space::S`: The parent finite element space.
- `child_space::S`: The child finite element space.
- `global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}`: The global subdivision
    matrix. The size of this matrix is `(num_child_basis, num_parent_basis)` where
    `num_child_basis` is the dimension of `child_space` and `num_parent_basis` the dimension
    of `parent_space`.
- `parent_to_child_elements::Vector{Vector{Int}}`: A vector of vectors containing the child
    element IDs for each parent element.
- `child_to_parent_elements::Vector{Int}`: A vector containing the parent element ID for
    each child element.
- `parent_to_child_basis::Vector{Vector{Int}}`: A vector of vectors containing the child
    basis function IDs for each parent basis function.
- `child_to_parent_basis::Vector{Vector{Int}}`: A vector of vectors containing the
    parent basis function IDs for each child basis function.
"""
struct TwoScaleOperator{manifold_dim, num_components, num_patches, PS, CS, R} <:
       AbstractTwoScaleOperator{manifold_dim, num_components, num_patches}
    parent_space::PS
    child_space::CS
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    parent_child_relations::R

    function TwoScaleOperator(
        parent_space::PS,
        child_space::CS,
        global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int},
        parent_to_child_elements::PCE,
        child_to_parent_elements::CPE,
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        PS <: AbstractFESpace{manifold_dim, num_components, num_patches},
        CS <: AbstractFESpace{manifold_dim, num_components, num_patches},
        PCE,
        CPE,
    }
        num_child_basis, num_parent_basis = size(global_subdiv_matrix)
        parent_to_child_basis = Vector{Vector{Int}}(undef, num_parent_basis)
        child_to_parent_basis = Vector{Vector{Int}}(undef, num_child_basis)
        gm_data = SparseArrays.findnz(global_subdiv_matrix)
        transpose_matrix = SparseArrays.sparse(gm_data[2], gm_data[1], gm_data[3])
        for i in 1:num_parent_basis
            parent_to_child_basis[i] = global_subdiv_matrix.rowval[SparseArrays.nzrange(
                global_subdiv_matrix, i
            )]
        end

        for i in 1:num_child_basis
            child_to_parent_basis[i] = transpose_matrix.rowval[SparseArrays.nzrange(
                transpose_matrix, i
            )]
        end

        parent_child_relations = ParentChildRelations(
            parent_to_child_elements,
            child_to_parent_elements,
            parent_to_child_basis,
            child_to_parent_basis,
        )
        R = typeof(parent_child_relations)

        return new{manifold_dim, num_components, num_patches, PS, CS, R}(
            parent_space, child_space, global_subdiv_matrix, parent_child_relations
        )
    end
end

"""
    compose(op1::TwoScaleOperator, op2::TwoScaleOperator)

Compose two two-scale operators `op1` (from space A to B) and `op2` (from space B to C)
to create a new two-scale operator from space A to C.
"""
function compose(op1::TwoScaleOperator, op2::TwoScaleOperator)
    parent_space = op2.parent_space
    child_space = op1.child_space
    global_subdiv_matrix = op1.global_subdiv_matrix * op2.global_subdiv_matrix
    
    parent_to_child_elements = parent -> begin
        children_in_B = get_element_children(op2.parent_child_relations, parent)
        children_in_C = Int[]
        for cB in children_in_B
            append!(children_in_C, get_element_children(op1.parent_child_relations, cB))
        end
        return unique(children_in_C)
    end
    
    child_to_parent_elements = child -> begin
        parent_in_B = get_element_parent(op1.parent_child_relations, child)
        return get_element_parent(op2.parent_child_relations, parent_in_B)
    end

    return TwoScaleOperator(
        parent_space,
        child_space,
        global_subdiv_matrix,
        parent_to_child_elements,
        child_to_parent_elements
    )
end

function build_trivial_two_scale_operator(parent_space::AbstractFESpace)
    parent_to_child_elements = parent -> [parent]
    child_to_parent_elements = child -> child
    return TwoScaleOperator(
        parent_space,
        parent_space,
        SparseArrays.sparse(1.0 * LinearAlgebra.I, get_num_basis(parent_space), get_num_basis(parent_space)),
        parent_to_child_elements,
        child_to_parent_elements
    )
end

"""
    ∘

Operator for composition. The unicode character command is `\\circ`.
"""
const ∘ = compose

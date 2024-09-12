struct TensorProductTwoScaleOperator{n, F1, F2, T1, T2} <: AbstractTwoScaleOperator
    coarse_space::F1
    fine_space::F2
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    twoscale_operator_1::T1
    twoscale_operator_2::T2

    function TensorProductTwoScaleOperator(twoscale_operator_1::T1, twoscale_operator_2::T2) where {T1<:AbstractTwoScaleOperator, T2<:AbstractTwoScaleOperator}

        coarse_space = TensorProductSpace(twoscale_operator_1.coarse_space, twoscale_operator_2.coarse_space)
        fine_space = TensorProductSpace(twoscale_operator_1.fine_space, twoscale_operator_2.fine_space)

        get_manifold_dim(coarse_space) == get_manifold_dim(fine_space) ? nothing : throw(ArgumentError("Tensor product dimensions do not match."))

        n = get_manifold_dim(coarse_space)
        
        gm = kron(twoscale_operator_1.global_subdiv_matrix, twoscale_operator_2.global_subdiv_matrix)

        F1 = typeof(coarse_space); F2 = typeof(fine_space)

        new{n, F1, F2, T1, T2}(coarse_space, fine_space, gm, twoscale_operator_1, twoscale_operator_2)
    end
end

# Getters for TensorProductTwoScaleOperator

function get_local_subdiv_matrix(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, fine_el_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    ordered_index = linear_to_ordered_index(fine_el_id, _get_num_elements_per_space(twoscale_operator.fine_space))

    local_subdiv_matrix = kron(get_local_subdiv_matrix(twoscale_operator.twoscale_operator_2, ordered_index[2]), get_local_subdiv_matrix(twoscale_operator.twoscale_operator_1, ordered_index[1]))
        
    return local_subdiv_matrix
end

function get_finer_elements(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, el_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    max_ind_f = _get_num_elements_per_space(twoscale_operator.fine_space)

    finer_ordered_indices = _get_finer_elements_per_space(twoscale_operator, el_id)

    finer_elements = Vector{Int}(undef, prod(length.(finer_ordered_indices)))

    idx = 1
    for el ∈ Iterators.product(finer_ordered_indices[1], finer_ordered_indices[2])
        finer_elements[idx] = ordered_to_linear_index(collect(el), max_ind_f)
        idx += 1 
    end

    return finer_elements
end

function _get_finer_elements_per_space(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, el_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    max_ind_c = _get_num_elements_per_space(twoscale_operator.coarse_space)
    ordered_index = linear_to_ordered_index(el_id, max_ind_c)

    return (get_finer_elements(twoscale_operator.twoscale_operator_1, ordered_index[1]), get_finer_elements(twoscale_operator.twoscale_operator_2, ordered_index[2]))
end

function get_finer_elements(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, el_ids::Vector{Int}) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    return reduce(vcat, get_finer_elements.(Ref(twoscale_operator), el_ids))
end


function get_coarser_element(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, el_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    max_ind_c = _get_num_elements_per_space(twoscale_operator.coarse_space)

    coarser_ordered_indices = _get_coarser_elements_per_space(twoscale_operator, el_id)

    return ordered_to_linear_index(coarser_ordered_indices, max_ind_c)
end



function _get_coarser_elements_per_space(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, el_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    max_ind_f = _get_num_elements_per_space(twoscale_operator.fine_space)
    ordered_index = linear_to_ordered_index(el_id, max_ind_f)

    return (get_coarser_element(twoscale_operator.twoscale_operator_1, ordered_index[1]), get_coarser_element(twoscale_operator.twoscale_operator_2, ordered_index[2]))
end


function get_finer_basis_id(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, basis_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    max_ind_f = _get_num_basis_per_space(twoscale_operator.fine_space)

    finer_ordered_indices = _get_finer_basis_id_per_space(twoscale_operator, basis_id)

    finer_basis_ids = Vector{Int}(undef, prod(length.(finer_ordered_indices)))

    idx = 1
    for id ∈ Iterators.product(finer_ordered_indices...)
        finer_basis_ids[idx] = ordered_to_linear_index(collect(id), max_ind_f)
        idx += 1 
    end

    return finer_basis_ids
end

function _get_finer_basis_id_per_space(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, basis_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    max_ind_c = _get_num_basis_per_space(twoscale_operator.coarse_space)
    ordered_index = linear_to_ordered_index(basis_id, max_ind_c)

    return (get_finer_basis_id(twoscale_operator.twoscale_operator_1, ordered_index[1]), get_finer_basis_id(twoscale_operator.twoscale_operator_2, ordered_index[2]))
end

function get_coarser_basis_id(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, basis_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    max_ind_c = _get_num_basis_per_space(twoscale_operator.coarse_space)
    
    coarser_ordered_indices = _get_coarser_basis_id_per_space(twoscale_operator, basis_id)

    coarser_basis_ids = Vector{Int}(undef, prod(length.(coarser_ordered_indices)))

    idx = 1
    for id ∈ Iterators.product(coarser_ordered_indices...)
        coarser_basis_ids[idx] = ordered_to_linear_index(collect(id), max_ind_c)
        idx += 1 
    end

    return coarser_basis_ids
end

function _get_coarser_basis_id_per_space(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, T1, T2}, basis_id::Int) where {n, F1<:AbstractFiniteElementSpace{n1} where {n1}, F2<:AbstractFiniteElementSpace{n2} where {n2}, T1<: AbstractTwoScaleOperator, T2<: AbstractTwoScaleOperator}
    max_ind_f = _get_num_basis_per_space(twoscale_operator.fine_space)

    ordered_index = linear_to_ordered_index(basis_id, max_ind_f)

    return (get_coarser_basis_id(twoscale_operator.twoscale_operator_1, ordered_index[1]), get_coarser_basis_id(twoscale_operator.twoscale_operator_2, ordered_index[2]))
end

function subdivide_bspline(tp_space::TensorProductSpace{n, F1, F2}, nsubdivs::Tuple{Int, Int}) where {n, F1 <: BSplineSpace, F2 <: BSplineSpace}
    space_1_ts, space_1_fine = subdivide_bspline(tp_space.function_space_1, nsubdivs[1])
    space_2_ts, space_2_fine = subdivide_bspline(tp_space.function_space_2, nsubdivs[2])

    return TensorProductTwoScaleOperator(space_1_ts, space_2_ts), TensorProductSpace(space_1_fine, space_2_fine)
end

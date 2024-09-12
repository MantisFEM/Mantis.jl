struct TensorProductTwoScaleOperator{n, F1, F2, O1, O2} <: AbstractTwoScaleOperator
    coarse_space::F1
    fine_space::F2
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Tv, Ti} where {Tv, Ti}
    twoscale_operator_1::O1
    twoscale_operator_2::O2

    function TensorProductTwoScaleOperator(twoscale_operator_1::O1, twoscale_operator_2::O2) where {O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator}

        coarse_space = TensorProductSpace(twoscale_operator_1.coarse_space, twoscale_operator_2.coarse_space)
        fine_space = TensorProductSpace(twoscale_operator_1.fine_space, twoscale_operator_2.fine_space)

       get_manifold_dim(coarse_space) == get_manifold_dim(fine_space) ? nothing : throw(ArgumentError("Tensor product dimensions do not match."))

        n = get_manifold_dim(coarse_space)
        
        gm = get_global_subdiv_matrix(twoscale_operator_2.global_subdiv_matrix, twoscale_operator_1.global_subdiv_matrix)

        F1 = typeof(coarse_space); F2 = typeof(fine_space) 

        new{n, F1, F2, O1, O2}(coarse_space, fine_space, gm, twoscale_operator_1, twoscale_operator_2)
    end
end

# Getters for TensorProductTwoScaleOperator

function get_global_subdiv_matrix(gm1::SparseArrays.SparseMatrixCSC{Tv1, Ti1}, gm2::SparseArrays.SparseMatrixCSC{Tv2, Ti2}) where {Tv1, Ti1, Tv2, Ti2}
    n_nonzero = SparseArrays.nnz(gm1) * SparseArrays.nnz(gm2)

    gm1_rows = SparseArrays.rowvals(gm1)
    gm1_vals = SparseArrays.nonzeros(gm1)
    gm2_rows = SparseArrays.rowvals(gm2)
    gm2_vals = SparseArrays.nonzeros(gm2)

    Nc1 = gm1.n
    Nf1 = gm1.m
    Nc2 = gm2.n
    Nf2 = gm2.m 

    max_indc = [Nc1, Nc2]
    max_indf = [Nf1, Nf2]
    
    gm_rows = Vector{Int}(undef, n_nonzero)
    gm_cols = Vector{Int}(undef, n_nonzero)
    gm_vals = Vector{Float64}(undef, n_nonzero)

    idx_count = 1
    for j1 ∈ 1:1:Nc1, i1 ∈ SparseArrays.nzrange(gm1, j1)
        r1 = gm1_rows[i1]
        fc = gm1_vals[i1]
        
        idx2_count = 1
        for j2 in 1:1:Nc2, i2 in SparseArrays.nzrange(gm2, j2)
            r2 = gm2_rows[i2]

            gm_rows[idx_count] = ordered_to_linear_index([r1,r2], max_indf)
            gm_cols[idx_count] = ordered_to_linear_index([j1,j2], max_indc)
            gm_vals[idx_count] = fc * gm2_vals[idx2_count]

            idx_count += 1
            idx2_count += 1
        end
    end

    return SparseArrays.sparse(gm_rows, gm_cols, gm_vals, Nf2*Nf1, Nc2*Nc1)
end

function get_local_subdiv_matrix(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, fine_el_id::Int) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
    ordered_index = linear_to_ordered_index(fine_el_id, _get_num_elements_per_space(twoscale_operator.fine_space))

    local_subdiv_matrix = kron(get_local_subdiv_matrix(twoscale_operator.twoscale_operator_2, ordered_index[2]), get_local_subdiv_matrix(twoscale_operator.twoscale_operator_1, ordered_index[1]))
        
    return local_subdiv_matrix
end

function get_finer_elements(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, el_id::Int) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
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

function _get_finer_elements_per_space(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, el_id::Int) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
    max_ind_c = _get_num_elements_per_space(twoscale_operator.coarse_space)
    ordered_index = linear_to_ordered_index(el_id, max_ind_c)

    return (get_finer_elements(twoscale_operator.twoscale_operator_1, ordered_index[1]), get_finer_elements(twoscale_operator.twoscale_operator_2, ordered_index[2]))
end

function get_finer_elements(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, el_ids::Vector{Int}) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
    return get_finer_elements.((twoscale_operator,), el_ids)
end

function get_coarser_element(twoscale_operator::TensorProductTwoScaleOperator{n,m}, el_id::Int) where {n, m}
    max_ind_c = _get_num_elements_per_space(twoscale_operator.coarse_space)

    coarser_ordered_indices = _get_coarser_elements_per_space(twoscale_operator, el_id)

    return ordered_to_linear_index(coarser_ordered_indices, max_ind_c)
end

function _get_coarser_elements_per_space(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, el_id::Int) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
    max_ind_f = _get_num_elements_per_space(twoscale_operator.fine_space)
    ordered_index = linear_to_ordered_index(el_id, max_ind_f)

    return (get_coarser_element(twoscale_operator.twoscale_operator_1, ordered_index[1]), get_coarser_element(twoscale_operator.twoscale_operator_2, ordered_index[2]))
end

function get_finer_basis_id(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, basis_id::Int) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
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

function _get_finer_basis_id_per_space(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, basis_id::Int) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
    max_ind_c = _get_num_basis_per_space(twoscale_operator.coarse_space)
    ordered_index = linear_to_ordered_index(basis_id, max_ind_c)

    return (get_finer_basis_id(twoscale_operator.twoscale_operator_1, ordered_index[1]), get_finer_basis_id(twoscale_operator.twoscale_operator_2, ordered_index[2]))
end

function get_coarser_basis_id(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, basis_id::Int) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
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

function _get_coarser_basis_id_per_space(twoscale_operator::TensorProductTwoScaleOperator{n, F1, F2, O1, O2}, basis_id::Int) where{n, F1<:TensorProductSpace{n, T1, T2} where{T1, T2},  F2<:TensorProductSpace{n, T3, T4} where{T3, T4}, O1<:AbstractTwoScaleOperator, O2<:AbstractTwoScaleOperator }
    max_ind_f = _get_num_basis_per_space(twoscale_operator.fine_space)

    ordered_index = linear_to_ordered_index(basis_id, max_ind_f)

    return (get_coarser_basis_id(twoscale_operator.twoscale_operator_1, ordered_index[1]), get_coarser_basis_id(twoscale_operator.twoscale_operator_2, ordered_index[2]))
end

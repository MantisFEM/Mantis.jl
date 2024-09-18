using LinearAlgebra
using SparseArrays
import Combinatorics

"""
    TensorProductSpace{n, F1, F2} <: AbstractFiniteElementSpace{n}

A tensor-product space built as the tensor-product of two input spaces.
The functions in this space are `n`-variate where `n` is equal to the sum of the
dimensions of the two input spaces.

# Fields
- `function_space_1::F1`: First input function space
- `function_space_2::F2`: Second input function space
- `dof_partition::Vector{Vector{Int}}`: Partition of degrees of freedom
- `data::Dict`: Additional user-defined data

# Constructors
    TensorProductSpace(function_space_1::F1, function_space_2::F2, data::Dict) where {F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}
    TensorProductSpace(function_space_1::F1, function_space_2::F2) where {F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}
"""
struct TensorProductSpace{n, F1, F2} <: AbstractFiniteElementSpace{n}
    function_space_1::F1
    function_space_2::F2
    dof_partition::Vector{Vector{Vector{Int}}}
    data::Dict

    function TensorProductSpace(function_space_1::F1, function_space_2::F2, data::Dict) where {F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}
        n = n1 + n2

        # Get dof partitions of the constituent spaces
        dof_partition_1 = get_dof_partition(function_space_1)
        n_patches_1 = length(dof_partition_1)
        dof_partition_2 = get_dof_partition(function_space_2)
        n_patches_2 = length(dof_partition_2)
        
        # Dimensions of constituent function spaces
        tp_dim = (get_num_basis(function_space_1), get_num_basis(function_space_2))
        
        # Allocate memory for degree of freedom partitioning
        dof_partition = Vector{Vector{Vector{Int}}}(undef, n_patches_1 * n_patches_2)
        
        # Loop over all dimensions and build the appropriate index subsets
        for i ∈ 1:n_patches_1
            partition_patch_1 = dof_partition_1[i]
            for j ∈ 1:n_patches_2
                partition_patch_2 = dof_partition_2[j]
                ij = i + (j-1)*n_patches_1
                dof_partition[ij] = Vector{Vector{Int}}(undef, length(partition_patch_1)*length(partition_patch_2))
                for l_1 ∈ eachindex(partition_patch_1)
                    for l_2 ∈ eachindex(partition_patch_2)
                        l = l_1 + (l_2-1)*length(partition_patch_1)
                        dof_partition[ij][l] = Vector{Int}(undef, length(partition_patch_1[l_1])*length(partition_patch_2[l_2]))
                        # Compute and store linearly indexed indices
                        idx = 1
                        for id in Iterators.product(partition_patch_1[l_1], partition_patch_2[l_2])
                            dof_partition[ij][l][idx] = ordered_to_linear_index(id, tp_dim)
                            idx += 1
                        end
                    end
                end
            end
        end
        new{n,F1,F2}(function_space_1, function_space_2, dof_partition, data)
    end

    function TensorProductSpace(function_space_1::F1, function_space_2::F2) where {F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}
        return TensorProductSpace(function_space_1, function_space_2, Dict())
    end
end


function get_basis_indices(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}
    max_ind_basis = _get_num_basis_per_space(tp_space)
    indices_per_dim = _get_basis_indices_per_dim(tp_space, el_id)
    
    # Compute basis indices
    basis_indices = Vector{Int}(undef, prod(length.(indices_per_dim)))
    idx = 1
    for basis in Iterators.product(indices_per_dim[1], indices_per_dim[2])
        basis_indices[idx] = ordered_to_linear_index(basis, max_ind_basis)
        idx += 1
    end
    
    return basis_indices
end

function _get_basis_indices_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}
    max_ind_el = _get_num_elements_per_space(tp_space)
    ordered_index = linear_to_ordered_index(el_id, max_ind_el)

    return get_basis_indices(tp_space.function_space_1, ordered_index[1]), get_basis_indices(tp_space.function_space_2, ordered_index[2])
end

"""
    get_num_elements(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Get the total number of elements for the partition over which the function space is defined.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space

# Returns
- `::Int`: Total number of elements
"""
function get_num_elements(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    return prod(_get_num_elements_per_space(tp_space))
end

"""
    _get_num_elements_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Helper function to get the number of elements for each constituent space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space

# Returns
- `::Tuple{Int,Int}`: Number of elements in each constituent space
"""
function _get_num_elements_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    return (get_num_elements(tp_space.function_space_1), get_num_elements(tp_space.function_space_2))
end

"""
    get_num_basis(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Get the dimension of the tensor-product function space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space

# Returns
- `::Int`: The dimension of the space
"""
function get_num_basis(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    return prod(_get_num_basis_per_space(tp_space))
end

"""
    _get_num_basis_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Helper function to get the dimension of each constituent space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space

# Returns
- `::Tuple{Int,Int}`: Dimensions of each constituent space
"""
function _get_num_basis_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    return (get_num_basis(tp_space.function_space_1), get_num_basis(tp_space.function_space_2))
end

"""
    get_polynomial_degree_per_dim(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}

Get the polynomial degree for each dimension of the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space

# Returns
- `::Tuple{Int,Int}`: Polynomial degrees for each constituent space
"""
function get_polynomial_degree_per_dim(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}
    return (get_polynomial_degree(tp_space.function_space_1), get_polynomial_degree(tp_space.function_space_2))
end

"""
    get_max_local_dim(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}

Get the maximum local dimension of the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space

# Returns
- `::Int`: The maximum local dimension
"""
function get_max_local_dim(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}
    return get_max_local_dim(tp_space.function_space_1) * get_max_local_dim(tp_space.function_space_2)
end

"""
    get_dof_partition(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}

Get the degree of freedom (dof) partition for the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space

# Returns
- `::Vector{Vector{Int}}`: The dof partition
"""
function get_dof_partition(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}
    return tp_space.dof_partition
end

"""
    get_support(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1, F2}

Get the support of a basis function in the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space
- `basis_id::Int`: ID of the basis function

# Returns
- `::Vector{Int}`: Support of the basis function
"""
function get_support(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1, F2}
    max_ind_els = _get_num_elements_per_space(tp_space)
    support_per_dim = _get_support_per_dim(tp_space, basis_id)
    
    support = Vector{Int}(undef, prod(length.(support_per_dim)))
    idx = 1
    for id in Iterators.product(support_per_dim[1], support_per_dim[2])
        support[idx] = ordered_to_linear_index(id, max_ind_els)
        idx += 1
    end
    
    return support
end

"""
    _get_support_per_dim(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1, F2}

Helper function to get the support of a basis function in each constituent space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space
- `basis_id::Int`: ID of the basis function

# Returns
- `::Tuple{Vector{Int},Vector{Int}}`: Support in each constituent space
"""
function _get_support_per_dim(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1, F2}
    max_ind_basis = _get_num_basis_per_space(tp_space)
    ordered_index = linear_to_ordered_index(basis_id, max_ind_basis)

    return get_support(tp_space.function_space_1, ordered_index[1]), get_support(tp_space.function_space_2, ordered_index[2])
end

using LinearAlgebra
using SparseArrays

"""
    get_extraction(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}

Get the extraction operator for an element in the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space.
- `el_id::Int`: ID of the element.

# Returns
- `::Tuple{Matrix{Float64}, Vector{Int}}`: Extraction coefficients and basis indices.
"""
function get_extraction(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}
    max_ind_basis = _get_num_basis_per_space(tp_space)
    extraction_per_dim = _get_extraction_per_dim(tp_space, el_id)
    
    # Compute Kronecker product of extraction coefficients
    extraction_coeffs = kron(extraction_per_dim[2][1], extraction_per_dim[1][1])
    
    # Compute basis indices
    basis_indices = Vector{Int}(undef, prod(map(extraction -> length(extraction[2]), extraction_per_dim)))
    idx = 1
    for basis in Iterators.product(extraction_per_dim[1][2], extraction_per_dim[2][2])
        basis_indices[idx] = ordered_to_linear_index(basis, max_ind_basis)
        idx += 1
    end
    
    return extraction_coeffs, basis_indices
end

"""
    _get_extraction_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}

Helper function to get the extraction operator for each constituent space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space.
- `el_id::Int`: ID of the element.

# Returns
- `::Tuple{Tuple{Matrix{Float64}, Vector{Int}}, Tuple{Matrix{Float64}, Vector{Int}}}`: Extraction operators for each constituent space.
"""
function _get_extraction_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}
    max_ind_el = _get_num_elements_per_space(tp_space)
    ordered_index = linear_to_ordered_index(el_id, max_ind_el)
    return (get_extraction(tp_space.function_space_1, ordered_index[1]),
            get_extraction(tp_space.function_space_2, ordered_index[2]))
end

"""
    get_local_basis(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}

Compute the local basis functions and their derivatives for a tensor product space element.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space.
- `el_id::Int`: Element ID.
- `xi::NTuple{n,Vector{Float64}}`: Tuple of vectors representing evaluation points in each dimension.
- `nderivatives::Int`: Number of derivatives to compute.

# Returns
- `Dict{NTuple{n,Int}, Matrix{Float64}}`: Dictionary mapping derivative orders to basis function values.
"""
function get_local_basis(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}

    # Compute local basis for each constituent space
    local_basis_per_dim = _get_local_basis_per_dim(tp_space, el_id, xi, nderivatives)

    # Generate keys for all possible derivative combinations
    der_keys = _integer_sums(nderivatives, n+1)
    # Initialize storage of local basis functions and derivatives
    local_basis = Vector{Vector{Matrix{Float64}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        num_j_ders = binomial(n + j - 1, n - 1)
        local_basis[j + 1] = Vector{Matrix{Float64}}(undef, num_j_ders)
    end

    # Compute tensor product of constituent basis functions for each derivative combination
    for key in der_keys
        key = key[1:n]
        j = sum(key)
        der_idx = _get_derivative_idx(key)
        key_1 = key[1:n1]
        j_1 = sum(key_1)
        der_idx_1 = _get_derivative_idx(key_1)
        key_2 = key[n1+1:n]
        j_2 = sum(key_2)
        der_idx_2 = _get_derivative_idx(key_2)
        local_basis[j + 1][der_idx] = kron(local_basis_per_dim[2][j_2 + 1][der_idx_2], local_basis_per_dim[1][j_1 + 1][der_idx_1])
    end

    return local_basis
end

"""
    _get_local_basis_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}

Helper function to compute local basis functions for each constituent space of a tensor product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: The tensor-product space.
- `el_id::Int`: Element ID.
- `xi::NTuple{n,Vector{Float64}}`: Tuple of vectors representing evaluation points in each dimension.
- `nderivatives::Int`: Number of derivatives to compute.

# Returns
- `Tuple{Tuple{Matrix{Float64}, Vector{Int}}, Tuple{Matrix{Float64}, Vector{Int}}}`: Local basis functions for each constituent space.
"""
function _get_local_basis_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1}, F2 <: AbstractFiniteElementSpace{n2}} where {n1, n2}
    # Get number of elements in each constituent space
    max_ind_el = _get_num_elements_per_space(tp_space)

    # Convert linear element ID to ordered index
    ordered_index = linear_to_ordered_index(el_id, max_ind_el)

    # Split evaluation points for each constituent space
    xi_1 = xi[1:n1]
    xi_2 = xi[n1+1:n]

    return get_local_basis(tp_space.function_space_1, ordered_index[1], xi_1, nderivatives), get_local_basis(tp_space.function_space_2, ordered_index[2], xi_2, nderivatives)
end

# Methods for ease of function space creation

function create_bspline_space(starting_point::NTuple{n, Float64}, box_size::NTuple{n, Float64}, num_elements::NTuple{n, Int}, degree::NTuple{n, Int}, regularity::NTuple{n, Int}) where {n}
    
    dim_wise_spaces = [create_bspline_space(starting_point[i], box_size[i], num_elements[i], degree[i], regularity[i]) for i ∈ 1:n]

    tensor_space = TensorProductSpace(dim_wise_spaces[1], dim_wise_spaces[2]) # Initialize tensor product construction

    for i ∈ 2:n-1
        tensor_space = TensorProductSpace(tensor_space, dim_wise_spaces[i+1])
    end

    return tensor_space
end
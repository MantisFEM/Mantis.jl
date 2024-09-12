using LinearAlgebra

"""
    RationalFiniteElementSpace{n,F} <: AbstractFiniteElementSpace{n}

A rational finite element space obtained by dividing all elements of a given function space by a fixed element from the same function space. The latter is defined with the help of specified weights.

# Fields
- `function_space::F`: The underlying function space.
- `weights::Vector{Float64}`: The weights associated with the basis functions of the function space.
"""
struct RationalFiniteElementSpace{n,F} <: AbstractFiniteElementSpace{n}
    function_space::F
    weights::Vector{Float64}

    function RationalFiniteElementSpace(function_space::F, weights::Vector{Float64}) where {F <: AbstractFiniteElementSpace{n}} where {n}
        # Ensure that the dimension of the function space matches the length of the weights vector
        @assert get_num_basis(function_space) == length(weights) "Dimension mismatch"
        new{n,F}(function_space, weights)
    end
end

"""
    get_num_basis(rat_space::RationalFiniteElementSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}

Returns the dimension of the rational finite element space.

# Arguments
- `rat_space::RationalFiniteElementSpace{n,F}`: The rational finite element space.

# Returns
The dimension of the rational finite element space.
"""
function get_num_basis(rat_space::RationalFiniteElementSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}
    return get_num_basis(rat_space.function_space)
end

"""
    get_num_elements(rat_space::RationalFiniteElementSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}

Returns the number of elements in the partition on which the rational finite element space is defined.

# Arguments
- `rat_space::RationalFiniteElementSpace{n,F}`: The rational finite element space.

# Returns
The number of elements in the rational finite element space.
"""
function get_num_elements(rat_space::RationalFiniteElementSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}
    return get_num_elements(rat_space.function_space)
end

"""
    get_polynomial_degree(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int) where {n, F <: AbstractFiniteElementSpace{n}}

Returns the polynomial degree (or the degree of the underlying function space) of the rational finite element space for a specific element.

# Arguments
- `rat_space::RationalFiniteElementSpace{n,F}`: The rational finite element space.
- `element_id::Int`: The index of the element.

# Returns
The polynomial degree (or the degree of the underlying function space) of the rational finite element space for the specified element.
"""
function get_polynomial_degree(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int) where {n, F}
    return get_polynomial_degree(rat_space.function_space, element_id)
end

@doc raw"""
    get_dof_partition(space::RationalFiniteElementSpace{n,F})

Get the partition of degrees of freedom for the rational finite element space.

# Arguments
- `space::RationalFiniteElementSpace{n,F}`: The rational finite element space.

# Returns
The partition of degrees of freedom.
"""
function get_dof_partition(space::RationalFiniteElementSpace{n,F}) where {n, F}
    return get_dof_partition(space.function_space)
end

"""
    evaluate(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F <: AbstractFiniteElementSpace{n}}

Evaluates the basis functions and their derivatives for the rational finite element space.

# Arguments
- `rat_space::RationalFiniteElementSpace{n,F}`: The rational finite element space.
- `element_id::Int`: The index of the element.
- `xi::NTuple{n,Vector{Float64}}`: The coordinates at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to compute.

# Returns
A tuple containing:
- The basis functions and their derivatives evaluated at the specified coordinates.
- The basis indices.
"""
function evaluate(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F <: AbstractFiniteElementSpace{n}}
    # Evaluate the basis functions and their derivatives for the underlying function space
    homog_basis, basis_indices = evaluate(rat_space.function_space, element_id, xi, nderivatives)
    n_eval = prod(length.(xi))

    # Compute the rational basis functions and their derivatives
    for j in 0:nderivatives
        if j == 0
            # Compute the weight
            temp = homog_basis[1][1] * LinearAlgebra.Diagonal(rat_space.weights[basis_indices])
            weight = reshape(sum(temp, dims=2), n_eval)
            # Rationalize with the weights
            homog_basis[1][1] .= LinearAlgebra.Diagonal(weight) \ temp
        elseif j == 1
            der_keys = _integer_sums(j, n)
            for key in der_keys
                # Get the location where the derivative is stored
                der_idx = _get_derivative_idx(key)
                # Compute the weight and its derivative
                temp = homog_basis[1][1] * LinearAlgebra.Diagonal(rat_space.weights[basis_indices])
                dtemp = homog_basis[j+1][der_idx] * LinearAlgebra.Diagonal(rat_space.weights[basis_indices])
                weight = reshape(sum(temp, dims=2), n_eval)
                dweight = reshape(sum(dtemp, dims=2), n_eval)
                # Compute the derivative of the rational basis functions
                homog_basis[j+1][der_idx] .= LinearAlgebra.Diagonal(weight) \ dtemp - LinearAlgebra.Diagonal(dweight ./ weight.^2) * temp
            end
        elseif j > 1
            error("Derivatives of rational spaces of order not implemented")
        end
    end

    return homog_basis, basis_indices
end

"""
    get_max_local_dim(rat_space::RationalFiniteElementSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}

Returns the maximum local dimension of the rational finite element space.

# Arguments
- `rat_space::RationalFiniteElementSpace{n,F}`: The rational finite element space.

# Returns
The maximum local dimension of the rational finite element space.
"""
function get_max_local_dim(rat_space::RationalFiniteElementSpace{n,F}) where {n, F <: AbstractFiniteElementSpace{n}}
    return get_max_local_dim(rat_space.function_space)
end

"""
    get_local_basis(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F <: AbstractFiniteElementSpace{n}}

Returns the local basis functions for the rational finite element space.

# Arguments
- `rat_space::RationalFiniteElementSpace{n,F}`: The rational finite element space.
- `element_id::Int`: The index of the element.
- `xi::NTuple{n,Vector{Float64}}`: The coordinates at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to compute.

# Returns
The local basis functions and their derivatives evaluated at the specified coordinates.
"""
function get_local_basis(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F <: AbstractFiniteElementSpace{n}}
    return evaluate(rat_space, element_id, xi, nderivatives)[1]
end

"""
    get_extraction(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int) where {n, F <: AbstractFiniteElementSpace{n}}

Returns the extraction matrix and basis indices for the rational finite element space.

# Arguments
- `rat_space::RationalFiniteElementSpace{n,F}`: The rational finite element space.
- `element_id::Int`: The index of the element.

# Returns
A tuple containing:
- The extraction matrix, which is an identity matrix.
- The basis indices.
"""
function get_extraction(rat_space::RationalFiniteElementSpace{n,F}, element_id::Int) where {n, F <: AbstractFiniteElementSpace{n}}
    # Get the basis indices for the underlying function space
    _, basis_indices = get_extraction(rat_space.function_space, element_id)
    n_supp = length(basis_indices)
    # Return the extraction matrix and basis indices
    return Matrix{Float64}(LinearAlgebra.I, n_supp, n_supp), basis_indices
end

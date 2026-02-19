
"""
    RationalFESpace{manifold_dim, F} <: AbstractFESpace{manifold_dim, 1, 1}

A rational finite element space obtained by dividing all elements of a given function space
by a fixed element from the same function space. The latter is defined with the help of
specified weights.

# Fields
- `function_space::F`: The underlying function space.
- `weights::Vector{Float64}`: The weights associated with the basis functions of the
    function space.
"""
struct RationalFESpace{manifold_dim, F} <: AbstractFESpace{manifold_dim, 1, 1}
    function_space::F
    weights::Vector{Float64}

    function RationalFESpace(
        function_space::F, weights::Vector{Float64}
    ) where {manifold_dim, F <: AbstractFESpace{manifold_dim, 1, 1}}
        if get_num_basis(function_space) != length(weights)
            error("Dimension mismatch")
        end

        return new{manifold_dim, F}(function_space, weights)
    end
end

function get_geometry(space::RationalFESpace)
    return get_geometry(space.function_space)
end

function get_parametric_geometry(space::RationalFESpace)
    return get_parametric_geometry(space.function_space)
end

function get_num_basis(space::RationalFESpace)
    return get_num_basis(space.function_space)
end
function get_num_basis(space::RationalFESpace, element_id::Int)
    return get_num_basis(space.function_space, element_id)
end

function get_basis_indices(space::RationalFESpace, element_id::Int)
    return get_basis_indices(space.function_space, element_id)
end

function get_num_elements(space::RationalFESpace)
    return get_num_elements(space.function_space)
end

"""
    get_polynomial_degree(space::RationalFESpace, element_id::Int)

Returns the polynomial degree (or the degree of the underlying function space) of the
rational finite element space for a specific element.

# Arguments
- `space::RationalFESpace`: The rational finite element space.
- `element_id::Int`: The index of the element.

# Returns
The polynomial degree (or the degree of the underlying function space) of the rational
finite element space for the specified element.
"""
function get_polynomial_degree(space::RationalFESpace, element_id::Int)
    return get_polynomial_degree(space.function_space, element_id)
end

function get_dof_partition(space::RationalFESpace)
    return get_dof_partition(space.function_space)
end

function evaluate(
    space::RationalFESpace{manifold_dim, F},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
) where {manifold_dim, F <: AbstractFESpace{manifold_dim, 1, 1}}
    # Evaluate the basis functions and their derivatives for the underlying function space
    homog_basis, basis_indices = evaluate(
        space.function_space, element_id, xi, nderivatives
    )
    n_eval = Points.get_num_points(xi)

    # Compute the rational basis functions and their derivatives
    for j in 0:nderivatives
        if j == 0
            # Compute the weight
            temp =
                homog_basis[1][1][1] * LinearAlgebra.Diagonal(space.weights[basis_indices])
            weight = reshape(sum(temp; dims=2), n_eval)
            # Rationalize with the weights
            homog_basis[1][1][1] .= LinearAlgebra.Diagonal(weight) \ temp
        elseif j == 1
            der_keys = integer_sums(j, manifold_dim)
            for key in der_keys
                # Get the location where the derivative is stored
                der_idx = get_derivative_idx(key)
                # Compute the weight and its derivative
                temp =
                    homog_basis[1][1][1] *
                    LinearAlgebra.Diagonal(space.weights[basis_indices])
                dtemp =
                    homog_basis[j + 1][der_idx][1] *
                    LinearAlgebra.Diagonal(space.weights[basis_indices])
                weight = reshape(sum(temp; dims=2), n_eval)
                dweight = reshape(sum(dtemp; dims=2), n_eval)
                # Compute the derivative of the rational basis functions
                homog_basis[j + 1][der_idx][1] .=
                    LinearAlgebra.Diagonal(weight) \ dtemp -
                    LinearAlgebra.Diagonal(dweight ./ weight .^ 2) * temp
            end
        elseif j > 1
            error("Derivatives of rational spaces of order $j not implemented")
        end
    end

    return homog_basis, basis_indices
end

function get_max_local_dim(space::RationalFESpace)
    return get_max_local_dim(space.function_space)
end

function get_local_basis(
    space::RationalFESpace{manifold_dim, F},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim, F <: AbstractFESpace{manifold_dim, 1, 1}}
    return evaluate(space, element_id, xi, nderivatives)[1]
end

function get_extraction(space::RationalFESpace, element_id::Int, component_id::Int=1)
    # Get the basis indices for the underlying function space
    basis_indices = get_basis_indices(space.function_space, element_id)

    return LinearAlgebra.I, 1:length(basis_indices)
end

function get_basis_permutation(space::RationalFESpace, element_id::Int, component_id::Int=1)
    return 1:get_num_basis(space, element_id)
end

############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ExteriorDerivative{manifold_dim, form_rank, expression_rank, G, F} <:
    AbstractForm{manifold_dim, form_rank, expression_rank, G}

Represents the exterior derivative of an `AbstractForm`.

# Fields
- `form::AbstractForm{manifold_dim, form_rank, expression_rank, G}`: The form to
    which the exterior derivative is applied.
- `label::String`: The exterior derivative label. This is a concatenation of `"d"` with the
    label of `form`.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: The form rank of the exterior derivative. If the form rank of `form` is `k`
    then `form_rank` is `k+1`.
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0,
    with one single set of basis forms have rank 1, with two sets of basis forms have rank
    2. Higher ranks are not possible.
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the underlying geometry.
- `F <: Forms.AbstractForm{manifold_dim, form_rank - 1, expression_rank, G}`: The
    type of `form`.

# Inner Constructors
- `ExteriorDerivative(form::F)`: General constructor.
"""
struct ExteriorDerivative{manifold_dim, form_rank, expression_rank, G, F} <:
       AbstractForm{manifold_dim, form_rank, expression_rank, G}
    form::F
    label::String

    function ExteriorDerivative(
        form::F
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
    }
        if form_rank == manifold_dim
            throw(ArgumentError("""\
                Tried to compute the exterior derivative of a volume form. The manifold \
                dimension is $(manifold_dim) and the form rank is $(form_rank). \
                """))
        end

        return new{manifold_dim, form_rank + 1, expression_rank, G, F}(
            form, "d(" * get_label(form) * ")"
        )
    end
end

"""
    d

Symbolic wrapper for the exterior derivative operator. See [`ExteriorDerivative`](@ref) for
the details.
"""
const d = ExteriorDerivative

############################################################################################
#                                         Getters                                          #
############################################################################################
"""
    get_form(ext_der::ExteriorDerivative)

Returns the form to which the exterior derivative is applied.

# Arguments
- `ext_der::ExteriorDerivative`: The exterior derivative.

# Returns
- `<:AbstractForm`: The form to which the exterior derivative is applied.
"""
get_form(ext_der::ExteriorDerivative) = ext_der.form

"""
    get_form_space_tree(ext_der::ExteriorDerivative)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the exterior derivative, e.g., for
`d((α ∧ β) + γ)`, it returns the spaces of `α`, `β`, and `γ`, if all have expression_rank > 1. \
If `α` has expression_rank = 0, it returns only the spaces of `β` and `γ`.

# Arguments
- `ext_der::ExteriorDerivative`: The exterior derivative structure.

# Returns
- `Tuple(<:AbstractForm)`: The list of spaces of forms present in the tree of the exterior derivative.
"""
function get_form_space_tree(ext_der::ExteriorDerivative)
    return get_form_space_tree(get_form(ext_der))
end

"""
    get_geometry(ext_der::ExteriorDerivative)

Returns the geometry of the form associated with the exterior derivative.

# Arguments
- `ext_der::ExteriorDerivative`: The exterior derivative.

# Returns
- `<:Geometry.AbstractGeometry`: The geometry of the form.
"""
get_geometry(ext_der::ExteriorDerivative) = get_geometry(get_form(ext_der))

"""
    evaluate(
        ext_der::ExteriorDerivative{manifold_dim},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Computes the exterior derivative at the element given by `element_id`, and canonical points
`xi`.

# Arguments
- `ext_der::ExteriorDerivative{manifold_dim}`: The exterior derivative structure.
- `element_id::Int`: The element identifier.
- `xi::Points.AbstractPoints{manifold_dim}`: The set of canonical points.

# Returns
- `::Vector{Array{Float64, expression_rank + 1}}`: The evaluated exterior derivative. The
    number of entries in the `Vector` is `binomial(manifold_dim, form_rank)`. The size
    of the `Array` is `(num_eval_points, num_basis)`, where `num_eval_points =
    Points.get_num_points(xi)` and `num_basis` is the number of basis functions used to represent
    the `form` on `element_id` ― for `expression_rank = 0` the inner `Array` is equivalent
    to a `Vector`.
"""
function evaluate(
    ext_der::ExteriorDerivative{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return _evaluate_exterior_derivative(get_form(ext_der), element_id, xi)
end

############################################################################################
#                                     Abstract method                                      #
############################################################################################

function _evaluate_exterior_derivative(
    form::AbstractForm{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    throw(ArgumentError("Method not implement for type $(typeof(form))."))
end

############################################################################################
#                                        Form Field                                        #
############################################################################################

function _evaluate_exterior_derivative(
    form::FormField{manifold_dim, form_rank, G, FS},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {
    manifold_dim,
    form_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    FS <: AbstractFormSpace{manifold_dim, form_rank, G},
}
    d_form_basis_eval, form_basis_indices = _evaluate_exterior_derivative(
        get_form_space(form), element_id, xi
    )

    # This is equal to binomial(manifold_dim, form_rank + 1).
    n_derivative_components = size(d_form_basis_eval, 1)

    d_form_eval = Vector{Vector{Float64}}(undef, n_derivative_components)

    for derivative_form_component_idx in 1:n_derivative_components
        d_form_eval[derivative_form_component_idx] =
            d_form_basis_eval[derivative_form_component_idx] *
            form.coefficients[form_basis_indices[1]]
    end

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow
    # multi-indexed expressions, like wedges.
    return d_form_eval, [[1]]
end

############################################################################################
#                                        Form Space                                        #
############################################################################################

function _evaluate_exterior_derivative(
    form_space::FormSpace{manifold_dim, 0, G},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
    # Preallocate memory for output array
    n_derivative_form_components = manifold_dim
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_id)
    n_evaluation_points = Points.get_num_points(xi)

    # We can avoid this if we change the output format of evaluation of directsum spaces
    # flip the second with the third index there...
    local_d_form_basis_eval = [
        zeros(Float64, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    # Evaluate derivatives
    d_local_fem_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, 1
    )

    # Store the required values
    key = zeros(Int, manifold_dim)
    for coordinate_idx in 1:manifold_dim
        key[coordinate_idx] = 1
        der_idx = FunctionSpaces.get_derivative_idx(key)
        key[coordinate_idx] = 0
        @. local_d_form_basis_eval[coordinate_idx] = d_local_fem_basis[2][der_idx][1]
    end

    return local_d_form_basis_eval, form_basis_indices
end

function _evaluate_exterior_derivative(
    form_space::FormSpace{2, 1, G}, element_id::Int, xi::Points.AbstractPoints{2}
) where {G <: Geometry.AbstractGeometry{2}}
    # manifold_dim = 2
    n_derivative_form_components = 1 # binomial(manifold_dim, 2)
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_id)
    n_evaluation_points = Points.get_num_points(xi)

    # Preallocate memory for output array
    local_d_form_basis_eval = [
        zeros(Float64, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    # Evaluate derivatives
    d_local_fem_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, 1
    )

    # The exterior derivative is
    # (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    # Store the required values
    der_idx_1 = FunctionSpaces.get_derivative_idx([1, 0])
    der_idx_2 = FunctionSpaces.get_derivative_idx([0, 1])
    @. local_d_form_basis_eval[1] =
        d_local_fem_basis[2][der_idx_1][2] - d_local_fem_basis[2][der_idx_2][1]

    return local_d_form_basis_eval, form_basis_indices
end

function _evaluate_exterior_derivative(
    form_space::FormSpace{3, 1, G}, element_id::Int, xi::Points.AbstractPoints{3}
) where {G <: Geometry.AbstractGeometry{3}}
    # manifold_dim = 3
    n_derivative_form_components = 3 # binomial(manifold_dim, 2)

    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_id)
    n_evaluation_points = Points.get_num_points(xi)

    # Preallocate memory for output array
    local_d_form_basis_eval = [
        zeros(Float64, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    # Evaluate the underlying FEM space and its first order derivatives (all derivatives for each component)
    d_local_fem_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, 1
    )

    # The exterior derivative is
    # (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃ + (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁ + (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    der_idx_1 = FunctionSpaces.get_derivative_idx([1, 0, 0])
    der_idx_2 = FunctionSpaces.get_derivative_idx([0, 1, 0])
    der_idx_3 = FunctionSpaces.get_derivative_idx([0, 0, 1])
    # First: (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃
    @. local_d_form_basis_eval[1] =
        d_local_fem_basis[2][der_idx_2][3] - d_local_fem_basis[2][der_idx_3][2]
    # Second: (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁
    @. local_d_form_basis_eval[2] =
        d_local_fem_basis[2][der_idx_3][1] - d_local_fem_basis[2][der_idx_1][3]
    # Third: (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    @. local_d_form_basis_eval[3] =
        d_local_fem_basis[2][der_idx_1][2] - d_local_fem_basis[2][der_idx_2][1]

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return local_d_form_basis_eval, form_basis_indices
end

function _evaluate_exterior_derivative(
    form_space::FormSpace{3, 2, G}, element_id::Int, xi::Points.AbstractPoints{3}
) where {G <: Geometry.AbstractGeometry{3}}
    # manifold_dim = 3
    n_derivative_form_components = 1 # binomial(manifold_dim, 2)

    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_id)
    n_evaluation_points = Points.get_num_points(xi)

    # Preallocate memory for output array
    local_d_form_basis_eval = [
        zeros(Float64, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    # Evaluate the underlying FEM space and its first order derivatives (all derivatives for each component)
    d_local_fem_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, 1
    )

    # The form is
    # α₁ dξ₂∧dξ₃ + α₂ dξ₃∧dξ₁ + α₃ dξ₁∧dξ₂
    # The exterior derivative is
    # (∂α₁/∂ξ₁ + ∂α₂/∂ξ₂ + ∂α₃/∂ξ₃) dξ₁∧dξ₂∧dξ₃
    der_idx_1 = FunctionSpaces.get_derivative_idx([1, 0, 0])
    der_idx_2 = FunctionSpaces.get_derivative_idx([0, 1, 0])
    der_idx_3 = FunctionSpaces.get_derivative_idx([0, 0, 1])
    @. local_d_form_basis_eval[1] =
        d_local_fem_basis[2][der_idx_1][1] +
        d_local_fem_basis[2][der_idx_2][2] +
        d_local_fem_basis[2][der_idx_3][3]

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return local_d_form_basis_eval, form_basis_indices
end

# #############################################################################################
# #                                           dd                                              #
# #############################################################################################
# function _evaluate_exterior_derivative(
#     dform::ExteriorDerivative{manifold_dim, form_rank_d, expression_rank, G, F},
#     element_id::Int,
#     xi::NTuple{manifold_dim, Vector{Float64}}
# ) where {
#     manifold_dim,
#     form_rank_d,
#     expression_rank,
#     form_rank,
#     G <: Geometry.AbstractGeometry{manifold_dim},
#     F <: AbstractFormField{manifold_dim, form_rank, G},
# }
#     # Determine shape of the output
#     n_evaluation_points = prod(size.(xi, 1))
#     n_derivative_components = binomial(manifold_dim, form_rank_d + 1)
#     n_basis_functions = 1

#     # Generate the zeros output array
#     ddform_eval = [
#         zeros(Float64, n_evaluation_points, n_basis_functions) for
#         _ in 1:n_derivative_components
#     ]

#     # We need to wrap form_basis_indices in [] to return a vector of vector to allow
#     # multi-indexed expressions, like wedges.
#     return ddform_eval, [[1]]
# end

#############################################################################################
#                                     Wedge product                                         #
#############################################################################################

function _evaluate_exterior_derivative(
    form::Wedge{manifold_dim, form_rank, expression_rank, G, F1, F2},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {
    manifold_dim,
    form_rank,
    form_rank_1,
    form_rank_2,
    expression_rank,
    expression_rank_1,
    expression_rank_2,
    G <: Geometry.AbstractGeometry{manifold_dim},
    F1 <: AbstractForm{manifold_dim, form_rank_1, expression_rank_1, G},
    F2 <: AbstractForm{manifold_dim, form_rank_2, expression_rank_2, G},
}
    # The exterior derivative of a wedge product follows the Leibniz rule:
    # d(αᵏ ∧ βᵐ) = dαᵏ ∧ βᵐ + (-1)^k αᵏ ∧ dβᵐ

    # Extract the forms that compose the wedge product and their exterior derivatives
    α = form.form_1
    β = form.form_2
    dα = ExteriorDerivative(α)
    dβ = ExteriorDerivative(β)

    # Compute the Leibniz expression components
    dα_wedge_β = Forms.Wedge(dα, β)
    α_wedge_dβ = Forms.Wedge(α, dβ)

    return evaluate(dα_wedge_β + (-1)^get_form_rank(α) * α_wedge_dβ, element_id, xi)
end

#############################################################################################
#                                Unary transformation                                     #
#############################################################################################
function _evaluate_exterior_derivative(
    form::UnaryFormTransformation{manifold_dim, form_rank, expression_rank, G, F, T},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    F <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
    T <: Function,
}
    # The exterior derivative of a binary transformation follows the law:
    # d(c*αᵏ) = c*dαᵏ

    # Extract the forms that compose the binary transformation and their exterior derivatives
    α = get_form(form)
    uni_transformation = get_transformation(form)

    # Evaluate the distributive expression components, sum them, and return
    return evaluate(uni_transformation(d(α)), element_id, xi)
end

#############################################################################################
#                                 Binary transformation                                     #
#############################################################################################
function _evaluate_exterior_derivative(
    form::BinaryFormTransformation{manifold_dim, form_rank, expression_rank, F1, F2, G, T},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    F1 <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
    F2 <: AbstractForm{manifold_dim, form_rank, expression_rank, G},
    T <: Function,
}
    # The exterior derivative of a binary transformation follows the distributive law:
    # d(αᵏ + βᵏ) = dαᵏ +  dβᵏ

    # Extract the forms
    forms = get_forms(form)

    # Extract the transformation
    binary_transformation = get_transformation(form)

    # Evaluate the distributive expression components, sum them, and return
    return evaluate(binary_transformation(d(forms[1]), d(forms[2])), element_id, xi)
end

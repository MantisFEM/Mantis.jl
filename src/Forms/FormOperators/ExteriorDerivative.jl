############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ExteriorDerivative{manifold_dim, form_rank, expression_rank, F} <:
    AbstractForm{manifold_dim, form_rank, expression_rank}

Represents the exterior derivative of an `AbstractForm`.

The `manifold_dim` and `expression_rank` are inherited from the form to which the exterior
derivative is applied. The `form_rank` of the exterior derivative is the form rank of the
input form plus one.

Formally, applying the exterior derivative to a volume form (the rank of the form equals
the dimension of the manifold) returns zero. However, in `Mantis`, the constructor throws
an error instead.

# Constructors
- `ExteriorDerivative(form::F)`: General constructor for any `AbstractForm`.

# Examples
Creating the exterior derivative of a ``0``-form:
```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 2), (1, 1));

julia> Λ⁰ₕ = Forms.FormSpace(0, B, "0-form");  # 0-form space with B as basis.

julia> dΛ⁰ₕ = d(Λ⁰ₕ);  # Note that dΛ⁰ₕ is a 1-form.

julia> isa(dΛ⁰ₕ, Forms.ExteriorDerivative{2, 1, 1})
true

```

# Fields
- `form::F`: The form to which the exterior derivative is applied. Note that the form rank
    of this form is one lower than the `form_rank` of the exterior derivative.
- `label::L`: The exterior derivative label. This is a concatenation of "d" with the label
    of `form`.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref) for the details.
- `F <: Forms.AbstractForm{manifold_dim, form_rank - 1, expression_rank}`: The type of
    `form`.
- `L <: AbstractString`: The type of the label. Since a "d" is added to the label, this
    type may differ from the label type of the underlying form.
"""
struct ExteriorDerivative{manifold_dim, form_rank, expression_rank, F, L} <:
       AbstractForm{manifold_dim, form_rank, expression_rank}
    form::F
    label::L

    function ExteriorDerivative(
        form::F
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank},
    }
        if form_rank == manifold_dim
            throw(ArgumentError("""\
                Tried to compute the exterior derivative of a volume form. The manifold \
                dimension is $(manifold_dim) and the form rank is $(form_rank). \
                """))
        end

        old_label = get_label(form)
        new_label = convert(typeof(old_label), "d(" * old_label * ")")

        return new{manifold_dim, form_rank + 1, expression_rank, F, typeof(new_label)}(
            form, new_label
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
#                                     Evaluate methods                                     #
############################################################################################

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
    return throw(MethodError(_evaluate_exterior_derivative, (form, element_id, xi)))
end

############################################################################################
#                                        Form Field                                        #
############################################################################################

function _evaluate_exterior_derivative(
    form::FormField{manifold_dim, form_rank, FS},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}
    d_form_basis_eval, form_basis_indices = _evaluate_exterior_derivative(
        get_form(form), element_id, xi
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

# The evaluations of the exterior derivative of a Form Space make use of the fact that the
# pullback and exterior derivative commute. This is only true when the Form Space uses the
# conventional (and default) FormPullback. This is why the pullback is enforced in the type.

function _evaluate_exterior_derivative(
    form_space::FormSpace{manifold_dim, 0, FormPullback},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    n_derivative_form_components = manifold_dim
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_id)
    n_evaluation_points = Points.get_num_points(xi)

    # Evaluate derivatives and values of the FE space.
    fe_space = get_fe_space(form_space)
    fe_evals, form_basis_indices = FunctionSpaces.evaluate(fe_space, element_id, xi, 1)

    pb = get_pullback(form_space)

    # Create a new array with the correct number of components.
    T = eltype(eltype(eltype(eltype(fe_evals))))
    form_evals = [
        zeros(T, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    # Store the required values
    for coordinate_idx in 1:manifold_dim
        key = ntuple(manifold_dim) do dim
            return dim == coordinate_idx ? 1 : 0
        end

        der_idx = FunctionSpaces.get_derivative_idx(key)
        pullback!(fe_evals[2][der_idx], pb, element_id, xi)
        for p in eachindex(form_evals[coordinate_idx], fe_evals[2][der_idx][1])
            form_evals[coordinate_idx][p] = fe_evals[2][der_idx][1][p]
        end
    end

    return form_evals, [form_basis_indices]
end

function _evaluate_exterior_derivative(
    form_space::FormSpace{2, 1, FormPullback}, element_id::Int, xi::Points.AbstractPoints{2}
)
    n_derivative_form_components = 1
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_id)
    n_evaluation_points = Points.get_num_points(xi)

    # Evaluate derivatives and values of the FE space.
    fe_space = get_fe_space(form_space)
    fe_evals, form_basis_indices = FunctionSpaces.evaluate(fe_space, element_id, xi, 1)

    pb = get_pullback(form_space)

    # Create a new array with the correct number of components.
    T = eltype(eltype(eltype(eltype(fe_evals))))
    form_evals = [
        zeros(T, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    # The exterior derivative is
    # (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    der_idx_1 = FunctionSpaces.get_derivative_idx((1, 0))
    der_idx_2 = FunctionSpaces.get_derivative_idx((0, 1))
    pullback!(fe_evals[2][der_idx_1], pb, element_id, xi)
    pullback!(fe_evals[2][der_idx_2], pb, element_id, xi)
    for p in eachindex(form_evals[1], fe_evals[2][der_idx_1][2], fe_evals[2][der_idx_2][1])
        form_evals[1][p] = fe_evals[2][der_idx_1][2][p] - fe_evals[2][der_idx_2][1][p]
    end

    return form_evals, [form_basis_indices]
end

function _evaluate_exterior_derivative(
    form_space::FormSpace{3, 1, FormPullback}, element_id::Int, xi::Points.AbstractPoints{3}
)
    n_derivative_form_components = 3
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_id)
    n_evaluation_points = Points.get_num_points(xi)

    # Evaluate derivatives and values of the FE space.
    fe_space = get_fe_space(form_space)
    fe_evals, form_basis_indices = FunctionSpaces.evaluate(fe_space, element_id, xi, 1)

    pb = get_pullback(form_space)

    # Create a new array with the correct number of components.
    T = eltype(eltype(eltype(eltype(fe_evals))))
    form_evals = [
        zeros(T, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    # The exterior derivative is
    # (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃ + (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁ + (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    der_idx_1 = FunctionSpaces.get_derivative_idx((1, 0, 0))
    der_idx_2 = FunctionSpaces.get_derivative_idx((0, 1, 0))
    der_idx_3 = FunctionSpaces.get_derivative_idx((0, 0, 1))
    pullback!(fe_evals[2][der_idx_1], pb, element_id, xi)
    pullback!(fe_evals[2][der_idx_2], pb, element_id, xi)
    pullback!(fe_evals[2][der_idx_3], pb, element_id, xi)
    # First: (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃
    for p in eachindex(form_evals[1], fe_evals[2][der_idx_2][3], fe_evals[2][der_idx_3][2])
        form_evals[1][p] = fe_evals[2][der_idx_2][3][p] - fe_evals[2][der_idx_3][2][p]
    end
    # Second: (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁
    for p in eachindex(form_evals[1], fe_evals[2][der_idx_3][1], fe_evals[2][der_idx_1][3])
        form_evals[1][p] = fe_evals[2][der_idx_3][1][p] - fe_evals[2][der_idx_1][3][p]
    end
    # Third: (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    for p in eachindex(form_evals[1], fe_evals[2][der_idx_1][2], fe_evals[2][der_idx_2][1])
        form_evals[1][p] = fe_evals[2][der_idx_1][2][p] - fe_evals[2][der_idx_2][1][p]
    end

    return form_evals, [form_basis_indices]
end

function _evaluate_exterior_derivative(
    form_space::FormSpace{3, 2, FormPullback}, element_id::Int, xi::Points.AbstractPoints{3}
)
    n_derivative_form_components = 1
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_id)
    n_evaluation_points = Points.get_num_points(xi)

    # Evaluate derivatives and values of the FE space.
    fe_space = get_fe_space(form_space)
    fe_evals, form_basis_indices = FunctionSpaces.evaluate(fe_space, element_id, xi, 1)

    pb = get_pullback(form_space)

    # Create a new array with the correct number of components.
    T = eltype(eltype(eltype(eltype(fe_evals))))
    form_evals = [
        zeros(T, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    # The form is
    # α₁ dξ₂∧dξ₃ + α₂ dξ₃∧dξ₁ + α₃ dξ₁∧dξ₂
    # The exterior derivative is
    # (∂α₁/∂ξ₁ + ∂α₂/∂ξ₂ + ∂α₃/∂ξ₃) dξ₁∧dξ₂∧dξ₃
    der_idx_1 = FunctionSpaces.get_derivative_idx((1, 0, 0))
    der_idx_2 = FunctionSpaces.get_derivative_idx((0, 1, 0))
    der_idx_3 = FunctionSpaces.get_derivative_idx((0, 0, 1))
    pullback!(fe_evals[2][der_idx_1], pb, element_id, xi)
    pullback!(fe_evals[2][der_idx_2], pb, element_id, xi)
    pullback!(fe_evals[2][der_idx_3], pb, element_id, xi)
    for p in eachindex(
        form_evals[1],
        fe_evals[2][der_idx_1][1],
        fe_evals[2][der_idx_2][2],
        fe_evals[2][der_idx_3][3],
    )
        form_evals[1][p] =
            fe_evals[2][der_idx_1][1][p] +
            fe_evals[2][der_idx_2][2][p] +
            fe_evals[2][der_idx_3][3][p]
    end

    return form_evals, [form_basis_indices]
end

############################################################################################
#                                  Constant Form Space                                     #
############################################################################################

function _evaluate_exterior_derivative(
    ::ConstantFormSpace{manifold_dim, 0, FormPullback}, ::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    # Preallocate memory for output array
    n_derivative_form_components = manifold_dim
    n_basis_functions = 1
    n_evaluation_points = Points.get_num_points(xi)
    local_d_form_basis_eval = [
        zeros(Float64, n_evaluation_points, n_basis_functions) for
        _ in 1:n_derivative_form_components
    ]

    return local_d_form_basis_eval, [[1]]
end

#############################################################################################
#                                     Wedge product                                         #
#############################################################################################

function _evaluate_exterior_derivative(
    form::Wedge{manifold_dim, form_rank, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank}
    # The exterior derivative of a wedge product follows the Leibniz rule:
    # d(αᵏ ∧ βᵐ) = dαᵏ ∧ βᵐ + (-1)^k αᵏ ∧ dβᵐ

    # Extract the forms that compose the wedge product and their exterior derivatives
    α, β = get_forms(form)
    dα = d(α)
    dβ = d(β)

    # Compute the Leibniz expression components
    dα_wedge_β = Forms.Wedge(dα, β)
    α_wedge_dβ = Forms.Wedge(α, dβ)

    return evaluate(dα_wedge_β + (-1)^get_form_rank(α) * α_wedge_dβ, element_id, xi)
end

#############################################################################################
#                                Unary transformation                                     #
#############################################################################################
function _evaluate_exterior_derivative(
    form::UnaryFormTransformation{manifold_dim, form_rank, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank}
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
    form::BinaryFormTransformation{manifold_dim, form_rank, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank}
    # The exterior derivative of a binary transformation follows the distributive law:
    # d(αᵏ + βᵏ) = dαᵏ +  dβᵏ

    # Extract the forms
    forms = get_forms(form)

    # Extract the transformation
    binary_transformation = get_transformation(form)

    # Evaluate the distributive expression components, sum them, and return
    return evaluate(binary_transformation(d(first(forms)), d(last(forms))), element_id, xi)
end

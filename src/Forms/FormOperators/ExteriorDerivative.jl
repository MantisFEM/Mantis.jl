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
            form.coefficients[form_basis_indices[derivative_form_component_idx][1]]
    end

    return d_form_eval, _field_indices(n_derivative_components)
end

############################################################################################
#                                        Form Space                                        #
############################################################################################

"""
    _first_derivative_idx(::Val{manifold_dim}, direction::Int) where {manifold_dim}

Returns the index of the first derivative in `direction` within the derivative evaluations
of a `manifold_dim`-variate function space; see `FunctionSpaces.get_derivative_idx`.
"""
function _first_derivative_idx(::Val{manifold_dim}, direction::Int) where {manifold_dim}
    key = ntuple(dim -> Int(dim == direction), Val(manifold_dim))

    return FunctionSpaces.get_derivative_idx(key)
end

# Single-component

function _evaluate_exterior_derivative(
    form_space::FormSpace{manifold_dim, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    d_local_fem_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, 1
    )

    # All components of the gradient are spanned by the same (single-component) basis.
    d_eval = Vector{Matrix{Float64}}(undef, manifold_dim)
    for direction in 1:manifold_dim
        der_idx = _first_derivative_idx(Val(manifold_dim), direction)
        d_eval[direction] = d_local_fem_basis[2][der_idx][1]
    end

    d_indices = [copy(form_basis_indices[1]) for _ in 1:manifold_dim]

    return d_eval, d_indices
end

# Multi-component

"""
    _d_terms(::Val{manifold_dim}, ::Val{form_rank}) where {manifold_dim, form_rank}

Returns, for each component of the exterior derivative of a `form_rank`-form with multiple
components, the terms `(component, direction, sign)` such that that component of dα is

`∑ sign * ∂α_component/∂ξ_direction`.

The supported combinations are 1-forms in 2D and 3D, and 2-forms in 3D:
```
d(α₁dξ₁ + α₂dξ₂) = (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂

d(α₁dξ₁ + α₂dξ₂ + α₃dξ₃) =
    (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃ + (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁ + (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂

d(α₁dξ₂∧dξ₃ + α₂dξ₃∧dξ₁ + α₃dξ₁∧dξ₂) = (∂α₁/∂ξ₁ + ∂α₂/∂ξ₂ + ∂α₃/∂ξ₃) dξ₁∧dξ₂∧dξ₃
```
Other combinations throw an `ArgumentError`.

# Returns
- `::NTuple{binomial(manifold_dim, form_rank + 1), NTuple{N, Tuple{Int, Int, Float64}}}`: The
    terms per component of dα.
"""
_d_terms(::Val{manifold_dim}, ::Val{form_rank}) where {manifold_dim, form_rank} = throw(
    ArgumentError(
        LazyString(
            "Combination manifold_dim=",
            manifold_dim,
            " and form_rank=",
            form_rank,
            " not supported",
        ),
    ),
)
# 1-forms in 2D.
_d_terms(::Val{2}, ::Val{1}) = (((2, 1, 1.0), (1, 2, -1.0)),)
# 1-forms in 3D.
_d_terms(::Val{3}, ::Val{1}) =
    (((3, 2, 1.0), (2, 3, -1.0)), ((1, 3, 1.0), (3, 1, -1.0)), ((2, 1, 1.0), (1, 2, -1.0)))
# 2-forms in 3D.
_d_terms(::Val{3}, ::Val{2}) = (((1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)),)

function _evaluate_exterior_derivative(
    form_space::FormSpace{manifold_dim, form_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank}
    d_local_fem_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_id, xi, 1
    )
    terms = _d_terms(Val(manifold_dim), Val(form_rank))
    num_d_components = length(terms)
    d_eval = Vector{Matrix{Float64}}(undef, num_d_components)
    d_indices = Vector{Vector{Vector{Int}}}(undef, num_d_components)
    for (d_component, d_component_terms) in enumerate(terms)
        components = [c for (c, _, _) in d_component_terms]
        combined_indices, column_maps = _combine_indices(
            form_basis_indices, components, Val(1)
        )
        d_eval[d_component] = _allocate_indices(get_num_points(xi), combined_indices)
        for (c, direction, sign) in d_component_terms
            der_idx = _first_derivative_idx(Val(manifold_dim), direction)
            _add_columns!(
                d_eval[d_component], d_local_fem_basis[2][der_idx][c], column_maps[c], sign
            )
        end
        d_indices[d_component] = _tuple_to_vec_indices(combined_indices)
    end

    return d_eval, d_indices
end

############################################################################################
#                                  Constant Form Space                                     #
############################################################################################

function _evaluate_exterior_derivative(
    ::ConstantFormSpace{manifold_dim, 0}, ::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    n_evaluation_points = Points.get_num_points(xi)
    local_d_form_basis_eval = [
        zeros(Float64, n_evaluation_points, 1) for _ in 1:manifold_dim
    ]

    return local_d_form_basis_eval, [[[1]] for _ in 1:manifold_dim]
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
    α = form.form_1
    β = form.form_2
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

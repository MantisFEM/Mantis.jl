############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    Hodge{manifold_dim, form_rank, expression_rank, F} <:
    AbstractForm{manifold_dim, form_rank, expression_rank}

Represents the hodge star of an `AbstractForm`.

The `manifold_dim` of the `Hodge` is inherited from the input form, while the `form_rank`
is the `manifold_dim` minus the form rank of the input form.

# Inner Constructors
- `Hodge(form::F)`: General constructor.

# Examples
```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 2), (1, 1));

julia> Λ⁰ₕ = Forms.FormSpace(0, B, "0-form");  # 0-form space with B as basis.

julia> Λ²ₕ = Forms.FormSpace(2, B, "2-form");  # 2-form space with B as basis.

julia> starΛ⁰ₕ = ★(Λ⁰ₕ);

julia> isa(starΛ⁰ₕ, Forms.Hodge{2, 2, 1})
true

julia> ★Λ²ₕ = ★(Λ²ₕ);

julia> isa(★Λ²ₕ, Forms.Hodge{2, 0, 1})
true

```

# Fields
- `form::F`: The form to which the hodge star is applied.
- `label::L`: The hodge star label. This is a concatenation of "★" with the label of `form`.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [AbstractForm](@ref) for the details.
- `F <: Forms.AbstractForm{manifold_dim, manifold_dim-form_rank, expression_rank}`: The
    type of `form`.
- `L <: AbstractString`: The type of the label. Since a "★" is added to the label, this
    type may differ from the label type of the underlying form.
"""
struct Hodge{manifold_dim, form_rank, expression_rank, F, L} <:
       AbstractForm{manifold_dim, form_rank, expression_rank}
    form::F
    label::L

    function Hodge(
        form::F
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank},
    }
        if expression_rank > 1
            msg_1 = "Hodge-star only valid for expressions with expression rank < 2. "
            msg_2 = "Expression rank $(expression_rank) was given."
            throw(ArgumentError(msg_1 * msg_2))
        end
        hodge_rank = manifold_dim - form_rank

        old_label = get_label(form)
        new_label = convert(
            typeof(old_label), LaTeXStrings.L"\star" * "(" * old_label * ")"
        )

        return new{manifold_dim, hodge_rank, expression_rank, F, typeof(new_label)}(
            form, new_label
        )
    end
end

"""
    ★

Symbolic wrapper for the hodge star operator. The unicode character command is `\\bigstar`.
See [`Hodge`](@ref) for the details.
"""
const ★ = Hodge

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    hodge::Hodge{manifold_dim}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    return _evaluate_hodge(get_form(hodge), element_id, xi)
end

############################################################################################
#                                     Abstract method                                      #
############################################################################################

function _evaluate_hodge(
    form::AbstractForm{manifold_dim, form_rank, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank}
    throw(ArgumentError("Method not implement for type $(typeof(form))."))
end

############################################################################################
#                                       Combinations                                       #
############################################################################################

# 0-forms (manifold_dim)
function _evaluate_hodge(
    form_expression::AbstractForm{manifold_dim, 0, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank}
    _, sqrt_g = Geometry.metric(get_geometry(form_expression), element_id, xi)
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # We restrict the evaluation of hodge-star to expression of rank 1 and lower, therefore
    # we can select just the first index of the vector of indices, for higher rank
    # expression we will need to change this
    mat_size = (size(form_eval[1], 1), size(form_eval[1], 2))
    hodge_eval = [Matrix{Float64}(undef, mat_size)]
    # ⋆α₁⁰ = α₁⁰√det(gᵢⱼ) dξ₁∧…∧dξₙ.
    for id in CartesianIndices(mat_size)
        hodge_eval[1][id] = form_eval[1][id] * sqrt_g[id[1]]
    end

    return hodge_eval, form_indices
end

# n-forms (manifold_dim)
function _evaluate_hodge(
    form_expression::AbstractForm{manifold_dim, manifold_dim, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank}
    _, sqrt_g = Geometry.metric(get_geometry(form_expression), element_id, xi)
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # Because we restrict ourselves to expression with rank 0 or 1, we can
    # extract the first set of indices (only set), for higher ranks, we need
    # to change this
    mat_size = (size(form_eval[1], 1), size(form_eval[1], 2))
    hodge_eval = [Matrix{Float64}(undef, mat_size)]
    # ⋆α₁ⁿdξ₁∧…∧dξₙ = α₁ⁿ(√det(gᵢⱼ))⁻¹.
    for id in CartesianIndices(mat_size)
        hodge_eval[1][id] = form_eval[1][id] * (1 / sqrt_g[id[1]])
    end

    return hodge_eval, form_indices
end

# 1-forms (2 dimensions)
function _evaluate_hodge(
    form_expression::AbstractForm{2, 1, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{2},
) where {expression_rank}
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression), element_id, xi)
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # Because we restrict ourselves to expression with rank 0 or 1, we can
    # extract the first set of indices (only set), for higher ranks, we need
    # to change this
    mat_size = (size(form_eval[1], 1), size(form_eval[1], 2))
    hodge_eval = [zeros(mat_size) for _ in 1:2]
    # ⋆(α₁¹dξ₁+α₂¹dξ₂) = [-(α₁¹g²¹+α₂¹g²²)dξ₁ + (α₁¹g¹¹+α₂¹g¹²)dξ₂]√det(gᵢⱼ).
    for id in CartesianIndices(mat_size)
        for component in 1:2
            # First: -(α₁¹g²¹+α₂¹g²²)dξ₁
            hodge_eval[1][id] += -form_eval[component][id] * inv_g[id[1]][2, component]
            # Second: (α₁¹g¹¹+α₂¹g¹²)dξ₂
            hodge_eval[2][id] += form_eval[component][id] * inv_g[id[1]][1, component]
        end

        hodge_eval[1][id] *= sqrt_g[id[1]]
        hodge_eval[2][id] *= sqrt_g[id[1]]
    end

    return hodge_eval, form_indices
end

# 1-forms (3 dimensions)
function _evaluate_hodge(
    form_expression::AbstractForm{3, 1, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank}
    # Compute the metric terms
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression), element_id, xi)
    # Evaluate the form expression to which we wish to apply the Hodge-⋆
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # Preallocate memory for the Hodge evaluation matrix
    # Because we restrict ourselves to expression with rank 0 or 1, we can
    # extract the first set of indices (only set), for higher ranks, we need
    # to change this
    mat_size = (size(form_eval[1], 1), size(form_eval[1], 2))
    hodge_eval = [zeros(mat_size) for _ in 1:3]
    # Compute the Hodge-⋆ following the analytical expression
    # ⋆(α₁¹dξ₁+α₂¹dξ₂+α₃¹dξ₃) = [(α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃ + (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁ + (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂]√det(gᵢⱼ).
    # First: (α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃  --> component 1
    # Second: (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁ --> component 2
    # Third: (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂  --> component 3
    for id in CartesianIndices(mat_size)
        for h_component in 1:3
            for x_component in 1:3
                hodge_eval[h_component][id] +=
                    form_eval[x_component][id] * inv_g[id[1]][h_component, x_component]
            end

            hodge_eval[h_component][id] *= sqrt_g[id[1]]
        end
    end

    return hodge_eval, form_indices
end

# 2-forms (3 dimensions)
function _evaluate_hodge(
    form_expression::AbstractForm{3, 2, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank}
    # The Hodge-⋆ of a 2-form in 3D is the inverse of the Hodge-⋆ of a 1-form in 3D
    # Therefore we can use the metric tensor instead of the inverse metric tensor
    # and use the same expression as for the Hodge-⋆ of 1-forms but now using the
    # metric tensor instead of the inverse of the metric tensor.
    # Compute the metric terms
    g, sqrt_g = Geometry.metric(get_geometry(form_expression), element_id, xi)
    # Evaluate the form expression to which we wish to apply the Hodge-⋆
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # Preallocate memory for the Hodge evaluation matrix
    mat_size = (size(form_eval[1], 1), size(form_eval[1], 2))
    hodge_eval = [zeros(mat_size) for _ in 1:3]
    # Compute the Hodge-⋆ following the analytical expression
    # ⋆(α₁dξ²∧dξ³+α₂dξ³∧dξ¹+α₃dξ¹∧dξ²) = [(α₁g₁₁+α₂¹g₁₂+α₃¹g₁₃)dξ₁ + (α₁g₂₁+α₂g₂₂+α₃g₂₃)dξ₂ + (α₁g₃₁+α₂g₃₂+α₃g₃₃)dξ₃]/√det(gᵢⱼ).
    # First: (α₁g₁₁+α₂¹g₁₂+α₃¹g₁₃)dξ₁  --> component 1
    # Second: (α₁g₂₁+α₂g₂₂+α₃g₂₃)dξ₂   --> component 2
    # Third: (α₁g₃₁+α₂g₃₂+α₃g₃₃)dξ₃    --> component 3
    for id in CartesianIndices(mat_size)
        for h_component in 1:3
            for x_component in 1:3
                hodge_eval[h_component][id] +=
                    form_eval[x_component][id] * g[id[1]][h_component, x_component]
            end

            hodge_eval[h_component][id] /= sqrt_g[id[1]]
        end
    end

    return hodge_eval, form_indices
end

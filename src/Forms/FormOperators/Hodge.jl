############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    Hodge{manifold_dim, form_rank, expression_rank, G, F} <:
    AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}

Represents the hodge star of an `AbstractFormExpression`.

# Fields
- `form::AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}`: The form to
    which the hodge star is applied.
- `label::String`: The hodge star label. This is a concatenation of `"★"` with the
    label of `form`.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: The form rank of the hodge star. If the form rank of `form` is `k`
    then `form_rank` is `manifold_dim - k`.
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0,
    with one single set of basis forms have rank 1, with two sets of basis forms have rank
    2. Higher ranks are not possible.
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the underlying geometry.
- `F <: Forms.AbstractFormExpression{manifold_dim, manifold_dim-form_rank, expression_rank,
    G}`: The type of `form`.

# Inner Constructors
- `Hodge(form::F)`: General constructor.
"""
struct Hodge{manifold_dim, form_rank, expression_rank, G, F} <:
       AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
    form::F
    label::String

    function Hodge(
        form::F
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    }
        if expression_rank > 1
            msg_1 = "Hodge-star only valid for expressions with expression rank < 2. "
            msg_2 = "Expression rank $(expression_rank) was given."
            throw(ArgumentError(msg_1 * msg_2))
        end
        hodge_rank = manifold_dim - form_rank

        return new{manifold_dim, hodge_rank, expression_rank, G, F}(
            form, "★(" * get_label(form) * ")"
        )
    end
end

"""
    ★(form::AbstractFormExpression)

Symbolic wrapper for the hodge star operator. The unicode character command is `\\bigstar`.

# Arguments
- `form::AbstractFormExpression`: The form to which the hodge star is applied.

# Returns
- `Hodge`: The hodge operator.
"""
function ★(form::AbstractFormExpression)
    return Hodge(form)
end

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_form(form_expression::Hodge)

Returns the form to which the hodge star is applied.

# Arguments
- `form_expression::Hodge`: The hodge star structure.

# Returns
- `<:AbstractFormExpression`: The form to which the hodge star is applied.
"""
get_form(form_expression::Hodge) = form_expression.form

"""
    get_form_space_tree(hodge::Hodge)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the Hodge, e.g., for
`★((α ∧ β) + γ)`, it returns the spaces of `α`, `β`, and `γ`, if all have exprssion_rank > 1. 
If `α` has expression_rank = 0, it returns only the spaces of `β` and `γ`.

# Arguments
- `hodge::Hodge`: The Hodge structure.

# Returns
- `Tuple(<:AbstractFormExpression)`: The list of spaces forms present in the tree of the Hodge.
"""
function get_form_space_tree(hodge::Hodge)
    return get_form_space_tree(hodge.form)
end


"""
    get_geometry(form_expression::Hodge)

Returns the geometry of form expression used in the hodge star.

# Arguments
- `form_expression::Hodge`: The hodge star structure.

# Returns
- `<:Geometry.AbstractGeometry`: The geometry of the form expression.
"""
get_geometry(form_expression::Hodge) = get_geometry(get_form(form_expression))

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

"""
    evaluate(
        hodge::Hodge{manifold_dim},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Computes the hodge star at the element given by `element_id`, and canonical points `xi`.

# Arguments
- `hodge::Hodge{manifold_dim}`: The hodge star structure.
- `element_id::Int`: The element identifier.
- `xi::NTuple{manifold_dim, Vector{Float64}`: The set of canonical points.

# Returns
- `::Vector{Array{Float64, expression_rank + 1}}`: The evaluated hodge star. The number of
    entries in the `Vector` is `binomial(manifold_dim, manifold_dim - form_rank)`. The size
    of the `Array` is `(num_eval_points, num_basis)`, where `num_eval_points =
    Points.get_num_points(xi)` and `num_basis` is the number of basis functions used to represent
    the `form` on `element_id` ― for `expression_rank = 0` the inner `Array` is equivalent
    to a `Vector`.
"""
function evaluate(
    hodge::Hodge{manifold_dim}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    return _evaluate_hodge(get_form(hodge), element_id, xi)
end

############################################################################################
#                                     Abstract method                                      #
############################################################################################

function _evaluate_hodge(
    form::AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry}
    throw(ArgumentError("Method not implement for type $(typeof(form))."))
end

############################################################################################
#                                       Combinations                                       #
############################################################################################

# 0-forms (manifold_dim)
function _evaluate_hodge(
    form_expression::AbstractFormExpression{manifold_dim, 0, expression_rank, G},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
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
    form_expression::AbstractFormExpression{manifold_dim, manifold_dim, expression_rank, G},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
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
    form_expression::AbstractFormExpression{2, 1, expression_rank, G},
    element_id::Int,
    xi::Points.AbstractPoints{2},
) where {expression_rank, G <: Geometry.AbstractGeometry{2}}
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
            hodge_eval[1][id] += -form_eval[component][id] * inv_g[id[1], 2, component]
            # Second: (α₁¹g¹¹+α₂¹g¹²)dξ₂
            hodge_eval[2][id] += form_eval[component][id] * inv_g[id[1], 1, component]
        end

        hodge_eval[1][id] *= sqrt_g[id[1]]
        hodge_eval[2][id] *= sqrt_g[id[1]]
    end

    return hodge_eval, form_indices
end

# 1-forms (3 dimensions)
function _evaluate_hodge(
    form_expression::AbstractFormExpression{3, 1, expression_rank, G},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank, G <: Geometry.AbstractGeometry{3}}
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
                    form_eval[x_component][id] * inv_g[id[1], h_component, x_component]
            end

            hodge_eval[h_component][id] *= sqrt_g[id[1]]
        end
    end

    return hodge_eval, form_indices
end

# 2-forms (3 dimensions)
function _evaluate_hodge(
    form_expression::AbstractFormExpression{3, 2, expression_rank, G},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank, G <: Geometry.AbstractGeometry{3}}
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
                    form_eval[x_component][id] * g[id[1], h_component, x_component]
            end

            hodge_eval[h_component][id] /= sqrt_g[id[1]]
        end
    end

    return hodge_eval, form_indices
end

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
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref) for the details.
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
    return throw(MethodError(_evaluate_hodge, (form, element_id, xi)))
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
    # ⋆α₁⁰ = α₁⁰√det(gᵢⱼ) dξ₁∧…∧dξₙ.
    hodge_eval = [form_eval[1] .* sqrt_g]

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
    # ⋆α₁ⁿdξ₁∧…∧dξₙ = α₁ⁿ(√det(gᵢⱼ))⁻¹.
    hodge_eval = [form_eval[1] ./ sqrt_g]

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
    # ⋆(α₁¹dξ₁+α₂¹dξ₂) = [-(α₁¹g²¹+α₂¹g²²)dξ₁ + (α₁¹g¹¹+α₂¹g¹²)dξ₂]√det(gᵢⱼ).
    # So component 1 uses row 2 of g⁻¹ with a minus sign, and component 2 uses row 1.
    hodge_rows = (2, 1)
    hodge_signs = (-1.0, 1.0)
    hodge_eval = Vector{Array{Float64, 1 + expression_rank}}(undef, 2)
    hodge_indices = Vector{Vector{Vector{Int}}}(undef, 2)
    for h_component in 1:2
        weights = _hodge_weights(
            inv_g, sqrt_g, hodge_rows[h_component], hodge_signs[h_component], 2
        )
        hodge_eval[h_component], hodge_indices[h_component] = _mix_components(
            form_eval, form_indices, 1:2, weights, Val(expression_rank)
        )
    end

    return hodge_eval, hodge_indices
end

# 1-forms (3 dimensions)
function _evaluate_hodge(
    form_expression::AbstractForm{3, 1, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank}
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression), element_id, xi)
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # ⋆(α₁¹dξ₁+α₂¹dξ₂+α₃¹dξ₃) = [(α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃ + (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁ + (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂]√det(gᵢⱼ).
    hodge_eval = Vector{Array{Float64, 1 + expression_rank}}(undef, 3)
    hodge_indices = Vector{Vector{Vector{Int}}}(undef, 3)
    for h_component in 1:3
        weights = _hodge_weights(inv_g, sqrt_g, h_component, 1.0, 3)
        hodge_eval[h_component], hodge_indices[h_component] = _mix_components(
            form_eval, form_indices, 1:3, weights, Val(expression_rank)
        )
    end

    return hodge_eval, hodge_indices
end

# 2-forms (3 dimensions)
function _evaluate_hodge(
    form_expression::AbstractForm{3, 2, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank}
    # The Hodge-⋆ of a 2-form in 3D is the inverse of the Hodge-⋆ of a 1-form in 3D.
    # Therefore we can use the same expression as for the Hodge-⋆ of 1-forms but now using
    # the metric tensor instead of the inverse of the metric tensor.
    g, sqrt_g = Geometry.metric(get_geometry(form_expression), element_id, xi)
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # ⋆(α₁dξ²∧dξ³+α₂dξ³∧dξ¹+α₃dξ¹∧dξ²) = [(α₁g₁₁+α₂¹g₁₂+α₃¹g₁₃)dξ₁ + (α₁g₂₁+α₂g₂₂+α₃g₂₃)dξ₂ + (α₁g₃₁+α₂g₃₂+α₃g₃₃)dξ₃]/√det(gᵢⱼ).
    hodge_eval = Vector{Array{Float64, 1 + expression_rank}}(undef, 3)
    hodge_indices = Vector{Vector{Vector{Int}}}(undef, 3)
    inv_sqrt_g = 1.0 ./ sqrt_g
    for h_component in 1:3
        weights = _hodge_weights(g, inv_sqrt_g, h_component, 1.0, 3)
        hodge_eval[h_component], hodge_indices[h_component] = _mix_components(
            form_eval, form_indices, 1:3, weights, Val(expression_rank)
        )
    end

    return hodge_eval, hodge_indices
end

"""
    _hodge_weights(
        metric_tensor::AbstractVector,
        scaling::AbstractVector,
        row::Int,
        sign::Float64,
        num_components::Int,
    )

Computes, for each component `x` of the input form, the point-wise weight
`sign * metric_tensor[p][row, x] * scaling[p]` with which it contributes to a component of
its Hodge-⋆.

# Returns
- `::Vector{Vector{Float64}}`: The weights per input component and per evaluation point.
"""
function _hodge_weights(
    metric_tensor::AbstractVector,
    scaling::AbstractVector,
    row::Int,
    sign::Float64,
    num_components::Int,
)
    weights = [Vector{Float64}(undef, length(scaling)) for _ in 1:num_components]
    for x_component in 1:num_components
        for point_id in eachindex(scaling)
            weights[x_component][point_id] =
                sign * metric_tensor[point_id][row, x_component] * scaling[point_id]
        end
    end

    return weights
end

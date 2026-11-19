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
#                                       Combinations                                       #
############################################################################################

# Single-component

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

# Multi-component

"""
    _hodge_rows(::Val{manifold_dim}, ::Val{form_rank}) where {manifold_dim, form_rank}

Returns, for each component `i` of the Hodge-⋆ of a `form_rank`-form with multiple
components, the row of the metric term `M` used by that component. Together with
[`_hodge_signs`](@ref) and [`_hodge_metric_terms`](@ref), component `i` of ⋆α is

`signs[i] * ∑ⱼ M[rows[i], j] αⱼ * s`.

The supported combinations are 1-forms in 2D and 3D, and 2-forms in 3D:
```
⋆(α₁dξ₁ + α₂dξ₂) = [-(α₁g²¹ + α₂g²²)dξ₁ + (α₁g¹¹ + α₂g¹²)dξ₂]√det(gᵢⱼ)

⋆(α₁dξ₁ + α₂dξ₂ + α₃dξ₃) =
    [(α₁g¹¹ + α₂g¹² + α₃g¹³)dξ₂∧dξ₃ + (α₁g²¹ + α₂g²² + α₃g²³)dξ₃∧dξ₁
     + (α₁g³¹ + α₂g³² + α₃g³³)dξ₁∧dξ₂]√det(gᵢⱼ)

⋆(α₁dξ₂∧dξ₃ + α₂dξ₃∧dξ₁ + α₃dξ₁∧dξ₂) =
    [(α₁g₁₁ + α₂g₁₂ + α₃g₁₃)dξ₁ + (α₁g₂₁ + α₂g₂₂ + α₃g₂₃)dξ₂
     + (α₁g₃₁ + α₂g₃₂ + α₃g₃₃)dξ₃]/√det(gᵢⱼ)
```
The last one is the inverse of the 3D 1-form case. Other combinations throw an
`ArgumentError`.

# Returns
- `::NTuple{binomial(manifold_dim, form_rank), Int}`: The row per component of ⋆α.
"""
_hodge_rows(::Val{manifold_dim}, ::Val{form_rank}) where {manifold_dim, form_rank} = throw(
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

"""
    _hodge_signs(::Val{manifold_dim}, ::Val{form_rank}) where {manifold_dim, form_rank}

Returns the sign of each component of the Hodge-⋆ of a `form_rank`-form with multiple
components. See [`_hodge_rows`](@ref) for the details.

# Returns
- `::NTuple{binomial(manifold_dim, form_rank), Float64}`: The sign per component of ⋆α.
"""
function _hodge_signs end

# 1-forms in 2D.
_hodge_rows(::Val{2}, ::Val{1}) = (2, 1)
_hodge_signs(::Val{2}, ::Val{1}) = (-1.0, 1.0)
# 1- and 2-forms in 3D.
_hodge_rows(::Val{3}, ::Union{Val{1}, Val{2}}) = (1, 2, 3)
_hodge_signs(::Val{3}, ::Union{Val{1}, Val{2}}) = (1.0, 1.0, 1.0)

"""
    _hodge_metric_terms(
        geometry::Geometry.AbstractGeometry, ::Val{form_rank}, element_id, xi
    ) where {form_rank}

Returns the metric terms with which the components of a `form_rank`-form are combined into
the components of its Hodge-⋆; see [`_hodge_rows`](@ref). These are
- 1-forms (2D and 3D): `M = g⁻¹` and `s = √det(gᵢⱼ)`.
- 2-forms in 3D: `M = g` and `s = 1/√det(gᵢⱼ)`.

# Returns
- `M::Vector{<:AbstractMatrix}`: The (inverse) metric tensor per evaluation point.
- `s::Vector{Float64}`: The scaling per evaluation point.
"""
function _hodge_metric_terms(
    geometry::Union{Geometry.AbstractGeometry{2}, Geometry.AbstractGeometry{3}},
    ::Val{1},
    element_id,
    xi,
)
    inv_g, _, sqrt_g = Geometry.inv_metric(geometry, element_id, xi)

    return inv_g, sqrt_g
end

function _hodge_metric_terms(
    geometry::Geometry.AbstractGeometry{3}, ::Val{2}, element_id, xi
)
    g, sqrt_g = Geometry.metric(geometry, element_id, xi)

    return g, 1 ./ sqrt_g
end

function _evaluate_hodge(
    form_expression::AbstractForm{manifold_dim, form_rank, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank}
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    geometry = get_geometry(form_expression)
    rows = _hodge_rows(Val(manifold_dim), Val(form_rank))
    signs = _hodge_signs(Val(manifold_dim), Val(form_rank))
    component_term, scaling_term = _hodge_metric_terms(
        geometry, Val(form_rank), element_id, xi
    )
    num_components = binomial(manifold_dim, form_rank)
    hodge_eval = Vector{Array{Float64, 1 + expression_rank}}(undef, num_components)
    hodge_indices = Vector{Vector{Vector{Int}}}(undef, num_components)
    combined_indices, column_maps = _combine_indices(
        form_indices, 1:num_components, Val(expression_rank)
    )
    for component_id in 1:num_components
        weights = _hodge_weights(
            component_term,
            scaling_term,
            rows[component_id],
            signs[component_id],
            num_components,
        )
        hodge_eval[component_id] = _allocate_indices(get_num_points(xi), combined_indices)
        for c in 1:num_components
            _add_columns!(
                hodge_eval[component_id], form_eval[c], column_maps[c], weights[c]
            )
        end
        hodge_indices[component_id] = _tuple_to_vec_indices(combined_indices)
    end

    return hodge_eval, hodge_indices
end

"""
    _hodge_weights(
        component_term::AbstractVector,
        scaling_term::AbstractVector,
        row::Int,
        sign::Float64,
        num_components::Int,
    )

Computes, for each component `c` of the input form, the point-wise weight
`sign * component_term[p][row, c] * scaling_term[p]` with which it contributes to a component
of its Hodge-⋆. The `component_term` and `scaling_term` are given by
[`_hodge_metric_terms`](@ref), and the `row` and `sign` by [`_hodge_rows`](@ref) and
[`_hodge_signs`](@ref).

# Returns
- `::Vector{Vector{Float64}}`: The weights per input component and per evaluation point.
"""
function _hodge_weights(
    component_term::AbstractVector,
    scaling_term::AbstractVector,
    row::Int,
    sign::Float64,
    num_components::Int,
)
    weights = [Vector{Float64}(undef, length(scaling_term)) for _ in 1:num_components]
    for c in 1:num_components, point_id in eachindex(scaling_term)
        weights[c][point_id] =
            sign * component_term[point_id][row, c] * scaling_term[point_id]
    end

    return weights
end

############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    Wedge{manifold_dim, form_rank, expression_rank, F1, F2, L} <:
    AbstractForm{manifold_dim, form_rank, expression_rank}

Represents the wedge between two differential forms.

The `manifold_dim` of the `Wedge` is inherited from the input forms (which have the same
`manifold_dim`). The `form_rank` and `expression_rank` of the `Wedge` are the sums of the
respective ranks of the input forms. If the `expression_rank` of the `Wedge` would become
larger than 2, an error is thrown.

# Constructors
- `Wedge(form_1::F1, form_2::F2)`: General constructor.

# Examples
```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 2), (1, 1));

julia> Λ⁰ₕ = Forms.FormSpace(0, B, "0-form");  # 0-form space with B as basis.

julia> Λ²ₕ = Forms.FormSpace(2, B, "2-form");  # 2-form space with B as basis.

julia> wedged = Λ⁰ₕ ∧ Λ²ₕ;  # Wedge operator between the two spaces.

julia> isa(wedged, Forms.Wedge{2, 2, 2})
true

```

# Fields
- `form_1::F1`: The first form.
- `form_2::F2`: The second form.
- `label::L`: A label for the wedge.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref) for the details.
- `F1 <: Forms.AbstractForm`: The type of `form_1`.
- `F2 <: Forms.AbstractForm`: The type of `form_2`.
- `L <: AbstractString`: The type of the label. Since a "∧" is added to the label, this
    type may differ from the label type of the underlying form.
"""
struct Wedge{manifold_dim, form_rank, expression_rank, F1, F2, L} <:
       AbstractForm{manifold_dim, form_rank, expression_rank}
    form_1::F1
    form_2::F2
    label::L

    function Wedge(
        form_1::F1, form_2::F2
    ) where {
        manifold_dim,
        form_rank_1,
        form_rank_2,
        expression_rank_1,
        expression_rank_2,
        F1 <: AbstractForm{manifold_dim, form_rank_1, expression_rank_1},
        F2 <: AbstractForm{manifold_dim, form_rank_2, expression_rank_2},
    }
        form_rank = form_rank_1 + form_rank_2
        expression_rank = expression_rank_1 + expression_rank_2
        if form_rank > manifold_dim
            throw(ArgumentError("""\
                The resulting form rank ($(form_rank)) exceeds the\
                manifold dimension ($(manifold_dim)).\
                """))
        end

        if expression_rank > 2
            throw(ArgumentError("""\
                The resulting expression rank ($(expression_rank)) exceeds 2.\
                """))
        end

        check_geometry(form_1, form_2)
        new_label_type = promote_type(typeof(get_label(form_1)), typeof(get_label(form_2)))
        new_label = convert(
            new_label_type,
            "(" * get_label(form_1) * LaTeXStrings.L"\wedge" * get_label(form_2) * ")",
        )

        return new{manifold_dim, form_rank, expression_rank, F1, F2, typeof(new_label)}(
            form_1, form_2, new_label
        )
    end
end

"""
    ∧

Symbolic wrapper for the wedge operator. The unicode character command is `\\wedge`. See
[`Wedge`](@ref) for the details.
"""
const ∧ = Wedge

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_forms(form::Wedge)

Return both forms to which the wedge is applied.
"""
function get_forms(form::Wedge)
    return form.form_1, form.form_2
end

function get_form(form::Wedge)
    return throw(
        ArgumentError("'get_form' is not defined for Wedges. Use 'get_forms' instead.")
    )
end

function get_geometry(form::Wedge)
    return get_geometry(first(get_forms(form)))
end

function get_estimated_nnz_per_elem(form::Wedge)
    return prod(map(get_estimated_nnz_per_elem, get_forms(form)))
end

function get_num_basis_per_expression(
    form::Wedge{manifold_dim, form_rank, 1}, element_id::Int
) where {manifold_dim, form_rank}
    bpb_all = map(get_num_basis, get_forms(form), (element_id, element_id))
    if length(bpb_all[1]) == 0
        return (bpb_all[2],)
    else
        return (bpb_all[1],)
    end
    # return (get_num_basis(get_form(form), element_id),)
end

function get_form_space_tree(wedge::Wedge)
    return tuple(get_form_space_tree(wedge.form_1)..., get_form_space_tree(wedge.form_2)...)
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    wedge::Wedge{manifold_dim}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    return _evaluate_wedge(get_forms(wedge)..., element_id, xi)
end

"""
    _evaluate_wedge(
        form_expression_1::AbstractForm{manifold_dim, form_rank_1, expression_rank_1},
        form_expression_2::AbstractForm{manifold_dim, form_rank_2, expression_rank_2},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, form_rank_1, form_rank_2, expression_rank_1, expression_rank_2}

Evaluates the wedge product of two form expressions. Each component of the wedge is a signed
sum of products of one component of each expression, as given by [`_wedge_terms`](@ref).
The components of each expression that contribute to a wedge component are merged (see
[`_merge_component_indices`](@ref)), so that the wedge component is evaluated on the union
of their indices.

# Returns
- `::Vector{Array{Float64, 1 + expression_rank_1 + expression_rank_2}}`: The evaluation
    per component of the wedge.
- `::Vector{Vector{Vector{Int}}}`: The indices per component of the wedge. For each
    component, these are the indices of the first expression followed by the indices of the
    second expression.
"""
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, form_rank_1, expression_rank_1},
    form_expression_2::AbstractForm{manifold_dim, form_rank_2, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_1, form_rank_2, expression_rank_1, expression_rank_2}
    wedge_terms = _wedge_terms(form_expression_1, form_expression_2)
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )
    n_evaluation_points = size(form_expression_1_eval[1], 1)

    n_components_wedge = length(wedge_terms)
    wedge_eval = Vector{Array{Float64, 1 + expression_rank_1 + expression_rank_2}}(
        undef, n_components_wedge
    )
    wedge_indices = Vector{Vector{Vector{Int}}}(undef, n_components_wedge)
    for wedge_component_id in eachindex(wedge_terms)
        component_terms = wedge_terms[wedge_component_id]
        merged_indices_1, column_maps_1 = _merge_component_indices(
            form_expression_1_indices,
            unique(term[2] for term in component_terms),
            Val(expression_rank_1),
        )
        merged_indices_2, column_maps_2 = _merge_component_indices(
            form_expression_2_indices,
            unique(term[3] for term in component_terms),
            Val(expression_rank_2),
        )

        component_eval = _allocate_merged(
            n_evaluation_points, (merged_indices_1..., merged_indices_2...)
        )
        for term in component_terms
            sign, component_1, component_2 = term
            _add_mapped_product!(
                component_eval,
                sign,
                form_expression_1_eval[component_1],
                column_maps_1[component_1],
                form_expression_2_eval[component_2],
                column_maps_2[component_2],
            )
        end

        wedge_eval[wedge_component_id] = component_eval
        wedge_indices[wedge_component_id] = _merged_to_indices((
            merged_indices_1..., merged_indices_2...
        ))
    end

    return wedge_eval, wedge_indices
end

"""
    _wedge_terms(
        form_expression_1::AbstractForm, form_expression_2::AbstractForm
    )

Returns the terms that make up each component of the wedge product of two forms.

# Returns
- `::Vector{Vector{NTuple{3, Int}}}`: For each component of the wedge, a list of terms
    `(sign, component_1, component_2)`, such that the component of the wedge is
    `∑ sign * α[component_1] * β[component_2]`, where `α` and `β` are the first and second
    form, respectively.

# Exceptions
- `ArgumentError`: If the wedge of the given form ranks is not implemented.
"""
function _wedge_terms(form_expression_1::AbstractForm, form_expression_2::AbstractForm)
    return throw(ArgumentError("Method not implemented for the given form expressions."))
end

# α⁰ ∧ β⁰ = (αβ)⁰
function _wedge_terms(
    ::AbstractForm{manifold_dim, 0}, ::AbstractForm{manifold_dim, 0}
) where {manifold_dim}
    return [[(1, 1, 1)]]
end

# α⁰ ∧ βᵏ = ∑ᵢ αβᵢ dξⁱ, with dξⁱ the k-form basis.
function _wedge_terms(
    ::AbstractForm{manifold_dim, 0}, ::AbstractForm{manifold_dim, form_rank_2}
) where {manifold_dim, form_rank_2}
    return [[(1, 1, component)] for component in 1:binomial(manifold_dim, form_rank_2)]
end

# αᵏ ∧ β⁰ = ∑ᵢ αᵢβ dξⁱ, with dξⁱ the k-form basis.
function _wedge_terms(
    ::AbstractForm{manifold_dim, form_rank_1}, ::AbstractForm{manifold_dim, 0}
) where {manifold_dim, form_rank_1}
    return [[(1, component, 1)] for component in 1:binomial(manifold_dim, form_rank_1)]
end

# α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ² in 2D.
_wedge_terms(::AbstractForm{2, 1}, ::AbstractForm{2, 1}) = [[(1, 1, 2), (-1, 2, 1)]]

# α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ² in 3D.
function _wedge_terms(::AbstractForm{3, 1}, ::AbstractForm{3, 1})
    return [[(1, 2, 3), (-1, 3, 2)], [(1, 3, 1), (-1, 1, 3)], [(1, 1, 2), (-1, 2, 1)]]
end

# α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃)dξ¹∧dξ²∧dξ³ in 3D, where
# β² = β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ².
function _wedge_terms(::AbstractForm{3, 1}, ::AbstractForm{3, 2})
    return [[(1, 1, 1), (1, 2, 2), (1, 3, 3)]]
end

# α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃)dξ¹∧dξ²∧dξ³ in 3D, where
# α² = α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ².
function _wedge_terms(::AbstractForm{3, 2}, ::AbstractForm{3, 1})
    return [[(1, 1, 1), (1, 2, 2), (1, 3, 3)]]
end

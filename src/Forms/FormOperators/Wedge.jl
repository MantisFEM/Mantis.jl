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
- `manifold_dim`, `form_rank`, `expression_rank`: See [AbstractForm](@ref) for the details.
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
    throw(ArgumentError("'get_form' is not defined for Wedges. Use 'get_forms' instead."))
end

function get_geometry(form::Wedge)
    return get_geometry(first(get_forms(form)))
end

function get_estimated_nnz_per_elem(form::Wedge)
    return mapreduce(get_estimated_nnz_per_elem, *, get_forms(form))
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

############################################################################################
#                                     Abstract method                                      #
############################################################################################

function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim},
    form_expression_2::AbstractForm{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    throw(ArgumentError("Method not implemented for the given form expressions."))
end

############################################################################################
#                                       Combinations                                       #
############################################################################################

# 0-forms ∧ 0-forms (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, 0, 0},
    form_expression_2::AbstractForm{manifold_dim, 0, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}

    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 0-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   β⁰ := β
    # then
    #   α⁰ ∧ β⁰ = (αβ)⁰

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a 0-form
    # therefore there is only one component
    # n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only one component because both expressions are 0-forms.
    # Only one column because both expressions have expresion rank 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1)]

    # Assign wedge_indices: only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    for cart_ind in CartesianIndices(wedge_eval[1])
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ 0-forms (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, 0, expression_rank_1},
    form_expression_2::AbstractForm{manifold_dim, 0, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank_1}

    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 0-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   β⁰ := β
    # then
    #   α⁰ ∧ β⁰ = (αβ)⁰

    # Get the number of basis for the first expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a 0-form
    # therefore there is only one component
    # n_components_wedge = 1

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only one component because both expressions are 0-forms.
    # The second expression has expresion rank 0 so the size is the same as the first
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = form_expression_1_indices # only the first expression has indices because the second expression has expression rank 0

    # Compute the wedge evaluation
    # Only need to loop ofter the basis of the first expression because the second
    # expression has expression rank 0
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][point, basis_id] =
            form_expression_1_eval[1][point, basis_id] * form_expression_2_eval[1][point, 1]
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ 0-forms (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, 0, 0},
    form_expression_2::AbstractForm{manifold_dim, 0, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank_2}

    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 0-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   β⁰ := β
    # then
    #   α⁰ ∧ β⁰ = (αβ)⁰

    # Get the number of basis for the second expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a 0-form
    # therefore there is only one component
    # n_components_wedge = 1

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only one component because both expressions are 0-forms.
    # The second expression has expresion rank 0 so the size is the same as the first
    wedge_eval = [
        Array{Float64, 1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = form_expression_2_indices # only the first expression has indices because the second expression has expression rank 0

    # Compute the wedge evaluation
    # Only need to loop ofter the basis of the first expression because the second
    # expression has expression rank 0
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][point, basis_id] =
            form_expression_1_eval[1][point, 1] * form_expression_2_eval[1][point, basis_id]
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ 0-forms (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, 0, expression_rank_1},
    form_expression_2::AbstractForm{manifold_dim, 0, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank_1, expression_rank_2}

    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 0-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   β⁰ := β
    # then
    #   α⁰ ∧ β⁰ = (αβ)⁰

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a 0-form
    # therefore there is only one component
    # n_components_wedge = 1

    # Allocate memory space for wedge_eval
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][point, basis_id_1] *
            form_expression_2_eval[1][point, basis_id_2]
    end

    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, form_rank_1, 0},
    form_expression_2::AbstractForm{manifold_dim, 0, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_1}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # k-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   αᵏ := ∑ᵢαᵢ dᵏξᵢ  (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth)
    #   β⁰ := β
    # then
    #   αᵏ ∧ β⁰ = ∑ᵢ αᵢ β dᵏξᵢ

    # Get the number of components of the wedge
    # In this case because it is the wedge of a k-form with a 0-form
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_1 = binomial(manifold_dim, form_rank_1)
    n_components_wedge = n_components_form_expression_1

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge] # Only one column because both expressions have expresion rank 0

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each
    # expression
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[wedge_component_idx][cart_ind] *
                form_expression_2_eval[1][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, form_rank_1, expression_rank_1},
    form_expression_2::AbstractForm{manifold_dim, 0, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_1, expression_rank_1}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # k-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   αᵏ := ∑ᵢαᵢ dᵏξᵢ  (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth)
    #   β⁰ := β
    # then
    #   αᵏ ∧ β⁰ = ∑ᵢ αᵢ β dᵏξᵢ

    # Get the number of components of the wedge
    # In this case because it is the wedge of a k-form with a 0-form
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_1 = binomial(manifold_dim, form_rank_1)
    n_components_wedge = n_components_form_expression_1

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    # The second expression has expresion rank 0 so the size is the same as the first expression's evaluation
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: only the first expression has indices because the second
    # expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[wedge_component_idx][cart_ind] *
                form_expression_2_eval[1][point, 1]
        end
    end

    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, form_rank_1, 0},
    form_expression_2::AbstractForm{manifold_dim, 0, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_1, expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # k-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   αᵏ := ∑ᵢαᵢ dᵏξᵢ  (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth)
    #   β⁰ := β
    # then
    #   αᵏ ∧ β⁰ = ∑ᵢ αᵢ β dᵏξᵢ

    # Get the number of components of the wedge
    # In this case because it is the wedge of a k-form with a 0-form
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_1 = binomial(manifold_dim, form_rank_1)
    n_components_wedge = n_components_form_expression_1

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    # The second first has expresion rank 0 so the size is the same as the second expression's evaluation
    wedge_eval = [
        Array{Float64, 1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ) for _ in 1:n_components_wedge
    ]

    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[wedge_component_idx][point, 1] *
                form_expression_2_eval[1][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, form_rank_1, expression_rank_1},
    form_expression_2::AbstractForm{manifold_dim, 0, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_1, expression_rank_1, expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # k-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   αᵏ := ∑ᵢαᵢ dᵏξᵢ  (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth)
    #   β⁰ := β
    # then
    #   αᵏ ∧ β⁰ = ∑ᵢ αᵢ β dᵏξᵢ

    # Get the number of components of the wedge
    # In this case because it is the wedge of a k-form with a 0-form
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_1 = binomial(manifold_dim, form_rank_1)
    n_components_wedge = n_components_form_expression_1

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ) for _ in 1:n_components_wedge
    ]

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[wedge_component_idx][point, basis_id_1] *
                form_expression_2_eval[1][point, basis_id_2]
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, 0, 0},
    form_expression_2::AbstractForm{manifold_dim, form_rank_2, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 0-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # k-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   βᵏ := ∑ᵢβᵢ dᵏξᵢ (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth)
    # then
    #   α⁰ ∧ βᵏ = ∑ᵢ α βᵢ dᵏξᵢ

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a k-form
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_2 = binomial(manifold_dim, form_rank_2)
    n_components_wedge = n_components_form_expression_2

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge]

    # Assign wedge_indices: only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[1][cart_ind] *
                form_expression_2_eval[wedge_component_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, 0, expression_rank_1},
    form_expression_2::AbstractForm{manifold_dim, form_rank_2, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_2, expression_rank_1}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 0-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # k-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   βᵏ := ∑ᵢβᵢ dᵏξᵢ (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth)
    # then
    #   α⁰ ∧ βᵏ = ∑ᵢ α βᵢ dᵏξᵢ

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a k-form
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_2 = binomial(manifold_dim, form_rank_2)
    n_components_wedge = n_components_form_expression_2

    # Get the number of basis in the first expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1, but only the first from has a variable form_rank
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[1][cart_ind] *
                form_expression_2_eval[wedge_component_idx][point, 1]
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, 0, 0},
    form_expression_2::AbstractForm{manifold_dim, form_rank_2, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_2, expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 0-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # k-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   βᵏ := ∑ᵢβᵢ dᵏξᵢ (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth)
    # then
    #   α⁰ ∧ βᵏ = ∑ᵢ α βᵢ dᵏξᵢ

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a k-form
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_2 = binomial(manifold_dim, form_rank_2)
    n_components_wedge = n_components_form_expression_2

    # Get the number of basis in the first expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2, but only the first from has a variable form_rank
    wedge_eval = [
        Array{Float64, 1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[1][point, 1] *
                form_expression_2_eval[wedge_component_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{manifold_dim, 0, expression_rank_1},
    form_expression_2::AbstractForm{manifold_dim, form_rank_2, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank_2, expression_rank_1, expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 0-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # k-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   βᵏ := ∑ᵢβᵢ dᵏξᵢ (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth)
    # then
    #   α⁰ ∧ βᵏ = ∑ᵢ α βᵢ dᵏξᵢ

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a k-form
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_2 = binomial(manifold_dim, form_rank_2)
    n_components_wedge = n_components_form_expression_2

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[1][point, basis_id_1] *
                form_expression_2_eval[wedge_component_idx][point, basis_id_2]
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{2, 1, 0},
    form_expression_2::AbstractForm{2, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{2},
)
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2], then
    #   α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1)]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    # Only one component because it is a 2-form in 2D
    for cart_ind in CartesianIndices(wedge_eval[1])
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[2][cart_ind] -
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{2, 1, expression_rank_1},
    form_expression_2::AbstractForm{2, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{2},
) where {expression_rank_1}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2], then
    #   α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    # Only one component because it is a 2-form in 2D
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[2][point, 1] -
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[1][point, 1]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{2, 1, 0},
    form_expression_2::AbstractForm{2, 1, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{2},
) where {expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2], then
    #   α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2
    wedge_eval = [
        Array{Float64, 1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    # Only one component because it is a 2-form in 2D
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][point, 1] * form_expression_2_eval[2][cart_ind] -
            form_expression_1_eval[2][point, 1] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{2, 1, expression_rank_1},
    form_expression_2::AbstractForm{2, 1, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{2},
) where {expression_rank_1, expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2], then
    #   α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # Only one component because it is a 2-form in 2D
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][point, basis_id_1] *
            form_expression_2_eval[2][point, basis_id_2] -
            form_expression_1_eval[2][point, basis_id_1] *
            form_expression_2_eval[1][point, basis_id_2]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, 0},
    form_expression_2::AbstractForm{3, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{3},
)
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 3

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for cart_ind in CartesianIndices(wedge_eval[1])
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[3][cart_ind] -
            form_expression_1_eval[3][cart_ind] * form_expression_2_eval[2][cart_ind]
    end

    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for cart_ind in CartesianIndices(wedge_eval[2])
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[3][cart_ind] * form_expression_2_eval[1][cart_ind] -
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[3][cart_ind]
    end

    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for cart_ind in CartesianIndices(wedge_eval[3])
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[2][cart_ind] -
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, expression_rank_1},
    form_expression_2::AbstractForm{3, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_1}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 3

    # Allocate memory space for wedge_eval
    # Only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[3][point, 1] -
            form_expression_1_eval[3][cart_ind] * form_expression_2_eval[2][point, 1]
    end

    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for cart_ind in CartesianIndices(wedge_eval[2])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[3][cart_ind] * form_expression_2_eval[1][point, 1] -
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[3][point, 1]
    end

    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for cart_ind in CartesianIndices(wedge_eval[3])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[2][point, 1] -
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[1][point, 1]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, 0},
    form_expression_2::AbstractForm{3, 1, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 3

    # Allocate memory space for wedge_eval
    # Only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2
    wedge_eval = [
        Array{Float64, 1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[2][point, 1] * form_expression_2_eval[3][cart_ind] -
            form_expression_1_eval[3][point, 1] * form_expression_2_eval[2][cart_ind]
    end

    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for cart_ind in CartesianIndices(wedge_eval[2])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[3][point, 1] * form_expression_2_eval[1][cart_ind] -
            form_expression_1_eval[1][point, 1] * form_expression_2_eval[3][cart_ind]
    end

    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for cart_ind in CartesianIndices(wedge_eval[3])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][point, 1] * form_expression_2_eval[2][cart_ind] -
            form_expression_1_eval[2][point, 1] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, expression_rank_1},
    form_expression_2::AbstractForm{3, 1, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_1, expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 3

    # Allocate memory space for wedge_eval
    wedge_eval = [
        Array{Float64, 1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[2][point, basis_id_1] *
            form_expression_2_eval[3][point, basis_id_2] -
            form_expression_1_eval[3][point, basis_id_1] *
            form_expression_2_eval[2][point, basis_id_2]
    end

    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for cart_ind in CartesianIndices(wedge_eval[2])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[3][point, basis_id_1] *
            form_expression_2_eval[1][point, basis_id_2] -
            form_expression_1_eval[1][point, basis_id_1] *
            form_expression_2_eval[3][point, basis_id_2]
    end

    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for cart_ind in CartesianIndices(wedge_eval[3])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][point, basis_id_1] *
            form_expression_2_eval[2][point, basis_id_2] -
            form_expression_1_eval[2][point, basis_id_1] *
            form_expression_2_eval[1][point, basis_id_2]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, 0},
    form_expression_2::AbstractForm{3, 2, 0},
    element_id::Int,
    xi::Points.AbstractPoints{3},
)
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 2-form

    # Compute the wedge product between the 1-form and the 2-form in 3D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β² := β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ²
    # with i = [1, 2, 3], then
    #   α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 1-form wedge with a 2-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            wedge_eval[1][cart_ind] +=
                form_expression_1_eval[form_components_idx][cart_ind] *
                form_expression_2_eval[form_components_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, expression_rank_1},
    form_expression_2::AbstractForm{3, 2, 0},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_1}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 2-form

    # Compute the wedge product between the 1-form and the 2-form in 3D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β² := β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ²
    # with i = [1, 2, 3], then
    #   α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 1-form wedge with a 2-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1
    wedge_eval = [
        zeros(
            n_evaluation_points, num_basis_form_expression_1...
        )::Array{Float64, 1 + expression_rank_1} for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] +=
                form_expression_1_eval[form_components_idx][cart_ind] *
                form_expression_2_eval[form_components_idx][point, 1]
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, 0},
    form_expression_2::AbstractForm{3, 2, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 2-form

    # Compute the wedge product between the 1-form and the 2-form in 3D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β² := β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ²
    # with i = [1, 2, 3], then
    #   α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 1-form wedge with a 2-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2
    wedge_eval = [
        zeros(
            n_evaluation_points, num_basis_form_expression_2...
        )::Array{Float64, 1 + expression_rank_2} for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] +=
                form_expression_1_eval[form_components_idx][point, 1] *
                form_expression_2_eval[form_components_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 1, expression_rank_1},
    form_expression_2::AbstractForm{3, 2, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_1, expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 2-form

    # Compute the wedge product between the 1-form and the 2-form in 3D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β² := β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ²
    # with i = [1, 2, 3], then
    #   α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 1-form wedge with a 2-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    wedge_eval = [
        zeros(
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        )::Array{Float64, 1 + expression_rank_1 + expression_rank_2} for
        _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] +=
                form_expression_1_eval[form_components_idx][point, basis_id_1] *
                form_expression_2_eval[form_components_idx][point, basis_id_2]
        end
    end

    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 2, 0},
    form_expression_2::AbstractForm{3, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{3},
)
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 2-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 2-form and the 1-form in 3D

    # Given
    #   α² := α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ²
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 2-form wedge with a 1-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            wedge_eval[1][cart_ind] +=
                form_expression_1_eval[form_components_idx][cart_ind] *
                form_expression_2_eval[form_components_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 2, expression_rank_1},
    form_expression_2::AbstractForm{3, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_1}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 2-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 2-form and the 1-form in 3D

    # Given
    #   α² := α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ²
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 2-form wedge with a 1-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1
    wedge_eval = [
        zeros(
            n_evaluation_points, num_basis_form_expression_1...
        )::Array{Float64, 1 + expression_rank_1} for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] +=
                form_expression_1_eval[form_components_idx][cart_ind] *
                form_expression_2_eval[form_components_idx][point, 1]
        end
    end

    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 2, 0},
    form_expression_2::AbstractForm{3, 1, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 2-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 2-form and the 1-form in 3D

    # Given
    #   α² := α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ²
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 2-form wedge with a 1-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2
    wedge_eval = [
        zeros(
            n_evaluation_points, num_basis_form_expression_2...
        )::Array{Float64, 1 + expression_rank_2} for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] +=
                form_expression_1_eval[form_components_idx][point, 1] *
                form_expression_2_eval[form_components_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractForm{3, 2, expression_rank_1},
    form_expression_2::AbstractForm{3, 1, expression_rank_2},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank_1, expression_rank_2}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 2-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 2-form and the 1-form in 3D

    # Given
    #   α² := α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ²
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 2-form wedge with a 1-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    wedge_eval = [
        zeros(
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        )::Array{Float64, 1 + expression_rank_1 + expression_rank_2} for
        _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] +=
                form_expression_1_eval[form_components_idx][point, basis_id_1] *
                form_expression_2_eval[form_components_idx][point, basis_id_2]
        end
    end

    return wedge_eval, wedge_indices
end

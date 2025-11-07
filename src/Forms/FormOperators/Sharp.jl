############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    Sharp{manifold_dim, G, F}

Represents the sharp operator, which converts a differential 1-form into a vector field.

# Fields
- `form::F`: The differential 1-form to be converted into a vector field.

# Type Parameters
- `manifold_dim`: The dimension of the manifold.
- `G`: The geometry type associated with the manifold.
- `F`: The type of the differential 1-form.

# Inner Constructors
- `Sharp(form::F)`: General constructor.
"""
struct Sharp{manifold_dim, G, F}
    form::F

    function Sharp(
        form::F
    ) where {manifold_dim, G, F <: AbstractFormExpression{manifold_dim, 1, 0, G}}
        if manifold_dim < 2
            throw(
                ArgumentError(
                    "Manifold dimension should be greater than 1. " *
                    "Dimension $(manifold_dim) was given.",
                ),
            )
        end

        return new{manifold_dim, G, F}(form)
    end
end

function ♯(form::AbstractFormExpression)
    return Sharp(form)
end

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_form(sharp::Sharp)

Returns the form to which the sharp is applied.

# Arguments
- `sharp::Sharp`: The sharp structure.

# Returns
- `<:AbstractFormExpression`: The form to which the sharp is applied.
"""
get_form(sharp::Sharp) = sharp.form

"""
    get_form_space_tree(wedge::Wedge)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the sharp operator, e.g., for
`♯((α ∧ β) + γ)`, it returns the spaces of `α`, `β`, and `γ`, if all have exprssion_rank > 1.
If `α` has expression_rank = 0, it returns only the spaces of `β` and `γ`.

# Arguments
- `sharp::Sharp`: The sharp structure.

# Returns
- `Tuple(<:AbstractFormExpression)`: The list of forms present in the tree of the sharp.
"""
function get_form_space_tree(sharp::Sharp)
    return get_form_space_tree(sharp.form)
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

"""
    evaluate(
        sharp::Sharp{manifold_dim}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
    ) where {manifold_dim}

Evaluates the sharp operator on a differential 1-form over a specified element of a
manifold, converting the form into a vector field. Note that both the 1-form and the
vector-field are defined in reference, curvilinear coordinates.

# Arguments
- `sharp::Sharp{manifold_dim}`: The sharp structure containing the form to be evaluated.
- `element_id::Int`: The identifier of the element on which the sharp is to be evaluated.
- `xi::Points.AbstractPoints{manifold_dim}`: A tuple containing vectors of floating-point
    numbers representing the coordinates at which the 1-form is evaluated. Each vector within
    the tuple corresponds to a dimension of the manifold.

# Returns
- `::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each ∂ᵢ,
    stores the sharp evaluation. The size of each matrix is (number of evaluation
    points)x(number of basis functions).
- `::Vector{Vector{Int}}`: Each component of the vector, corresponding to each ∂ᵢ, stores
    the indices of the evaluated basis functions.
"""
function evaluate(
    sharp::Sharp{manifold_dim}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    form = get_form(sharp)

    return _evaluate_sharp(form, element_id, xi)
end

function _evaluate_sharp(
    form_expression::AbstractFormExpression{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    inv_g, _, _ = Geometry.inv_metric(get_geometry(form_expression), element_id, xi)

    num_form_components = manifold_dim # = binomial(manifold_dim, 1)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)

    sharp_eval = Vector{Matrix{Float64}}(undef, num_form_components)

    # ♯: dξⁱ ↦ ♯(dξⁱ) = gⁱʲ∂ⱼ
    for component in 1:num_form_components
        sharp_eval[component] = @views hcat(
            [form_eval[i] .* inv_g[:, i, component] for i in 1:num_form_components]...
        )
    end

    return sharp_eval, form_indices
end

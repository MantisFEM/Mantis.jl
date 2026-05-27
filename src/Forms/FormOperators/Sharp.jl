############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    Sharp{manifold_dim, F}

Represents the sharp operator, which converts a differential 1-form into a vector field.

# Fields
- `form::F`: The differential 1-form to be converted into a vector field.

# Type Parameters
- `manifold_dim`: The dimension of the manifold.
- `F`: The type of the differential 1-form.

# Inner Constructors
- `Sharp(form::F)`: General constructor.
"""
struct Sharp{manifold_dim, F}
    form::F

    function Sharp(form::F) where {manifold_dim, F <: AbstractForm{manifold_dim, 1, 0}}
        if manifold_dim < 2
            throw(
                ArgumentError(
                    "Manifold dimension should be greater than 1. " *
                    "Dimension $(manifold_dim) was given.",
                ),
            )
        end

        return new{manifold_dim, F}(form)
    end
end

"""
    ♯

Symbolic wrapper for the sharp operator. The unicode character command is `\\sharp`. See
[`Sharp`](@ref) for the details.
"""
const ♯ = Sharp

############################################################################################
#                                         Getters                                          #
############################################################################################

# Since the Sharp is not an AbstractForm, we have to define a few simple method
# specifically for the Sharp.
get_form(sharp::Sharp) = sharp.form

function get_form_space_tree(sharp::Sharp)
    return get_form_space_tree(get_form(sharp))
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
	numbers representing the coordinates at which the 1-form is evaluated. Each vector
	within the tuple corresponds to a dimension of the manifold.

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
    form_expression::AbstractForm{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    inv_g, _, _ = Geometry.inv_metric(get_geometry(form_expression), element_id, xi)
    num_form_components = manifold_dim # = binomial(manifold_dim, 1)
    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    num_points = size(form_eval[1], 1)
    sharp_eval = [zeros(num_points, 1) for _ in 1:num_form_components]
    # ♯: dξⁱ ↦ ♯(dξⁱ) = gⁱʲ∂ⱼ
    for point in 1:num_points
        inv_g_point = inv_g[point]
        for new_component in 1:num_form_components
            sharp_row = view(sharp_eval[new_component], point, :)
            for old_component in 1:num_form_components
                scaling = inv_g_point[new_component, old_component]
                old_row = view(form_eval[old_component], point, :)
                if !iszero(scaling)
                    sharp_row .+= scaling .* old_row
                end
            end
        end
    end

    return sharp_eval, form_indices
end

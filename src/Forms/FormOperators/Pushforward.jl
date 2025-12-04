"""
    evaluate_pushforward(vfield::Vector{Matrix{Float64}},
                         jacobian::Array{Float64,3})

Evaluate the pushforward of the vector field at the discrete points where it has been evaluated. The pushforward is the action of the Jacobian of the field on the field itself.

# Arguments
- `vfield::Vector{Matrix{Float64}}`: The pointwise evaluated vector field to evaluate the pushforward for.
- `jacobian::Array{Float64,3}`: The Jacobian of the vector field evaluated at the discrete points.
- `manifold_dim::Int`: The dimension of the embedding manifold.

# Returns
- `::Vector{Matrix{Float64}}`: The evaluated pushforward of the vector field at the discrete points.
"""
function evaluate_pushforward(
    vfield::Vector{Matrix{Float64}}, jacobian::AbstractVector, manifold_dim::Int
)
    image_dim = size(jacobian[1], 1)
    # Gᵢ: v ↦ Gᵢ(v) = Jᵢⱼvʲ
    eval_dims = size(vfield[1])
    evaluated_pushforward = [zeros(eval_dims) for _ in 1:image_dim]
    for point in 1:eval_dims[1]
        jacobian_point = jacobian[point]
        for new_component in 1:image_dim
            pushforward_row = view(evaluated_pushforward[new_component], point, :)
            for old_component in 1:manifold_dim
                old_row = view(vfield[old_component], point, :)
                pushforward_row .+= jacobian_point[new_component, old_component] .* old_row
            end
        end
    end

    return evaluated_pushforward
end

"""
    evaluate_sharp_pushforward(
        form_expression::AbstractForm{manifold_dim, 1, 0},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Compute the pushforward of the sharp of a differential 1-form over a specified element of a
manifold, converting the form into a vector field. Note that the output vector-field is
defined in physical coordinates.

# Arguments
- `form_expression::AbstractForm{manifold_dim, 1, G}`: An expression representing
    the 1-form on the manifold.
- `element_id::Int`: The identifier of the element on which the sharp is to be evaluated.
- `xi::Points.AbstractPoints{manifold_dim}`: A tuple containing vectors of floating-point
    numbers representing the coordinates at which the 1-form is evaluated. Each vector within
    the tuple corresponds to a dimension of the manifold.

# Returns
- `evaluated_pushforward::Vector{Matrix{Float64}}`: Each component of the vector, stores the
    evaluated pushforward of the sharp of the 1-form. The size of each matrix is (number of
    evaluation points)x(number of basis functions).
- `sharp_indices::Vector{Vector{Int}}`: Each component of the vector, stores the indices of
    the evaluated basis functions.
"""
function evaluate_sharp_pushforward(
    form_expression::AbstractForm{manifold_dim, 1, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    sharp = Sharp(form_expression)
    sharp_eval, sharp_indices = evaluate(sharp, element_id, xi)

    # dξⁱ ↦ G∘♯ (dξⁱ)
    jacobian = Geometry.jacobian(get_geometry(form_expression), element_id, xi)
    evaluated_pushforward = evaluate_pushforward(sharp_eval, jacobian, manifold_dim)

    return evaluated_pushforward, sharp_indices
end

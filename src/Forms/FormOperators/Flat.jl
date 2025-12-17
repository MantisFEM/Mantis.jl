
# @doc raw"""
#     evaluate_flat(form_expression::AbstractForm{manifold_dim, 1, G},
#                    element_id::Int,
#                    xi::Points.AbstractPoints{manifold_dim})
#                    where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}

# Compute the flat of a vector field over a specified element of a manifold, converting the vector field into a 1-form. Note that both the 1-form and the vector-field are defined in reference, curvilinear coordinates.

# # Arguments
# - `form_expression::AbstractForm{manifold_dim, 1, G}`: An expression representing the vector field on the manifold.
# - `element_id::Int`: The identifier of the element on which the flat is to be evaluated.
# - `xi::Points.AbstractPoints{manifold_dim}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the 1-form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# # Returns
# - `flat_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each dξⁱ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
# - `flat_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each dξⁱ, stores the indices of the evaluated basis functions.
# """
# function evaluate_flat(form_expression::AbstractForm{manifold_dim, 1, G}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
#     manifold_dim >= 2 || throw(ArgumentError("Manifold dimension should be 2 or 3 for 1-forms. Dimension $manifold_dim was given."))

#     g, _ = Geometry.metric(form_expression.geometry, element_id, xi)

#     num_form_components = manifold_dim # = binomial(manifold_dim, 1)

#     form_eval, form_indices = evaluate(form_expression, element_id, xi)

#     flat_eval = Vector{Matrix{Float64}}(undef, num_form_components)

#     for component ∈ 1:num_form_components
#         flat_eval[component] = @views hcat([form_eval[i] .* g[:, i, component] for i in 1:num_form_components]...)
#     end

#     flat_indices = repeat([vcat(form_indices...)], num_form_components)

#     return flat_eval, flat_indices
# end

# @doc raw"""
#     evaluate_flat(form_expression::AbstractForm{manifold_dim, 2, G},
#                    element_id::Int,
#                    xi::Points.AbstractPoints{manifold_dim})
#                    where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}

# Compute the flat of a vector field over a specified element of a manifold, converting the vector field into a form.

# # Arguments
# - `form_expression::AbstractForm{manifold_dim, 2, G}`: An expression representing the vector field on the manifold.
# - `element_id::Int`: The identifier of the element on which the flat is to be evaluated.
# - `xi::Points.AbstractPoints{manifold_dim}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the 2-form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# # Returns
# - `flat_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each dξⁱ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
# - `hodge_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each dξⁱ, stores the indices of the evaluated basis functions. Uses the indices of the hodge evaluation because they are the same.
# """
# function evaluate_flat(form_expression::AbstractForm{manifold_dim, 2, G}, element_id::Int, xi::Points.AbstractPoints{manifold_dim}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
#     manifold_dim == 3 || throw(ArgumentError("Manifold dimension should be 3 for 2-forms. Dimension $manifold_dim was given."))

#     g, _ = Geometry.metric(form_expression.geometry, element_id, xi)

#     num_form_components = manifold_dim #binomial(manifold_dim, 2)

#     hodge_eval, hodge_indices = evaluate(hodge(form_expression), element_id, xi)

#     flat_eval = Vector{Matrix{Float64}}(undef, num_form_components)

#     # ♯: dξⁱ∧dξʲ ↦ ♯★dξⁱ∧dξʲ
#     for component ∈ 1:num_form_components
#         flat_eval[component] = @views reduce(+, [hodge_eval[i] .* g[:, i, component] for i in 1:num_form_components])
#     end

#     return flat_eval, hodge_indices
# end


@doc raw"""
    evaluate_rotated_proxy_vector_field(form_expression::AbstractForm{manifold_dim, form_rank, G},
                                        element_id::Int,
                                        xi::Points.AbstractPoints{manifold_dim})
                                        where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Compute the rotated proxy vector-field associated to a differential (n-1)-form over a specified element of an n-dimensional manifold, converting the form into a vector field. Note that both the form and the vector-field are defined in reference, curvilinear coordinates.

# Arguments
- `form_expression::AbstractForm{manifold_dim, form_rank, G}`: An expression representing the form on the manifold.
- `element_id::Int`: The identifier of the element on which the sharp is to be evaluated.
- `xi::Points.AbstractPoints{manifold_dim}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# Returns
- `sharp_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
- `hodge_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the indices of the evaluated basis functions. Uses the indices of the hodge evaluation because they are the same.

# Throws an error if the manifold dimension is less than 2 or the form rank is not one less than the manifold dimension.
"""
function evaluate_rotated_proxy_vector_field(
    form_expression::AbstractForm{manifold_dim, form_rank, G},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    manifold_dim >= 2 && form_rank == manifold_dim - 1 || throw(
        ArgumentError(
            "Manifold dimension should be 2 or higher and form rank should be 1 less than its value. Dimensionmanifold_dim and form rankform_rank were given.",
        ),
    )

    # Examples...
    # 2D: dξⁱ ↦ ♯∘★ (dξⁱ)
    # 3D: dξⁱ∧dξʲ ↦ ♯∘★ (dξⁱ∧dξʲ)
    # ... and so on.
    hodge_form = hodge(form_expression)
    return evaluate_sharp(hodge_form, element_id, xi)

    # OLD IMPLEMENTATION BELOW:
    # hodge_eval, hodge_indices = evaluate(hodge(form_expression), element_id, xi)
    # sharp_eval = Vector{Matrix{Float64}}(undef, num_form_components)

    # for component ∈ 1:num_form_components
    #     sharp_eval[component] = @views reduce(+,[hodge_eval[i] .* inv_g[:, i, component] for i in 1:num_form_components])
    # end

    # return sharp_eval, hodge_indices
end

@doc raw"""
    evaluate_rotated_proxy_vector_field_pushforward(form_expression::AbstractForm{manifold_dim, form_rank, G},
                                        element_id::Int,
                                        xi::Points.AbstractPoints{manifold_dim})
                                        where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Compute the pushforward of the rotated proxy vector-field associated to a differential (n-1)-form over a specified element of an n-dimensional manifold, converting the form into a vector field. Note that both the vector-field is defined in physical coordinates.

# Arguments
- `form_expression::AbstractForm{manifold_dim, form_rank, G}`: An expression representing the form on the manifold.
- `element_id::Int`: The identifier of the element on which the sharp is to be evaluated.
- `xi::Points.AbstractPoints{manifold_dim}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# Returns
- `sharp_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
- `hodge_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the indices of the evaluated basis functions. Uses the indices of the hodge evaluation because they are the same.

# Throws an error if the manifold dimension is less than 2 or the form rank is not one less than the manifold dimension.
"""
function evaluate_rotated_proxy_vector_field_pushforward(
    form_expression::AbstractForm{manifold_dim, form_rank, G},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    manifold_dim >= 2 && form_rank == manifold_dim - 1 || throw(
        ArgumentError(
            "Manifold dimension should be 2 or higher and form rank should be 1 less than its value. Dimensionmanifold_dim and form rankform_rank were given.",
        ),
    )

    # Examples...
    # 2D: dξⁱ ↦ G∘♯∘★(dξⁱ)
    # 3D: dξⁱ∧dξʲ ↦ G∘♯∘★(dξⁱ∧dξʲ)
    # ... and so on.
    hodge_form = hodge(form_expression)
    return evaluate_sharp_pushforward(hodge_form, element_id, xi)
end

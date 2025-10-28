
"""
    metric(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Returns the metric and its determinant.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `g::Vector{SMatrix{manifold_dim, manifold_dim}}`: Metric tensor per evaluation point.
- `sqrt_g::Vector{Float64}`: Square-root of the determinant of the metric per evaluation
    point.
"""
function metric(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    # The metric tensor gᵢⱼ is
    #   gᵢⱼ := ∑ₖᵐ ∂Φᵏ\∂ξᵢ ⋅ ∂Φᵏ\∂ξⱼ
    # Since the Jacobian matrix J is given by
    #   Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # The metric tensor can be computed as
    #   g = Jᵗ ⋅ J

    J = jacobian(geometry, element_id, xi)

    g = transpose.(J) .* J
    sqrt_g = sqrt.(abs.(det.(g)))

    return g, sqrt_g
end

"""
    inv_metric(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Returns the inverse metric, the metric and its determinant.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `::Vector{SMatrix{manifold_dim, manifold_dim}}`: Inverse metric tensor per evaluation
    point.
- `g::Vector{SMatrix{manifold_dim, manifold_dim}}`: Metric tensor per evaluation point.
- `sqrt_g::Vector{Float64}`: Square-root of the determinant of the metric per evaluation
    point.
"""
function inv_metric(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    g, sqrt_g = metric(geometry, element_id, xi)

    return inv.(g), g, sqrt_g
end

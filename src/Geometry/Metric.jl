
function metric(
    geometry::AbstractGeometry{manifold_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, num_patches}
    # The metric tensor gᵢⱼ is
    #   gᵢⱼ := ∑ₖᵐ ∂Φᵏ\∂ξᵢ ⋅ ∂Φᵏ\∂ξⱼ
    # Since the Jacobian matrix J is given by
    #   Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # The metric tensor can be computed as
    #   g = Jᵗ ⋅ J

    # Compute the jacobian
    J = jacobian(geometry, element_id, xi)

    g = [transpose(J[i])*J[i] for i in eachindex(J)]
    sqrt_g = sqrt.(abs.(det.(g)))

    return g, sqrt_g
end

function inv_metric(
    geometry::AbstractGeometry{manifold_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, num_patches}
    # The inverse of the metric tensor gᵢⱼ is
    #   [gⁱʲ] := [gᵢⱼ]⁻¹
    # The metric tensor is computed for all evaluation points.
    g, sqrt_g = metric(geometry, element_id, xi)

    return inv.(g), g, sqrt_g
end

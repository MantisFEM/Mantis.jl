
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

    n_evaluation_points = Points.get_num_points(xi)  # compute the number of nD points evaluated
    g = zeros(Float64, n_evaluation_points, manifold_dim, manifold_dim)

    # Compute the metric
    for index in CartesianIndices(g)
        (point, row, column) = Tuple(index)
        # Here we compute: gᵢⱼ := ∑ₖᵐ ∂Φᵏ\∂ξᵢ ⋅ ∂Φᵏ\∂ξⱼ
        # with:
        #   - i := row
        #   - j := column
        # The summation over k corresponds to the summation over the second dimension.
        for k in 1:manifold_dim
            g[point, row, column] += J[point, k, row] * J[point, k, column]
        end
    end

    sqrt_g = sqrt.(abs.(LinearAlgebra.det.(eachslice(g; dims=1))))

    return g, sqrt_g
end

function inv_metric(
    geometry::AbstractGeometry{manifold_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, num_patches}
    # The inverse of the metric tensor gᵢⱼ is
    #   [gⁱʲ] := [gᵢⱼ]⁻¹
    # The metric tensor is computed for all evaluation points. To avoid having to compute
    # the inverse of the metric tensor for each evaluation point, we explicitly encode
    # the inverse in order to be able to do it in one go for all evaluation points.
    # We can do this because we have a limited number of possible inverses we need to
    # compute, namely:
    #   manifold_dim x manifold_dim
    # with manifold_dim = 1, 2, 3
    # in the future if we wish to add space-time approaches we can add the manifold_dim = 4
    # case.

    # First compute the metric
    g, sqrt_g = metric(geometry, element_id, xi)

    # Then compute the inverse of the metric
    # Permuting the dimensions of g is needed because inv can't handle non-contiguous
    # columns, which is what happens when using views in the original g storage order.
    # This has been compared with looping over the points of g and copying the corresponding
    # inverse to a pre-allocated array, and it appeared to be faster and with less
    # allocations by about a 1/3 factor. Still, a better solution is needed ― possibly
    # changing the way geometry evaluations are stored.
    g_permuted = permutedims(g, (2, 3, 1))
    inv_g = Array{Float64, 3}(undef, size(g))
    for point in axes(g_permuted, 3)
        inv_g[point, :, :] .= LinearAlgebra.inv(view(g_permuted, :, :, point))
    end

    return inv_g, g, sqrt_g
end

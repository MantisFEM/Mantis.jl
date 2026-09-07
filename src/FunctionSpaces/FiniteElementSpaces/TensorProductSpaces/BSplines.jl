"""
    get_greville_points(
        space::TensorProductSpace{manifold_dim, num_patches, num_spaces, T}
    ) where {manifold_dim, num_patches, num_spaces, T <: NTuple{num_spaces, BSplineSpace}}

Compute the Greville points for a `TensorProductSpace` composed of `BSplineSpace`s.
"""
function get_greville_points(
    space::TensorProductSpace{manifold_dim, num_patches, num_spaces, T}
) where {manifold_dim, num_patches, num_spaces, T <: NTuple{num_spaces, BSplineSpace}}
    factor_greville_points = map(get_greville_points, get_factor_spaces(space))
    greville_points = Vector{Vector{Float64}}(undef, manifold_dim)
    factor_manifold_dims = get_factor_manifold_dims(space)
    cum_factor_manifold_dims = (0, cumsum(factor_manifold_dims)...)
    for space_id in 1:num_spaces, dim in 1:factor_manifold_dims[space_id]
        greville_points[dim + cum_factor_manifold_dims[space_id]] = factor_greville_points[space_id][dim]
    end

    return Tuple(greville_points)
end

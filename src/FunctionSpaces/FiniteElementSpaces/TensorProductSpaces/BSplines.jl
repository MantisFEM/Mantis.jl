"""
    get_greville_points(space::TensorProductSpace{manifold_dim, T}) where {
        manifold_dim, num_spaces, T <: NTuple{num_spaces, BSplineSpace}
    }

Compute the Greville points for a `TensorProductSpace` composed of `BSplineSpace`s.

# Arguments

  - `space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces, T}`:
    The tensor product space containing `BSplineSpace`s.

# Returns

  - `::NTuple{manifold_dim, Vector{Float64}}`: A tuple of vectors containing the Greville
    points for each dimension of the tensor product space.
"""
function get_greville_points(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces, T}
) where {
    manifold_dim,
    num_components,
    num_patches,
    num_spaces,
    T <: NTuple{num_spaces, BSplineSpace},
}
    const_greville_points = map(get_greville_points, get_constituent_spaces(space))
    greville_points = Vector{Vector{Float64}}(undef, manifold_dim)
    const_manifold_dims = get_constituent_manifold_dim(space)
    cum_const_manifold_dims = (0, cumsum(const_manifold_dims)...)
    for space_id in 1:num_spaces
        for dim in 1:const_manifold_dims[space_id]
            greville_points[dim + cum_const_manifold_dims[space_id]] = const_greville_points[space_id][dim]
        end
    end

    return Tuple(greville_points)
end

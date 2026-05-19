############################################################################################
#                                          fill!                                           #
############################################################################################

function point_evaluate(
    geometry::CartesianGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    # Define the affine map for the element. It is the same for every point.
    const_element_id, patch_id = get_constituent_element_id(geometry, element_id)
    scaling = get_element_lengths(geometry, element_id)
    offset = ntuple(
        dim -> get_breakpoint(geometry, patch_id, dim, const_element_id[dim]), manifold_dim
    )

    # Return affine map applied to point.
    return point -> map(affine_map, point, scaling, offset)
end

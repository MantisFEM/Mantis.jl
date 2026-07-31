
# Conversion from a TensorProductGeometry to a Cartesian geometry. Note that the type
# signature already enforces that the TensorProductGeometry consists of only
# CartesianGeometries.
function Base.convert(
    ::Type{CartesianGeometry},
    geometry::TensorProductGeometry{
        manifold_dim, image_dim, num_patches, num_geometries, T, CI
    },
) where {
    manifold_dim,
    image_dim,
    num_patches,
    num_geometries,
    T <: NTuple{num_geometries, CartesianGeometry},
    CI,
}
    const_geometries = get_constituent_geometries(geometry)
    const_manifold_dims = map(get_manifold_dim, const_geometries)
    const_num_patches = map(get_num_patches, const_geometries)
    cart_num_patches = CartesianIndices(const_num_patches)
    breakpoints_per_patch = ntuple(num_patches) do patch_id
        return merge_tuples(
            (map(get_breakpoints, const_geometries, Tuple(cart_num_patches[patch_id])))...
        )
    end

    return CartesianGeometry(breakpoints_per_patch)
end

merge_two_tuples(tup::NTuple{N, Any}, tup2::NTuple{N2, Any}) where {N, N2} =
    (tup..., tup2...)
function merge_tuples(tup::NTuple{N, Any}, tups...) where {N}
    if length(tups) == 0
        return tup
    elseif length(tups) == 1
        return merge_two_tuples(tup, first(tups))
    end
    return merge_tuples(merge_two_tuples(tup, first(tups)), Base.tail(tups)...)
end

############################################################################################
#                                         prealloc                                         #
############################################################################################

function Caching._preallocate(
    ::AbstractGeometry{manifold_dim, image_dim}, points::Points.AbstractPoints{manifold_dim}
) where {manifold_dim, image_dim}
    num_points = Points.get_num_points(points)
    prealloc = (; eval=MMatrix{num_points, image_dim, eltype(points)}(undef))

    return prealloc
end

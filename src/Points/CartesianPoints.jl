"""
    CartesianPoints{manifold_dim, T, CI, LI} <: AbstractPoints{manifold_dim}

Represents a set of points constructed from `manifold_dim` lists of uni-dimensional points.
Conceptually, this structure combines the functionalities of `CartesianIndices` and
`Iterators.product`.

# Fields
- `constituent_points::NTuple{manifold_dim, T}`: The set of points per manifold dimension.
- `cart_num_points::CI`: The `CartesianIndices` used to convert from linear to cartesian
  indexing.
- `lin_num_points::LI`: The `LinearIndices` used to convert from cartesian to linear
  indexing.
"""
struct CartesianPoints{manifold_dim, T, CI, LI} <: AbstractPoints{manifold_dim}
    constituent_points::NTuple{manifold_dim, T}
    cart_num_points::CI
    lin_num_points::LI

    function CartesianPoints(
        constituent_points::NTuple{manifold_dim, T}
    ) where {manifold_dim, T <: AbstractVector}
        cart_num_points = CartesianIndices(
            ntuple(dim -> length(constituent_points[dim]), manifold_dim)
        )
        lin_num_points = LinearIndices(cart_num_points)

        return new{manifold_dim, T, typeof(cart_num_points), typeof(lin_num_points)}(
            constituent_points, cart_num_points, lin_num_points
        )
    end
end

Base.eltype(::CartesianPoints{manifold_dim, T}) where {manifold_dim, T} = eltype(T)

"""
    get_cart_num_points(points::CartesianPoints)

Returns the `CartesianIndices` used to convert from linear to cartesian indexing.
"""
get_cart_num_points(points::CartesianPoints) = points.cart_num_points
get_num_points(points::CartesianPoints) = length(get_lin_num_points(points))

"""
    get_cart_num_points(points::CartesianPoints)

Returns the `LinearIndices` used to convert from cartesian to linear indexing.
"""
get_lin_num_points(points::CartesianPoints) = points.lin_num_points

"""
    get_constituent_num_points(
      points::CartesianPoints{manifold_dim}
    ) where {manifold_dim}

Returns the number of constituent points per manifold dimension.
"""
function get_constituent_num_points(
    points::CartesianPoints{manifold_dim}
) where {manifold_dim}
    return Tuple(maximum(get_cart_num_points(points)))
end

function Base.getindex(points::CartesianPoints, i::Int)
    return getindex(points, get_cart_num_points(points)[i])
end

function Base.getindex(
    points::CartesianPoints{manifold_dim},
    i::Union{NTuple{manifold_dim}, CartesianIndex{manifold_dim}},
) where {manifold_dim}
    const_points = get_constituent_points(points)

    return ntuple(dim -> const_points[dim][i[dim]], manifold_dim)
end

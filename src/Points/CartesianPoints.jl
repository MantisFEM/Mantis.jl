"""
    CartesianPoints{manifold_dim, T, CP, CI, LI} <: AbstractPoints{manifold_dim}

Represents a set of points constructed from `manifold_dim` lists of uni-dimensional points.
Conceptually, this structure combines the functionalities of `CartesianIndices` and
`Iterators.product`.

# Fields
- `constituent_points::CP`: The set of points per manifold dimension.
- `cart_num_points::CI`: The `CartesianIndices` used to convert from linear to cartesian
    indexing.
- `lin_num_points::LI`: The `LinearIndices` used to convert from cartesian to linear
    indexing.
- `iteration_order::NTuple{manifold_dim, Int}`: Used to determine the iteration order over
    `cart_num_points`. If the `dim`-th entry has value `i`, then dimension `dim` will be the
    `i`-th fastest changing index.
- `permuted_cart_num_points::CI`: A permuted version of `cart_num_points` as given by
    `iteration_order`.

# Example
```julia
julia> points = Points.CartesianPoints([1,2], [1,2,3]; iteration_order=(1,2));

julia> for point in points
           display(point)
       end
(1, 1)
(2, 1)
(1, 2)
(2, 2)
(1, 3)
(2, 3)

julia> points = Points.CartesianPoints([1,2], [1,2,3]; iteration_order=(2,1));

julia> for point in points
           display(point)
       end
(1, 1)
(1, 2)
(1, 3)
(2, 1)
(2, 2)
(2, 3)
```
"""
struct CartesianPoints{manifold_dim, T, CP, CI, LI} <: AbstractPoints{manifold_dim}
    constituent_points::CP
    cart_num_points::CI
    lin_num_points::LI
    iteration_order::NTuple{manifold_dim, Int}
    permuted_cart_num_points::CI

    function CartesianPoints(
        constituent_points::Vararg{AbstractVector{T}, manifold_dim};
        iteration_order::NTuple{manifold_dim, Int}=ntuple(k -> k, manifold_dim),
    ) where {manifold_dim, T <: Real}
        constituent_num_points = map(length, constituent_points)
        cart_num_points = CartesianIndices(
            ntuple(dim -> constituent_num_points[dim], manifold_dim)
        )
        lin_num_points = LinearIndices(cart_num_points)
        inv_iteration_order = invperm(iteration_order)
        permuted_cart_num_points = CartesianIndices(
            ntuple(dim -> constituent_num_points[inv_iteration_order[dim]], manifold_dim)
        )

        return new{
            manifold_dim,
            T,
            typeof(constituent_points),
            typeof(cart_num_points),
            typeof(lin_num_points),
        }(
            constituent_points,
            cart_num_points,
            lin_num_points,
            iteration_order,
            permuted_cart_num_points,
        )
    end
end

function CartesianPoints(
    constituent_points::NTuple{manifold_dim, V};
    iteration_order::NTuple{manifold_dim, Int}=ntuple(k -> k, manifold_dim),
) where {manifold_dim, V <: AbstractVector{<:Real}}
    return CartesianPoints(constituent_points...; iteration_order)
end

function CartesianPoints(constituent_points...; kwargs...)
    return CartesianPoints(promote(map(collect, constituent_points)...)...; kwargs...)
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
	get_iteration_order(points::CartesianPoints)

Returns the `iteration_order` order used to index `points`.
"""
get_iteration_order(points::CartesianPoints) = points.iteration_order

"""
	get_permuted_cart_num_points(points::CartesianPoints)

Returns the permuted `cart_num_points` used to index `points`, as given by
`iteration_order`.
"""
get_permuted_cart_num_points(points::CartesianPoints) = points.permuted_cart_num_points

"""
    get_constituent_num_points(points::CartesianPoints)

Returns the number of constituent points per manifold dimension.
"""
function get_constituent_num_points(points::CartesianPoints)
    return size(get_cart_num_points(points))
end

function Base.getindex(points::CartesianPoints{manifold_dim}, i::Int) where {manifold_dim}
    unordered_point = get_permuted_cart_num_points(points)[i]
    iteration_order = get_iteration_order(points)
    ordered_point = ntuple(dim -> unordered_point[iteration_order[dim]], manifold_dim)

    return getindex(points, ordered_point)
end

function Base.getindex(
    points::CartesianPoints{manifold_dim},
    i::Union{NTuple{manifold_dim}, CartesianIndex{manifold_dim}},
) where {manifold_dim}
    const_points = get_constituent_points(points)

    return ntuple(dim -> const_points[dim][i[dim]], manifold_dim)
end

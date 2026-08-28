"""
    TensorProductPoints{manifold_dim, T, TP, CI} <: AbstractPoints{manifold_dim, T}

Represents a set of points tensored from `manifold_dim` lists of uni-dimensional points.

See also: [`TensorProducts.TensorProduct`](@ref).

# Constructors
- `TensorProductPoints(
        input_points::Vararg{AbstractVector{<:Number}, manifold_dim};
        iteration_order::NTuple{manifold_dim, Int}=ntuple(k -> k, manifold_dim),
    ) where {manifold_dim}`: General constructor. Performs validity checks on the given
    points and iteration order.
- `TensorProductPoints(input_points::Tuple; kwargs...)`: Convenience constructor that will
    splat the given tuple and keyword arguments.

# Fields
- `tensor_product::TP`: The set of points per manifold dimension.
- `iteration_order::NTuple{manifold_dim, Int}`: Used to determine the iteration order over
    the tensored points. If the `k`-th entry has value `i`, then dimension `k` will be the
    `i`-th fastest changing index.
- `permuted_cart_num_points::CI`: A permuted version of tensored points, as given by
    `iteration_order`.

# Examples
```jldoctest
julia> using Mantis

julia> points = Points.TensorProductPoints([1,2], [1,2,3]; iteration_order=(1, 2));

julia> for point in points
           println(point)
       end
(1, 1)
(2, 1)
(1, 2)
(2, 2)
(1, 3)
(2, 3)

julia> points = Points.TensorProductPoints([1,2], [1,2,3]; iteration_order=(2, 1));

julia> for point in points
           println(point)
       end
(1, 1)
(1, 2)
(1, 3)
(2, 1)
(2, 2)
(2, 3)
```
"""
struct TensorProductPoints{manifold_dim, T, TP, CI} <: AbstractPoints{manifold_dim, T}
    tensor_product::TP
    iteration_order::NTuple{manifold_dim, Int}
    permuted_cart_num_points::CI

    function TensorProductPoints(
        input_points::Vararg{AbstractVector{<:Number}, manifold_dim};
        iteration_order::NTuple{manifold_dim, Int}=ntuple(k -> k, manifold_dim),
    ) where {manifold_dim}
        _construction_checks(input_points)

        if any(i -> i<1 || i>manifold_dim, iteration_order)
            return throw(
                ArgumentError(
                    LazyString(
                        "The iteration order must range from 1 to ",
                        manifold_dim,
                        ". Got ",
                        iteration_order,
                    ),
                ),
            )
        end

        if !(allunique(iteration_order))
            return throw(
                ArgumentError(
                    LazyString(
                        "The iteration order must contain only unique entries. Got ",
                        iteration_order,
                    ),
                ),
            )
        end

        input_num_points = map(length, input_points)
        types = map(eltype, input_points)
        T = promote_type(types...)
        promoted_points = ntuple(manifold_dim) do k
            if types[k] != T
                return convert.(T, input_points[k])
            end

            return input_points[k]
        end

        tensor_product = TensorProducts.TensorProduct(promoted_points)
        inv_iteration_order = invperm(iteration_order)
        permuted_cart_num_points = CartesianIndices(
            ntuple(dim -> input_num_points[inv_iteration_order[dim]], manifold_dim)
        )

        return new{
            manifold_dim, T, typeof(tensor_product), typeof(permuted_cart_num_points)
        }(
            tensor_product, iteration_order, permuted_cart_num_points
        )
    end
end

function TensorProductPoints(input_points::Tuple; kwargs...)
    return TensorProductPoints(input_points...; kwargs...)
end

get_tensor_product(points::TensorProductPoints) = points.tensor_product

"""
    get_factor_points(points::TensorProductPoints)

Equivalent to [`get_input_points`](@ref), but provides a consistent interface with
[`TensorProducts`](@ref).
"""
get_factor_points(points::TensorProductPoints) = get_input_points(points)

"""
    get_cart_num_points(points::TensorProductPoints)

Return the `CartesianIndices` used to convert from linear to cartesian indexing.
"""
function get_cart_num_points(points::TensorProductPoints)
    return TensorProducts.get_cart_ids(get_tensor_product(points))
end

function get_num_points(points::TensorProductPoints)
    return TensorProducts.get_num_objects(get_tensor_product(points))
end

"""
    get_lin_num_points(points::TensorProductPoints)

Returns the `LinearIndices` used to convert from cartesian to linear indexing.
"""
function get_lin_num_points(points::TensorProductPoints)
    return TensorProducts.get_lin_ids(get_tensor_product(points))
end

function get_input_points(points::TensorProductPoints)
    return TensorProducts.get_factors(get_tensor_product(points))
end

"""
	get_iteration_order(points::TensorProductPoints)

Returns the `iteration_order` order used to index `points`.
"""
get_iteration_order(points::TensorProductPoints) = points.iteration_order

"""
	get_permuted_cart_num_points(points::TensorProductPoints)

Returns the permuted tensored points used to index `points`, as given by iteration order.
"""
get_permuted_cart_num_points(points::TensorProductPoints) = points.permuted_cart_num_points

"""
    get_factor_num_points(points::TensorProductPoints)

Returns the number of input points per manifold dimension.
"""
get_factor_num_points(points::TensorProductPoints) = map(length, get_input_points(points))

function Base.getindex(
    points::TensorProductPoints{manifold_dim}, i::Int
) where {manifold_dim}
    unordered_point = get_permuted_cart_num_points(points)[i]
    iteration_order = get_iteration_order(points)
    ordered_point = ntuple(dim -> unordered_point[iteration_order[dim]], manifold_dim)

    return getindex(points, ordered_point)
end

function Base.getindex(
    points::TensorProductPoints{manifold_dim},
    i::Union{NTuple{manifold_dim}, CartesianIndex{manifold_dim}},
) where {manifold_dim}
    input_points = get_input_points(points)

    return ntuple(dim -> input_points[dim][i[dim]], manifold_dim)
end

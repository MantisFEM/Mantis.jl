"""
    PointSet{manifold_dim, T, P} <: AbstractPoints{manifold_dim, T}

Represents an unstructured set of points in `manifold_dim` dimensions.

# Constructors
- `PointSet(
       input_points::Vararg{AbstractVector{<:Number}, manifold_dim}
    ) where {manifold_dim}`: General constructor. Performs validity checks on the given
    points.
- `PointSet(input_points::Tuple)`: Convenience constructor that will splat the given tuple.

# Fields
- `input_points::P`: A list of points per manifold dimension. The entry `inputs[k][i]` gives
    the `k`-th coordinate of point `i`.
- `num_points::Int`: The total number of points.

# Examples
```jldoctest
julia> using Mantis

julia> points = Points.PointSet([1, 2, 3], [3, 2, 1]);

julia> for point in points
           println(point)
       end
(1, 3)
(2, 2)
(3, 1)

julia> points = Points.PointSet(LinRange(0, 1, 3), [1, 2, 3]);

julia> for point in points
           println(point)
       end
(0.0, 1.0)
(0.5, 2.0)
(1.0, 3.0)
```
"""
struct PointSet{manifold_dim, T, P} <: AbstractPoints{manifold_dim, T}
    input_points::P
    num_points::Int

    function PointSet(
        input_points::Vararg{AbstractVector{<:Number}, manifold_dim}
    ) where {manifold_dim}
        _construction_checks(input_points)

        input_num_points = map(length, input_points)
        allequal(input_num_points) ||
            throw(ArgumentError("Number of points in each dimension must match."))
        num_points = first(input_num_points)
        types = map(eltype, input_points)
        T = promote_type(types...)
        # all input points are promoted promoted to the same type.
        promoted_points = ntuple(manifold_dim) do k
            if types[k] != T
                return convert.(T, input_points[k])
            end

            return input_points[k]
        end

        return new{manifold_dim, T, typeof(promoted_points)}(promoted_points, num_points)
    end
end

PointSet(input_points::Tuple) = PointSet(input_points...)

function PointSet(point_set::Vector{NTuple{manifold_dim, T}}) where {manifold_dim, T}
    # Split a list of n manifold_dim-dimensional points into manifold_dim lists of n points
    num_points = length(point_set)
    input_points = ntuple(
        dim -> [point_set[point][dim] for point in 1:num_points], manifold_dim
    )

    return PointSet(input_points...)
end

function PointSet(point_set::Vector{Vector{T}}) where {T <: Number}
    # Split a list of n manifold_dim-dimensional points into manifold_dim lists of n points
    manifold_dim = length(first(point_set))
    input_points = ntuple(
        dim -> [point_set[point][dim] for point in 1:length(point_set)], manifold_dim
    )

    return PointSet(input_points...)
end

get_num_points(points::PointSet) = points.num_points

function Base.getindex(points::PointSet{manifold_dim}, i::Int) where {manifold_dim}
    return ntuple(dim -> get_input_points(points)[dim][i], manifold_dim)
end

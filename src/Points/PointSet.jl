"""
    PointSet{manifold_dim, T} <: AbstractPoints{manifold_dim, T}

Represents a set of points in `manifold_dim` dimensions.

# Fields
- `input_points::NTuple{manifold_dim, T}`: The set of points per manifold dimension.
"""
struct PointSet{manifold_dim, T, CP} <: AbstractPoints{manifold_dim, T}
    input_points::CP
    num_points::Int

    function PointSet(
        input_points::Vararg{AbstractVector, manifold_dim}
    ) where {manifold_dim}
        iszero(manifold_dim) && throw(ArgumentError("Empty argument tuple."))
        input_num_points = map(length, input_points)
        any(iszero, input_num_points) &&
            throw(ArgumentError("Number of points must be non-empty."))
        allequal(input_num_points) ||
            throw(ArgumentError("Number of points in each dimension must match."))
        num_points = first(input_num_points)
        types = map(eltype, input_points)
        T = promote_type(types...)
        T <: Number || throw(ArgumentError("Points must be a subtype of `Number`."))
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

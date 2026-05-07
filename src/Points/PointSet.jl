"""
    PointSet{manifold_dim, T} <: AbstractPoints{manifold_dim, T}

Represents a set of points in `manifold_dim` dimensions.

# Fields
- `constituent_points::NTuple{manifold_dim, T}`: The set of points per manifold dimension.
"""
struct PointSet{manifold_dim, T, CP} <: AbstractPoints{manifold_dim, T}
    constituent_points::CP
    num_points::Int

    function PointSet(
        constituent_points::Vararg{AbstractVector{T}, manifold_dim}
    ) where {manifold_dim, T <: Real}
        allequal(length, constituent_points) ||
            throw(ArgumentError("Number of points in each dimension must match."))
        num_points = length(constituent_points[1])

        return new{manifold_dim, T, typeof(constituent_points)}(
            constituent_points, num_points
        )
    end
end

function PointSet(
    constituent_points::NTuple{manifold_dim, V}
) where {manifold_dim, V <: AbstractVector{<:Real}}
    return PointSet(constituent_points...)
end

PointSet(constituent_points...) = PointSet(promote(map(collect, constituent_points)...)...)

function PointSet(point_set::Vector{NTuple{manifold_dim, T}}) where {manifold_dim, T}
    # Split a list of n manifold_dim-dimensional points into manifold_dim lists of n points
    num_points = length(point_set)
    constituent_points = ntuple(
        dim -> [point_set[point][dim] for point in 1:num_points], manifold_dim
    )

    return PointSet(constituent_points)
end

function PointSet(point_set::Vector{Vector{T}}) where {T <: Real}
    # Split a list of n manifold_dim-dimensional points into manifold_dim lists of n points
    manifold_dim = length(first(point_set))
    constituent_points = ntuple(
        dim -> [point_set[point][dim] for point in 1:length(point_set)], manifold_dim
    )

    return PointSet(constituent_points)
end

get_num_points(points::PointSet) = points.num_points

function Base.getindex(points::PointSet{manifold_dim}, i::Int) where {manifold_dim}
    return ntuple(dim -> get_constituent_points(points)[dim][i], manifold_dim)
end

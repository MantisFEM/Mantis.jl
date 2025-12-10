"""
    PointSet{manifold_dim, T} <: AbstractPoints{manifold_dim}

Represents a set of points in `manifold_dim` dimensions.

# Fields
- `constituent_points::NTuple{manifold_dim, T}`: The set of points per manifold dimension.
"""
struct PointSet{manifold_dim, T} <: AbstractPoints{manifold_dim}
    constituent_points::NTuple{manifold_dim, T}
    num_points::Int

    function PointSet(
        constituent_points::NTuple{manifold_dim, T}
    ) where {manifold_dim, T <: AbstractVector}
        num_points = length(constituent_points[1])

        return new{manifold_dim, T}(constituent_points, num_points)
    end

    function PointSet(point_set::Vector{NTuple{manifold_dim, T}}) where {manifold_dim, T}
        constituent_points = ntuple(
            dim -> [point_set[point][dim] for point in 1:length(point_set)], manifold_dim
        )

        return PointSet(constituent_points)
    end

    function PointSet(point_set::Vector{Vector{T}}) where {T <: Number}
        manifold_dim = length(point_set[1])
        constituent_points = ntuple(
            dim -> [point_set[point][dim] for point in 1:length(point_set)], manifold_dim
        )

        return PointSet(constituent_points)
    end
end

Base.eltype(::PointSet{manifold_dim, T}) where {manifold_dim, T} = T

get_num_points(points::PointSet) = points.num_points

function Base.getindex(points::PointSet{manifold_dim}, i::Int) where {manifold_dim}
    return ntuple(dim -> get_constituent_points(points)[dim][i], manifold_dim)
end

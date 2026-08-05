"""
    module Points

Contains all definitions of points used to evaluate geometries, function spaces, forms and
any other objects.
"""
module Points

using ..TensorProducts

############################################################################################
#                                      Abstract Types                                      #
############################################################################################

"""
    AbstractPoints{manifold_dim, T}

Supertype for all evaluable points.

# Type parameters
- `manifold_dim`: Dimension of the manifold where the points are evaluated.
- `T`: The `eltype` of the points; see `Base.eltype`.
"""
abstract type AbstractPoints{manifold_dim, T} end

############################################################################################
#                                    Abstract Methods                                      #
############################################################################################

"""
    get_manifold_dim(points::AbstractPoints{manifold_dim}) where {manifold_dim}

Returns the manifold dimension of the evaluable `points`.
"""
get_manifold_dim(::AbstractPoints{manifold_dim}) where {manifold_dim} = manifold_dim

"""
    get_num_points(points::P) where {manifold_dim, P <: AbstractPoints{manifold_dim}}

Returns the number of of evaluable `points` in the given point structure.
"""
function get_num_points(::P) where {P <: AbstractPoints}
    throw(MethodError(get_num_points, (P,)))
end

"""
    get_input_points(points::P) where {P <: AbstractPoints}

Returns the points given as input when constructing `points`, up to a _possible_ type
promotion.
"""
function get_input_points(points::P) where {P <: AbstractPoints}
    return points.input_points
end

"""
	scale_and_shift_points(
	    points::P, scalings::S, translations::T
	) where {
	    manifold_dim,
	    P <: AbstractPoints{manifold_dim},
	    S <: NTuple{manifold_dim, Real},
	    T <: NTuple{manifold_dim, Real},
	}

Applies an affine map defined by `scalings` and `translations` to each point in `points`. 

# Arguments
- `points::P`: The set of points.
- `scalings::S`: The scaling of the affine map.
- `translations::T`: The translation of the affine map.

# Returns
- `transformed_points::P`: The set of transformed points of the same type as the original
	`points`.
"""
function scale_and_shift_points(
    points::P, scalings::S, translations::T
) where {
    manifold_dim,
    P <: AbstractPoints{manifold_dim},
    S <: NTuple{manifold_dim, Real},
    T <: NTuple{manifold_dim, Real},
}
    input_points = get_input_points(points)
    transformed_points = ntuple(
        dim -> input_points[dim] .* scalings[dim] .+ translations[dim], manifold_dim
    )
    constructor = Base.typename(P).wrapper

    return constructor(transformed_points)
end

Base.eltype(::AbstractPoints{manifold_dim, T}) where {manifold_dim, T} = T
Base.firstindex(points::AbstractPoints) = 1
Base.lastindex(points::AbstractPoints) = get_num_points(points)
Base.keys(points::AbstractPoints) = firstindex(points):lastindex(points)
Base.length(points::AbstractPoints) = get_num_points(points)

function Base.iterate(points::AbstractPoints)
    i = firstindex(points)

    return getindex(points, i), i
end

function Base.iterate(points::AbstractPoints, i::Int)
    i += 1
    i > lastindex(points) && return nothing

    return getindex(points, i), i
end

############################################################################################
#                                         Includes                                         #
############################################################################################

include("./TensorProductPoints.jl")
include("./PointSet.jl")

end

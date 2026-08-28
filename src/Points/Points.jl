"""
    module Points

Provides collections of coordinates in either canonical, parametric, or physical domains, in
a standardised format.

This serves as an abstraction layer that is used to create evaluation points in several
downstream modules, such as [Geometry](@ref DocGeometryModule), [FunctionSpaces](@ref), or
[Forms](@ref).
"""
module Points

using ..TensorProducts

############################################################################################
#                                         Exports                                          #
############################################################################################

# Abstract types
# The public keyword is only available in Julia 1.11 and up. Since we also support LTS
# (currently 1.10), we add the following line from the manual:
# https://docs.julialang.org/en/v1.12/manual/modules/#Export-lists
VERSION >= v"1.11.0-DEV.469" && eval(Meta.parse("public AbstractPoints"))

export PointSet,
    TensorProductPoints,
    get_cart_num_points,
    get_input_points,
    get_lin_num_points,
    get_manifold_dim,
    get_num_points

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

function _construction_checks(
    input_points::NTuple{manifold_dim, AbstractVector{<:Number}}
) where {manifold_dim}
    iszero(manifold_dim) && throw(ArgumentError("Empty argument tuple."))
    input_num_points = map(length, input_points)
    any(iszero, input_num_points) &&
        throw(ArgumentError("Number of points must be non-empty."))

    return nothing
end

"""
    get_manifold_dim(points::AbstractPoints{manifold_dim}) where {manifold_dim}

Returns the manifold dimension of the evaluable `points`.
"""
get_manifold_dim(::AbstractPoints{manifold_dim}) where {manifold_dim} = manifold_dim

"""
    get_num_points(points::P) where {manifold_dim, P <: AbstractPoints{manifold_dim}}

Returns the number of evaluable points in the given point structure.
"""
function get_num_points(::P) where {P <: AbstractPoints}
    return throw(MethodError(get_num_points, (P,)))
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
    transformed_points = map((p, s, t) -> p .* s .+ t, input_points, scalings, translations)
    constructor = Base.typename(P).wrapper

    return constructor(transformed_points)
end

Base.eltype(::AbstractPoints{manifold_dim, T}) where {manifold_dim, T} = T
Base.firstindex(::AbstractPoints) = 1
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

include("TensorProductPoints.jl")
include("PointSet.jl")

end

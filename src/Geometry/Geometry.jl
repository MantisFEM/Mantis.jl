"""
    module Geometry

Contains all geometry definitions.
"""
module Geometry 

abstract type AbstractGeometry end


struct AnalGeometry{M, D} <: AbstractGeometry where {M <: Function, D <: Function}
    map::M
    dmap::D
end

function evaluate(geometry::AnalGeometry{M, D}, element_idx::Int, ξ::Vector{Float64}) where {M <: Function, D <: Function}
    return geometry.map(element_idx, ξ)
end

end
"""
    module Geometry

Contains all geometry definitions.
"""
module Geometry 

abstract type AbstractGeometry end
abstract type AbstractAnalGeometry <: AbstractGeometry end


include("./MappedGeometry.jl")
include("./CompositeGeometry.jl")
include("./RectangleGeometry.jl")
include("./MappedRectangleGeometry.jl")


end
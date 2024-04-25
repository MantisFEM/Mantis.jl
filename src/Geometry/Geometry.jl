"""
    module Geometry

Contains all geometry definitions.
"""
module Geometry 

using .. FunctionSpaces

abstract type AbstractGeometry end
abstract type AbstractAnalGeometry <: AbstractGeometry end
abstract type AbstractFEMGeometry <: AbstractGeometry end

include("./FEMGeometry.jl")
include("./MappedGeometry.jl")
include("./CompositeGeometry.jl")
include("./CartesianGeometry.jl")
include("./RectangleGeometry.jl")
include("./MappedRectangleGeometry.jl")

end
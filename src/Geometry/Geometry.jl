"""
    module Geometry

Contains all geometry definitions.
"""
module Geometry 

using .. FunctionSpaces

abstract type AbstractGeometry{n, m} end
abstract type AbstractAnalGeometry{n, m} <: AbstractGeometry{n, m} end
abstract type AbstractFEMGeometry{n, m} <: AbstractGeometry{n, m} end

include("./CartesianGeometry.jl")
include("./FEMGeometry.jl")
include("./MappedGeometry.jl")
include("./TensorProductGeometry.jl")

end
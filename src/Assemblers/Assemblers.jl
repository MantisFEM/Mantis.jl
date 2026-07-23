"""
    module Assemblers

Contains all assembly-related structs and functions.
"""
module Assemblers

import LinearAlgebra
import SparseArrays;
const spa = SparseArrays

using ..Geometry
using ..Forms
using ..Quadrature
using ..Mesh
using ..FunctionSpaces
using ..Analysis

abstract type AbstractInputs end

include("WeakFormulations/WeakFormulations.jl")
include("GlobalAssemblers.jl")

end

"""
This (sub-)module provides a collection of function spaces.
The exported names are:
"""
module FunctionSpaces

import .. Mesh
import SparseArrays

"""
    AbstractFunctionSpace

Supertype for all function spaces.
"""
abstract type AbstractFunctionSpace end

include("CanonicalSpaces/CanonicalSpaces.jl")  # Creates Module CanonicalSpaces
include("FiniteElementSpaces/FiniteElementSpaces.jl")  # Creates Module FiniteElementSpaces

end
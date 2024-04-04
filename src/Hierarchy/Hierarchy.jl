"""
This (sub-)module provides functionality related with hierarchical refinment.

The exported names are:
"""

module Hierarchy

import .. FiniteElementSpaces
import .. Mesh
import LinearAlgebra: I

include("TwoScale.jl")
include("KnotInsertion.jl")

end
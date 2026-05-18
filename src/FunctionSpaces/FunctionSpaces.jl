"""
This (sub-)module provides a collection of function spaces.
The exported names are:
"""
module FunctionSpaces

import Combinatorics
import Graphs
import LinearAlgebra
import Memoization
import SparseArrays
import ToeplitzMatrices

using ..Geometry
using ..Points
using ..GeneralHelpers
using ..Hierarchy

"""
    AbstractFunctionSpace

Supertype for all function spaces.
"""
abstract type AbstractFunctionSpace end

"""
    AbstractCanonicalSpace <: AbstractFunctionSpace

Supertype for all element-local bases. These spaces are only defined in the canonical domain
and are used to define the shape functions on the reference element.
"""
abstract type AbstractCanonicalSpace <: AbstractFunctionSpace end

abstract type AbstractECTSpaces <: AbstractCanonicalSpace end
abstract type AbstractLagrangePolynomials <: AbstractCanonicalSpace end
abstract type AbstractEdgePolynomials <: AbstractCanonicalSpace end

include("CanonicalSpaces/CanonicalSpaces.jl")

"""
    AbstractFESpace{manifold_dim, num_components, num_patches} <: AbstractFunctionSpace

Supertype for all finite element spaces. These can be of any dimension, with any number of
components, and on any number of patches.

# Type parameters
- `manifold_dim::Int`: Dimension of the manifold.
- `num_components::Int`: Number of (output) components of the function space.
- `num_patches::Int`: Number of patches over which the function space is defined.
"""
abstract type AbstractFESpace{manifold_dim, num_components, num_patches} <:
              AbstractFunctionSpace end

function get_manifold_dim(
    ::AbstractFESpace{manifold_dim, num_components, num_patches}
) where {manifold_dim, num_components, num_patches}
    return manifold_dim
end
function get_num_components(
    ::AbstractFESpace{manifold_dim, num_components, num_patches}
) where {manifold_dim, num_components, num_patches}
    return num_components
end
function get_num_patches(
    ::AbstractFESpace{manifold_dim, num_components, num_patches}
) where {manifold_dim, num_components, num_patches}
    return num_patches
end

include("FiniteElementSpaces/FiniteElementSpaces.jl")
include("AdaptiveRefinement/AdaptiveRefinement.jl")

# helper functions for convenience
include("./FunctionSpaceHelpers.jl")

end

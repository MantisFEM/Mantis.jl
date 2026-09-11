"""
    module Quadrature

Provides quadrature rules, i.e. sets of nodes and weights that approximate integrals by
weighted sums.

Rules are defined on the canonical domain ``[0, 1]^n``, matching the convention used by
[Points](@ref DocPointsModule) and [Geometry](@ref DocGeometryModule), so that a single rule
can be reused on every element of a mesh. The module distinguishes rules on a single element
from rules covering a whole domain, and is used by [Forms](@ref) to evaluate integrals.
"""
module Quadrature

import FastGaussQuadrature
import FFTW

import ..Points
import ..Geometry

"""
    AbstractQuadratureRule{manifold_dim}

Abstract type for a quadrature rule on an entire domain of dimension `manifold_dim`.

# Type parameters
- `manifold_dim`: Dimension of the domain
"""
abstract type AbstractQuadratureRule{manifold_dim} end

"""
    AbstractElementQuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim}

A quadrature rule on a single quadrature element.
"""
abstract type AbstractElementQuadratureRule{manifold_dim} <:
              AbstractQuadratureRule{manifold_dim} end

"""
    AbstractGlobalQuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim}

A quadrature rule on a global domain.
"""
abstract type AbstractGlobalQuadratureRule{manifold_dim} <:
              AbstractQuadratureRule{manifold_dim} end

include("ElementQuadratureRules/ElementQuadratureRules.jl")
include("GlobalQuadratureRules/GlobalQuadratureRules.jl")

include("QuadratureHelpers.jl")

end

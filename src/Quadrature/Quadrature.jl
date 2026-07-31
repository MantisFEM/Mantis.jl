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

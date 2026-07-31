"""
    get_nodes(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}

Returns the quadrature nodes of a quadrature rule.

# Arguments

  - `qr::AbstractElementQuadratureRule{manifold_dim}`: Rule to get the nodes from.

# Returns

  - `nodes::NTuple{manifold_dim, Vector{Float64}}`: Nodes of the quadrature rule.
"""
function get_nodes(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.nodes
end

"""
    get_weights(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}

Returns the quadrature weights of a quadrature rule.

# Arguments

  - `qr::AbstractElementQuadratureRule{manifold_dim}`: Rule to get the weights from.

# Returns

  - `weights::Vector{Float64}`: Weights of the quadrature rule.
"""
function get_weights(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.weights
end

"""
    get_label(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}

Returns the label of a quadrature rule.

# Arguments

  - `qr::AbstractElementQuadratureRule{manifold_dim}`: Rule to get the label from.

# Returns

  - `rule_label::String`: Label of the quadrature rule.
"""
function get_label(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.rule_label
end

"""
    CanonicalQuadratureRule{manifold_dim} <: ElementQuadratureRule{manifold_dim}

Represents a quadrature rule on a canonical element of dimension `manifold_dim`.

# Fields

  - `nodes::NTuple{manifold_dim, Vector{Float64}}`: Quadrature nodes per dimension.
  - `weights::Vector{Float64}`: Tensor product of quadrature rules. The shape is consistent
    with the output of the evaluate methods for `FunctionSpaces`.
  - `rule_label::String`: Name or type of quadrature rule. Used for human verification.

# Type parameters

  - `manifold_dim`: Dimension of the domain

# Inner Constructors

  - `CanonicalQuadratureRule(nodes::NTuple{manifold_dim, Vector{Float64}}, weights::Vector{Float64})`:
    General constructor.

# Outer Constructors

  - [`gauss_lobatto`](@ref).
  - [`gauss_legendre`](@ref).
  - [`clenshaw_curtis`](@ref).
  - [`newton_cotes`](@ref).
  - [`tensor_product_rule(p::NTuple{manifold_dim, Integer}, quad_rule::F, rule_args_1d...) where {manifold_dim, F <: Function}`](@ref).
  - [`tensor_product_rule(qrules_1d::NTuple{manifold_dim, CanonicalQuadratureRule{1}}) where {manifold_dim}`](@ref).
"""
struct CanonicalQuadratureRule{manifold_dim, P} <:
       AbstractElementQuadratureRule{manifold_dim}
    nodes::P
    weights::Vector{Float64}
    rule_label::String

    function CanonicalQuadratureRule(
        nodes::P, weights::Vector{Float64}, rule_label::String="canonical points"
    ) where {manifold_dim, P <: Points.AbstractPoints{manifold_dim}}
        return new{manifold_dim, P}(nodes, weights, rule_label)
    end
end

# One-dimensional quadrature rules.
# Quadrature rules on non-equally spaced nodes.
include("./Gauss.jl")
include("./ClenshawCurtis.jl")
# Quadrature rules on equally spaced nodes.
include("./NewtonCotes.jl")

# Multi-dimensional quadrature rules. The tensor product rules are made by combining one-
# dimensional rules, so this file must be included after the one-dimensional rules.
include("./TensorProduct.jl")

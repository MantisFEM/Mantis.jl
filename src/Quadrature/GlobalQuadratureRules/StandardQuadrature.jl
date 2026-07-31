"""
    StandardQuadrature{manifold_dim, Q} <: AbstractGlobalQuadratureRule{manifold_dim}

Represents a standard quadrature rule for a given manifold dimension, where each element has
the same canonical rule `Q`.

# Fields

  - `canonical_qrule::Q`: The canonical quadrature rule used for the elements.
  - `num_elements::Int`: The number of elements in the quadrature rule.

# Type parameters

  - `manifold_dim`: Dimension of the domain
  - `Q`: Type of the canonical quadrature rule.

# Inner Constructors

  - `StandardQuadrature(canonical_qrule::Q, num_elements::Int)`: Creates a new
    `StandardQuadrature` instance with the specified canonical quadrature rule and number of
    elements.
"""
struct StandardQuadrature{manifold_dim, Q} <: AbstractGlobalQuadratureRule{manifold_dim}
    canonical_qrule::Q
    num_elements::Int

    function StandardQuadrature(
        canonical_qrule::Q, num_elements::Int
    ) where {manifold_dim, Q <: CanonicalQuadratureRule{manifold_dim}}
        return new{manifold_dim, Q}(canonical_qrule, num_elements)
    end
end

"""
    get_canonical_quadrature_rule(standard_quadrature::StandardQuadrature)

Get the canonical quadrature rule of the standard quadrature.

# Arguments

  - `standard_quadrature::StandardQuadrature`: The standard quadrature rule.

# Returns

  - `CanonicalQuadratureRule`: The canonical quadrature rule of the standard quadrature.
"""
function get_canonical_quadrature_rule(standard_quadrature::StandardQuadrature)
    return standard_quadrature.canonical_qrule
end

function get_num_evaluation_elements(standard_quadrature::StandardQuadrature)
    return standard_quadrature.num_elements
end

"""
    get_num_base_elements(standard_quadrature::StandardQuadrature)

Returns the number of base elements in the standard quadrature.

# Arguments

  - `standard_quadrature::StandardQuadrature`: The standard quadrature rule.

# Returns

  - `::Int`: The number of base elements in the standard quadrature.
"""
function get_num_base_elements(standard_quadrature::StandardQuadrature)
    return standard_quadrature.num_elements
end

function get_element_idxs(::StandardQuadrature, element_idx::Int)
    return [element_idx]
end

function get_element_quadrature_rule(standard_quadrature::StandardQuadrature, ::Int)
    return get_canonical_quadrature_rule(standard_quadrature)
end

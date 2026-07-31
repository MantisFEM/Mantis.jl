"""
    get_canonical_quadrature_rules(
        q_rule::Function,
        nq_single::NTuple{manifold_dim, Int},
        nq_others::NTuple{manifold_dim, Int}...,
    ) where {manifold_dim}

Returns a tuple of tensor-product quadrature rules, of the type `q_rule`, for the given
number of quadrature points in each dimension.

# Arguments

  - `q_rule::Function`: The type of univariate quadrature rule to use.
  - `nq_single::NTuple{manifold_dim, Int}`: Number of quadrature points per dimension for the
    first quadrature rule.
  - `nq_others::NTuple{manifold_dim, Int}...`: Number of quadrature points per dimension for the
    other quadrature rules.

# Returns

  - `::NTuple{num_rules, CanonicalQuadratureRule{manifold_dim}}`: A tuple of quadrature rules where
    `num_rules` is the number of other quadrature rules given plus the single rule.
"""
function get_canonical_quadrature_rules(
    q_rule::Function,
    nq_single::NTuple{manifold_dim, Int},
    nq_others::NTuple{manifold_dim, Int}...,
) where {manifold_dim}
    q_single = tensor_product_rule(nq_single, q_rule)
    q_others = ntuple(length(nq_others)) do i
        return tensor_product_rule(nq_others[i], q_rule)
    end

    return (q_single, q_others...)
end

function get_global_quadrature_rules(
    q_rule::Function,
    num_elements::Int,
    nq_single::NTuple{manifold_dim, Int},
    nq_others::NTuple{manifold_dim, Int}...,
) where {manifold_dim}
    q_single = StandardQuadrature(tensor_product_rule(nq_single, q_rule), num_elements)
    q_others = ntuple(length(nq_others)) do i
        return StandardQuadrature(tensor_product_rule(nq_others[i], q_rule), num_elements)
    end

    return (q_single, q_others...)
end

"""
    tensor_product_rule(p::NTuple{manifold_dim, Integer}, quad_rule::F, rule_args_1d...) where {
        manifold_dim, F <: Function
    }

Returns a tensor product quadrature rule of given degree and rule type.

# Arguments

  - `p::NTuple{manifold_dim, Integer}`: Degree of the quadrature rule per dimension.
  - `quad_rule::F`: The function that returns a `CanonicalQuadratureRule{1}` given an integer degree.
    May take additional arguments.
  - `rule_args_1d...`: Additional arguments for the 1D quadrature rule. Optional.

# Returns

  - `::CanonicalQuadratureRule{manifold_dim}`: CanonicalQuadratureRule of the new dimension.
"""
function tensor_product_rule(
    p::NTuple{manifold_dim, Integer}, quad_rule::F, rule_args_1d...
) where {manifold_dim, F <: Function}
    const_qrules = NTuple{manifold_dim, CanonicalQuadratureRule{1}}(
        quad_rule(p[k], rule_args_1d...) for k in 1:manifold_dim
    )

    return tensor_product_rule(const_qrules)
end

"""
    tensor_product_rule(qrules_1d::NTuple{manifold_dim, CanonicalQuadratureRule{1}}) where {
        manifold_dim
    }

Returns a tensor product quadrature rule from the given rules.

# Arguments

  - `qrules_1d::NTuple{manifold_dim, CanonicalQuadratureRule{1}}`: Quadrature rules per dimension.

# Returns

  - `::CanonicalQuadratureRule{manifold_dim}`: CanonicalQuadratureRule of the new dimension.
"""
function tensor_product_rule(
    const_qrules::NTuple{manifold_dim, CanonicalQuadratureRule{1}}
) where {manifold_dim}
    const_nodes = ntuple(
        dim -> Points.get_constituent_points(get_nodes(const_qrules[dim]))[1], manifold_dim
    )
    nodes = Points.CartesianPoints(const_nodes)
    const_weights = NTuple{manifold_dim, Vector{Float64}}(
        get_weights(const_qrules[k]) for k in 1:manifold_dim
    )
    weights = _compute_tensor_product(const_weights)

    # If only one rule is used, the label is the same as the label of the 1D rule. This
    # ensures that the tensor product of a 1d rule is the same as the 1d rule.
    if manifold_dim == 1
        rule_label = get_label(const_qrules[1])
    else
        rule_labels = join([get_label(const_qrules[i]) for i in 1:manifold_dim], ", ")
        rule_label = "Tensor-product of ($rule_labels) rules"
    end

    return CanonicalQuadratureRule(nodes, weights, rule_label)
end

"""
    _compute_tensor_product(weights_1d::NTuple{manifold_dim, Vector{T}}) where {
        manifold_dim, T <: Number
    }

Compute the tensor product of the given 1D quadrature weights.

# Arguments

  - `weights_1d::NTuple{manifold_dim, Vector{T}}`: Quadrature weights per dimension.

# Returns

  - `::Vector{T}`: Tensor product of the quadrature weights.
"""
function _compute_tensor_product(
    weights_1d::NTuple{manifold_dim, Vector{T}}
) where {manifold_dim, T <: Number}
    weights = Vector{T}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end

    return weights
end

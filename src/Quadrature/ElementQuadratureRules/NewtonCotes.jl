"""
    newton_cotes(num_points::Integer, type::String="closed")

Computes the nodes `ξ` and weights `w` of a Newton-Cotes quadrature rule.

Newton-Cotes quadrature rules are based on equally spaced nodes. There are two types of
Newton-Cotes rules: closed and open. Closed Newton-Cotes rules include the endpoints of the
interval, while open Newton-Cotes rules do not.

The algorithm used to compute the weights is not the most efficient nor the most accurate.

# Arguments

  - `num_points::Integer`: Number of points in the quadrature rule.
  - `type::String`: Type of the Newton-Cotes rule. Valid types are "closed" and "open".

# Returns

  - `::CanonicalQuadratureRule{1}`: 1 dimensional quadrature rule containing the nodes and weights.
"""
function newton_cotes(num_points::Integer, type::String)
    # Compute the equally spaced nodes on the interval [-1, 1].
    if type == "closed"
        if num_points <= 1
            throw(
                DomainError("""\
              Invalid number of points: $num_points. A closed Newton-Cotes rule requires \
              at least 2 points.\
              """)
            )
        else
            ξ = [-1.0 + i * 2.0 / (num_points - 1) for i in 0:(num_points - 1)]
        end

    elseif type == "open"
        if num_points <= 0
            throw(
                DomainError("""\
              Invalid number of points: $num_points. An open Newton-Cotes rule requires \
              at least 1 point.\
              """)
            )
        else
            ξ = [-1.0 + i * 2.0 / (num_points + 1) for i in 1:num_points]
        end

    else
        throw(ArgumentError("""\
            Invalid Newton-Cotes type: $type. Valid types are 'closed' and 'open'.\
            """))
    end

    # Compute the weights by integrating the Lagrange basis functions. We need n Lagrange
    # basis functions when given n points.
    ql, wl = FastGaussQuadrature.gausslegendre(num_points)

    lagrange_at_ql = zeros(Float64, num_points, length(ql))
    for i in 1:num_points
        for j in eachindex(ql)
            l_poly_i = 1.0
            for k in 1:num_points
                if k != i
                    l_poly_i *= (ql[j] - ξ[k]) / (ξ[i] - ξ[k])
                end
            end

            lagrange_at_ql[i, j] = l_poly_i
        end
    end

    w = zeros(Float64, num_points)
    for i in 1:num_points
        for j in eachindex(ql, wl)
            w[i] += wl[j] * lagrange_at_ql[i, j]
        end
    end

    # Map roots and weights to the interval [0, 1].
    @. ξ = (ξ + 1.0) / 2.0
    @. w = 0.5 * w
    nodes = Points.CartesianPoints((ξ,))

    return CanonicalQuadratureRule(nodes, w, "Newton-Cotes ($type)")
end

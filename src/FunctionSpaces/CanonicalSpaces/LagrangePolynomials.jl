
"""
    Lagrange <: AbstractLagrangePolynomials

Lagrange interpolating polynomials.

# Fields
- `p::Int`: Degree of the Lagrange polynomial.
- `nodes::NT`: Points at which the polynomial should be interpolating. The length of the
    `nodes` vector dictates the degree.
- `barycentric_weights::Vector{T}`: Barycentric weights. `T` is `eltype(nodes)`.
"""
struct Lagrange{NT, T} <: AbstractLagrangePolynomials
    p::Int
    nodes::NT
    barycentric_weights::Vector{T}

    function Lagrange(nodes::NT) where {T, NT <: AbstractVector{T}}
        if length(nodes) <= 0
            throw(
                ArgumentError(
                    LazyString(
                        "Lagrange polynomials require at least 1 node. Got ",
                        length(nodes),
                        " nodes instead.",
                    ),
                ),
            )
        end
        p, barycentric_weights = compute_barycentric_weights(nodes)

        return new{NT, T}(p, nodes, barycentric_weights)
    end
end

"""
    compute_barycentric_weights(nodes)

Compute the barycentric weights for Lagrange interpolation using the given `nodes`.

Based on algorithm 9.2.1 of [Driscoll2022](@cite).
"""
function compute_barycentric_weights(nodes)
    p = length(nodes) - 1
    T = eltype(nodes)

    # Scaling to ensure stability.
    C = (nodes[p + 1] - nodes[1]) / T(4.0)

    # Adding one node at a time, compute inverses of the weights.
    w = ones(T, p + 1)
    for i in 0:(p - 1)
        for j in 1:(i + 1)
            # Current node difference.
            d = (nodes[j] - nodes[i + 2]) / C

            # Update previous.
            w[j] *= d

            # Compute new w.
            w[i + 2] *= -d
        end
    end

    # Go from inverses to weights.
    for i in eachindex(w)
        w[i] = one(T) / w[i]
    end

    return p, w
end

Memoization.@memoize function evaluate(
    polynomial::Lagrange, xi::Points.AbstractPoints{1}, nderivatives::Int=0
)
    neval = Points.get_num_points(xi)

    values = Vector{Vector{Matrix{eltype(xi)}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        values[j + 1] = Vector{Matrix{eltype(xi)}}(undef, 1)
        values[j + 1][1] = zeros(eltype(xi), neval, polynomial.p + 1)
    end

    nodes = polynomial.nodes
    weights = polynomial.barycentric_weights
    # loop over the evaluation points and evaluate all basis functions at each point
    @inbounds for point in eachindex(xi)
        _eval_lagrange!(view(values[1][1], point, :), nodes, weights, xi[point][1])
    end

    if nderivatives > 0
        D = _derivative_matrix(nodes)

        # Compute the first derivative
        derivative_idx = 1
        values[derivative_idx + 1][1] .= values[derivative_idx][1] * D

        # Loop over the remaining derivatives and compute them
        if nderivatives > 1
            # We use a recursive formula, so we need the previous D^{n} derivative to
            # compute the current one, which will be stored in the same matrix.
            D_n = copy(D)
            for derivative_idx in 2:nderivatives
                D_n = _derivative_matrix_next!(D_n, derivative_idx, D, nodes)
                values[derivative_idx + 1][1] .= values[derivative_idx][1] * D_n
            end
        end
    end

    return values
end

function _eval_lagrange!(result, nodes, weights, x::T) where {T <: Number}
    numerators = similar(weights)
    sum_terms = zero(T)
    for i in eachindex(numerators, nodes, weights)
        diff = x - nodes[i]

        if iszero(diff)
            # Exactly at a node, so return one at the node and zero elsewhere
            result .= zero(T)
            result[i] = one(T)
            return result
        else
            new_term = weights[i] / diff
            numerators[i] = new_term
            sum_terms += new_term
        end
    end

    for i in eachindex(result, numerators)
        result[i] = numerators[i] / sum_terms
    end

    return result
end

function get_derivative_space(elem_loc_basis::Lagrange)
    return Edge(elem_loc_basis.nodes)
end

"""
    Edge <: AbstractLagrangePolynomials

Edge histapolant polynomials of degree `p`.

The ``j``-th edge basis polynomial, ``e_{j}(\\xi)``, is given by, see
[Gerritsma2011](@cite),
```math
    e_{j}(\\xi) = -\\sum_{k=1}^{j} \\frac{\\mathrm{d} h_{k}(\\xi)}{\\mathrm{d}\\xi}, j = 1 , \\dots, p+1\\,.
```
where ``h_{k}(\\xi)`` is the ``k``-th Lagrange polynomial of degree ``(p+1)`` over a given
set of nodes. If ``\\xi_{i}`` are the given ``(p+1)`` nodes, then
```math
\\int_{\\xi_{i}}^{\\xi_{i+1}} e_{j}(\\xi)\\,\\mathrm{d}\\xi = \\delta_{i,j}, \\qquad i,j = 1, \\dots, p\\,,
```
i.e., they satisfy an integral Kronecker-``\\delta`` property.

See [Gerritsma2011](@cite) for more details.

# Fields
- `p::Int`: Degree of the Edge polynomial.
- `nodes::NT`: Nodes between which the polynomial should be histapolating. The length of
    the `nodes` vector dictates the degree.
- `lagrange_polynomial::Lagrange{NT, T}`: The underlying Lagrange polynomial. See
    [Lagrange](@ref) for the details.
"""
struct Edge{NT, T} <: AbstractEdgePolynomials
    p::Int
    nodes::NT
    lagrange_polynomial::Lagrange{NT, T}

    function Edge(nodes::NT) where {T, NT <: AbstractVector{T}}
        if length(nodes) <= 1
            throw(
                ArgumentError(
                    LazyString(
                        "Edge polynomials require at least 2 nodes. Got ",
                        length(nodes),
                        " nodes instead.",
                    ),
                ),
            )
        end
        lagrange_polynomial = Lagrange(nodes)

        return new{NT, T}(lagrange_polynomial.p - 1, nodes, lagrange_polynomial)
    end
end

Memoization.@memoize function evaluate(
    polynomial::Edge, xi::Points.AbstractPoints{1}, nderivatives::Int=0
)
    lagrange_eval = evaluate(polynomial.lagrange_polynomial, xi, nderivatives + 1)

    neval = Points.get_num_points(xi)
    edge_eval = Vector{Vector{Matrix{eltype(xi)}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        edge_eval[j + 1] = Vector{Matrix{eltype(xi)}}(undef, 1)
        edge_eval[j + 1][1] = zeros(eltype(xi), neval, polynomial.p + 1)
    end

    for i in 0:nderivatives
        for j in 1:(polynomial.p + 1)
            for j2 in 1:j
                for k in 1:neval
                    edge_eval[i + 1][1][k, j] += -lagrange_eval[i + 2][1][k, j2]
                end
            end
        end
    end

    return edge_eval
end

"""
    _derivative_matrix(nodes::Vector{Float64}; algorithm::Int=1)

Returns the first derivative of the polynomial lagrange basis
functions at the nodal points.

The derivative of the Lagrange interpolating basis functions
(``l_{n}^{p}(x)``) are given by:

```math
\\frac{dl_{n}(x)}{dx} = \\sum_{i=1, i\\neq n}^{p+1}\\prod_{j=1, j \\neq n, j\\neq i}^{p+1}\\frac{1}{x_{n}-x_{i}}\\frac{x-x_{j}}{x_{n}-x_{j}}
```

For computation at the nodes a more efficient and accurate formula can
be used, see [Costa2000](@cite):

```math
d_{k,j} = \\left\\{
\\begin{aligned}
&\\frac{c_{k}}{c_{j}}\\frac{1}{x_{k}-x_{j}}, \\qquad k \\neq j\\
&\\sum_{l=1,l\\neq k}^{p+1}\\frac{1}{x_{k}-x_{l}}, \\qquad k = j
\\end{aligned}
\\right.
```
with

```math
c_{k} = \\prod_{l=1,l\\neq k}^{p+1} (x_{k}-x_{l})
```

It returns a 2-dimensional matrix, ``D``, with the values of the derivative of
the polynomials, ``B_{j}``, of order ``p``

```math
D_{k,j} = \\frac{\\mathrm{d}B_{j}(x_{k})}{\\mathrm{d}x}
```
# Arguments
- `nodes::Vector{Float64}`: ``(p+1)`` nodes that define a set of Lagrange polynomials of
  degree ``p``, ``B_{j}^{p}(\\xi)``, for which to compute the derivative matrix. Note that
  the polynomials are such that ``B_{j}^{p}(\\xi_{i}) = \\delta_{j,i}`` with ``j,i = 1, \\dots, p+1``,
  `\\xi_{i} \\in [0.0, 1.0]`.

# Keyword arguments
- `algorithm::Int`: Flag to specify the algorithm to use
    1: <default> Stable algorithm using Eq. (7) in [1].
    2: Direct computation using Eq. (4) in [1].

# Returns
- `D::Array{Float64, 2}` :: The derivatives of the `(p+1)` polynomials evaluated at the `(p+1)` nodal points.
   ``D_{k,j} = \\frac{\\mathrm{d}B_{j}(x_{k})}{\\mathrm{d}x}``.
   (size: [p+1, p+1])
"""
function _derivative_matrix(nodes::AbstractVector{Float64}; algorithm::Int=1)
    #   Revisions:  2009-11-25 (apalha) First implementation.
    #               2014-12-03 (apalha) Removed pre-allocation of result.
    #                                   Replaced repmats by bsxfun for smaller
    #                                   memory footprint.
    #               2024-30-03 (apalha) Re-implemented in Julia, changed input arguments.

    # Get polynomial degree, p, the number of nodes plus one
    p = size(nodes)[1] - 1

    # The expression for the derivative matrix D is
    #             /
    #            | \frac{c_{k}}{c_{j}} (x_{k} - x_{j})^{-1},  for k \neq j,
    # D_{k,j} = <                                                                Eq. (6) of [1]
    #            | \sum_{l = 0, l \neq k} (x_{k} - x_{l})^{-1}, for k = j
    #             \
    #
    # with different approaches to compute both the ratio \frac{c_{k}}{c_{j}} and the diagonal terms
    # of the matrix D to minimize roundoff errors. Here we implement the different algorithms for
    # generality and for later comparison.

    D = zeros(Float64, p + 1, p + 1)  # allocate memory space for the differentiation matrix

    # Compute the differences ξ_{i} - ξ_{j}
    Δξ = broadcast(-, nodes, transpose(nodes))

    if algorithm == 1
        # Algorithm 2 (more stable computation)

        # Instead of computing \frac{c_{k}}{c_{j}} using the explicit formula for c, Eq. (4) in [1]
        #   c_{k} = \prod_{l = 1, l \neq k}^{p + 1} (ξ_{k} - ξ_{l})
        # Use logs and exponentials to convert the product into a sum and the division into a subtraction
        #   b_{k} = \sum_{l = 0, l \neq k}^{p + 1} ln(|ξ_{k} - ξ_{l}|)
        #   \frac{c_{k}}{c_{j}} = (1.0)^{k+j} exp(b_{k} - b_{j})

        # For the first part of computing the D matrix we need sums of log of Δξ ignoring the diagonal terms
        # to avoid an if we set them exp(1.0) to reduce errors
        Δξ[LinearAlgebra.diagind(Δξ)] .= exp(1.0)

        # The ratio c_{i} / c_{j} is computed with logs and exponentials to transform products into sums
        # c = reshape(prod(Δξ, dims=1), :, 1) from Eq. (4) (direct computation)
        b = mapreduce(u -> log(abs(u)), +, Δξ; dims=1)
        b .-= 1.0  # subtract the diagonal term that was added in the line above

        # Off diagonal elements
        for D_idx in CartesianIndices(D)
            # We also compute the diagonal terms, but with a wrong expression, for speed,
            # we compute them correctly in the following loop
            D[D_idx] =
                ((-1.0)^(D_idx[1] + D_idx[2])) * exp(b[D_idx[1]] - b[D_idx[2]]) / Δξ[D_idx]
        end

        # Diagonal elements

        # As mentioned above we now need to set the diagonal terms off Δξ to 0.0 because we need to
        # compute a sum over the axis, ignoring the diagonal
        Δξ[LinearAlgebra.diagind(Δξ)] .= 1.0

        # Now compute the diagonal terms of the differentiation matrix D
        D[LinearAlgebra.diagind(D)] .-= sum(D; dims=2) #-transpose(mapreduce(inv, +, Δξ, dims=1)) .+ 1.0  # remove the 1.0 in the diagonal, since we have the - sign in the matrix we need to add a + sign to remove the diagonal 1.0

    elseif algorithm == 2
        # Algorithm 2 (direct computation) Eq. (4)

        # For the first part of computing the D matrix we need products of Δξ ignoring the diagonal terms
        # to avoid an if we set them to 1.0, later we will set them to 0.0 since we need a sum ignoring the
        # diagonal
        Δξ[LinearAlgebra.diagind(Δξ)] = 1.0

        # Compute c_{i}
        c = reshape(prod(Δξ; dims=1), :, 1)  # just reshape it to be a column vector

        # Off diagonal elements
        for D_idx in CartesianIndices(D)
            # We also compute the diagonal terms, but with a wrong expression, for speed,
            # we compute them correctly in the following loop
            D[D_idx] = (c[D_idx[1]] / c[D_idx[2]]) / Δξ[D_idx]
        end

        # Diagonal elements

        # As mentioned above we now need to set the diagonal terms off Δξ to 0.0 because we need to
        # compute a sum over the axis, ignoring the diagonal
        Δξ[LinearAlgebra.diagind(Δξ)] .= 0.0

        # Now compute the diagonal terms of the differentiation matrix D
        D[LinearAlgebra.diagind(D)] .= mapreduce(inv, +, Δξ; dims=1)
    end

    return D
end

"""
_derivative_matrix_next!(D_m::Array{Float64, 2}, m::Int, D::Array{Float64, 2}, nodes::Vector{Float64})

Given the derivative matrix (of order 1), `D`, and the derivative matrix of order `n`, `D_m`,
compute the derivative of order `(n+1)`.

We follow the algorithm proposed in section 4 of [Costa2000](@cite).

# Arguments
- `D_m::Array{float64, 2}`: Derivative matrix of order `m` for the ``(p+1)`` `polynomials`` of degree ``p``,
   evaluated at the `(p+1)` nodal points, following the same format as the derivative matrix `D` below.
   (size: [p+1, p+1])
- `D::Array{Float64, 2}` :: The derivatives of the `(p+1)` polynomials evaluated at the `(p+1)` nodal points.
   ``D_{k,j} = \\frac{\\mathrm{d}B_{j}(x_{k})}{\\mathrm{d}x}``.
   (size: [p+1, p+1])
- `nodes::Vector{Float64}`: ``(p+1)`` nodes that define a set of Lagrange polynomials of
   degree ``p``, ``B_{j}^{p}(\\xi)``, for which to compute the derivative matrix. Note that
   the polynomials are such that ``B_{j}^{p}(\\xi_{i}) = \\delta_{j,i}`` with ``j,i = 1, \\dots, p+1``,
   `\\xi_{i} \\in [0.0, 1.0]`.

# Returns
- `D_m::Array{Float64, 2}` :: The derivatives or degree `(m+1)` of the `(p+1)` polynomials evaluated at the `(p+1)` nodal points.
   ``D^{(m)}_{k,j} = \\frac{\\mathrm{d}^{m}B_{j}(x_{k})}{\\mathrm{d}x^{m}}``. `D_m` given as input argument is updated with the new value.
   (size: [p+1, p+1])

"""
function _derivative_matrix_next!(
    D_m::Array{Float64, 2}, m::Int, D::Array{Float64, 2}, nodes::AbstractVector{Float64}
)
    # Compute the differences ξ_{i} - ξ_{j}
    Δξ = broadcast(-, nodes, transpose(nodes))

    # We will need to compute inverses of Δξ, to avoid skipping the diagonal, we set it to 1.0
    # This does not affect the final result since the diagonal terms of the derivative matrix
    # are computed in another step
    Δξ[LinearAlgebra.diagind(Δξ)] .= 1.0

    # Since we wish to update D_n to avoid creating another matrix, we need to first extract the
    # diagonal elements, since we need them in the computation
    D_m_diag = D_m[LinearAlgebra.diagind(D_m)]

    # Compute the off diagonal elements Eq. (13) from [1]
    for D_m_idx in CartesianIndices(D_m)
        D_m[D_m_idx] =
            m * (D_m_diag[D_m_idx[1]] * D[D_m_idx] - (D_m[D_m_idx] / Δξ[D_m_idx]))
    end

    # The diagonal terms are computed by the row sum formula (9) of [1]
    D_m[LinearAlgebra.diagind(D_m)] .-= sum(D_m; dims=2)  # this replaces the current value of the diagonal by the row sum without the diagonal

    return D_m
end

"""
    build_two_scale_matrix(ect_space::AbstractLagrangePolynomials, num_sub_elements::Int)

Uniformly subdivides the ECT space into `num_sub_elements` sub-elements. It is assumed that
`num_sub_elements` is a power of 2, else the method throws an argument error. It returns a
global subdivision matrix that maps the global basis functions of the ECT space to the
global basis functions of the subspaces.

# Arguments
- `ect_space::AbstractECTSpaces`: A ect space.
- `num_sub_elements::Int`: The number of subspaces to divide the EC T space into.

# Returns
- `::SparseMatrixCSC{Float64}`: A global subdivision matrix that maps the global basis
functions of the ECT space to the global basis functions of the subspaces.
"""
function build_two_scale_matrix(
    polynomials::AbstractLagrangePolynomials, num_sub_elements::Int
)
    p = get_polynomial_degree(polynomials)
    nodes = polynomials.nodes

    # build the global set of nodes on all sub-elements by scaling and translating the original nodes
    global_nodes = zeros(Float64, num_sub_elements * (p + 1))
    for i in 1:num_sub_elements
        global_nodes[((i - 1) * (p + 1) + 1):(i * (p + 1))] .=
            ((i - 1) .+ nodes) ./ num_sub_elements
    end

    # subdivision matrix is the evaluation of lagrange polynomials at the global nodes
    return evaluate(polynomials, global_nodes)[1][1], global_nodes
end

using LinearAlgebra

import PolynomialBases

abstract type AbstractLagrangePolynomials <: AbstractCanonicalSpace end
abstract type AbstractEdgePolynomials <: AbstractCanonicalSpace end

struct LobattoLegendre <: AbstractLagrangePolynomials
    p::Int  # Polynomial degree
    nodes::Vector{Float64}  # Polynomial grid nodes, there are p+1 nodes
    _core_polynomials::PolynomialBases.LobattoLegendre  # Lobatto-Legendre polynomial basis core structure from PolynomialBases
end

function LobattoLegendre(p::Int)
    ll_polynomials = PolynomialBases.LobattoLegendre(p)
    nodes = (ll_polynomials.nodes .+ 1.0)./2.0
    return LobattoLegendre(p, nodes, ll_polynomials)
end

struct GaussLegendre <: AbstractLagrangePolynomials
    p::Int  # Polynomial degree
    nodes::Vector{Float64}  # Polynomial grid nodes, there are p+1 nodes
    _core_polynomials::PolynomialBases.GaussLegendre  # Gauss-Legendre polynomial basis core structure from PolynomialBases
end

function GaussLegendre(p::Int)
    gl_polynomials = PolynomialBases.GaussLegendre(p)
    nodes = (gl_polynomials.nodes .+ 1.0)./2.0
    return GaussLegendre(p, nodes, gl_polynomials)
end

@doc raw"""
Edge polynomials of degree ``p`` over Gauss-Lobatto-Legendre nodes.

The ``j``-th edge basis polynomial, ``e_{j}(\xi)``, is given by, see [1],

```math
    e_{j}(\xi) = -\sum_{k=1}^{j} \frac{\mathrm{d} h_{k}(\xi)}{\mathrm{d}\xi}, j = 1 , \dots, p+1\,.
```

where ``h_{k}(\xi)`` is the ``k``-th Lagrange polynomial of degree ``(p+1)`` over 
Gauss-Lobatto-Legendre nodes.

If ``\xi_{i}`` are the ``(p+1)`` nodes of the associated Gauss-Lobatto-Legendre polynomials, then

```math
\int_{\xi_{i}}^{\xi_{i+1}} e_{j}(\xi)\,\mathrm{d}\xi = \delta_{i,j}, \qquad i,j = 1, \dots, p\,,
```

i.e., they satisfy an integral Kronecker-``\delta`` property.

# Fields
- `p::Int`: polynomial degree.
- `nodes::Vector{Float64}`: the nodes of the Gauss-Lobatto-Legendre Lagrange polynomials associated to it.

[1] Gerritsma, M.: Edge functions for spectral element methods. 
    Submitted to the proceedings of ICOSAHOM 2009
"""
struct EdgeLobattoLegendre <: AbstractEdgePolynomials
    p::Int  # Polynomial degree
    nodes::Vector{Float64}  # Polynomial grid nodes, there are p+1 nodes
    _core_polynomials::LobattoLegendre  # Lobatto-Legendre polynomial basis core structure from Mantis
end

function EdgeLobattoLegendre(p::Int)
    ll_polynomials = LobattoLegendre(p+1)
    nodes = ll_polynomials.nodes  # no need to rescale because they are already rescaled
    return EdgeLobattoLegendre(p, nodes, ll_polynomials)
end

@doc raw"""
    evaluate(polynomials::AbstractLagrangePolynomials, ξ::Vector{Float64}, nderivatives::Int64)

Evaluate the first `nderivatives` derivatives of the polynomials ``B_{j}(\xi)``, ``j = 1, \dots, p+1``, 
in `polynomials` at `ξ` for ``\xi \in [0.0, 1.0]``. 

# Arguments
- `polynomials::AbstractLagrangePolynomials`: ``(p+1)`` polynomials of degree ``p``,  
    ``B_{j}^{p}(\xi)`` with ``j = 1, \dots, p+1``, to evaluate.
- `ξ::Vector{Float64}`: vector of ``n`` evaluation points ``\xi \in [0.0, 1.0]``.
- `nderivatives::Int64`: maximum order of derivatives to be computed (nderivatives ``\leq p``). Will compute
   the polynomial, first derivative, second derivative, etc, up to nderivatives.

# Returns 
`d_polynomials::Array{Float64, 3}(n, p+1, nderivatives)` with the evaluation of `polynomials` and its derivatives
up to degree `nderivatives` at every point `ξ`. `d_polynomials[i, j, k]` ``= \frac{\mathrm{d}^{k}B_{j}}{\mathrm{d}\xi^{k}}(\xi_{i})``.

See also [`evaluate(polynomial::AbstractLagrangePolynomials, ξ::Float64, nderivatives::Int64)`](@ref).
"""
function evaluate(polynomials::AbstractLagrangePolynomials, ξ::Vector{Float64}, nderivatives::Int64)
    # Get information from inputs on size of computation 
    n_points = size(ξ, 1)  # the number of points where to evaluate the polynomials and their derivatives
    p = polynomials.p # the degree of the polynomials, the number of polynomials is p + 1

    # Uses the `PolynomialBases` package to evaluate the `polynomial`. This 
    # uses the attribute `_core_polynomials` in the `polynomial` struct. This 
    # allows every Lagrange polynomial to be evaluated by calling the same 
    # method from `PolynomialBases`.
    ξ_scaled = 2.0*ξ .- 1.0  # Abstract polynomials are evaluated for ξ ∈ [0, 1], but PolynomialBases uses ξ ∈ [-1, 1]
    
    # Allocate memory space for the result, derivatives are the third dimension
    # We evaluate the polynomials and the derivatives (hence the + 1) 
    d_polynomials = Array{Float64, 3}(undef, n_points, p + 1, nderivatives + 1)
    
    # Evaluate the polynomials at the points ξ
    # PolynomialBases.interpolation_matrix!(view(d_polynomials, :, :, 1), ξ, polynomials._core_polynomials.nodes, polynomials._core_polynomials.baryweights)
    d_polynomials[:, :, 1] .= PolynomialBases.interpolation_matrix(ξ_scaled, polynomials._core_polynomials.nodes, polynomials._core_polynomials.baryweights)
    
    # Evaluate the derivatives at the points ξ
    if nderivatives > 0
        # Given a Lagrange polynomial basis B_{i}(x), with i = 1, ..., p+1, over the node ξ_{i} 
        # its derivative evaluated at the points x_{j} is given by
        #
        #    d^{n}B_{i}(x_{j}/dx^{n} = \sum_{k=1}^{p+1} d^{n}B_{i}(ξ_{k})/dx^{n} B_{k}(x_{j})
        #
        # and the derivatives at the nodes d^{n}B_{i}(ξ_{k})/dx^{n} are given by the recursive formula
        #
        #   d^{n+1}B_{i}(ξ_{k}/dx^{n} = \sum_{l=1}^{p+1} d^{n}B_{i}(ξ_{l})/dx^{n} dB_{l}(ξ_{j})/dx
        #
        # with
        #
        #   dB_{i}(ξ_{j})/dx = D_{i,j} --> computed without the recursion
        # 
        # In matrix form the recursive relation takes the simple form 
        #   d^{n}B_{i}(ξ_{j})/dx^{n} := B^{n}_{j,i} = [D^{n} B^{0}]_{i,j} = D^{n}_{j,i}  (matrix products)
        # 
        # The last equality is due to the fact that 
        #   
        #   B_{i}(ξ_{j}) = δ_{j,i}  (Kronecker property)
        #
        # Therefore, we have that 
        #
        #   d^{n}B_{i}(x_{j}/dx^{n} = \sum_{k=1}^{p+1} d^{n}B_{i}(ξ_{k})/dx^{n} B_{k}(x_{j})
        #                           = \sum_{k=1}^{p+1} D^{n}_{k, i} B_{k}(x_{j})
        #                           = \sum_{k=1}^{p+1} D^{n}_{k, i} B^{0}_{j, k} = [B^{0} D^{n}]_{j, i}  (matrix product)
        
        # Compute the derivative matrix
        # Using PolynomialBases
        # D = PolynomialBases.derivative_matrix(polynomials._core_polynomials.nodes, polynomials._core_polynomials.baryweights)
        # D .*= 2.0 
        # Using internal function
        D = _derivative_matrix(polynomials.nodes)

        # Compute the first derivative
        # mul!(view(d_polynomials, :, :, derivative_idx + 1), view(d_polynomials, :, :, derivative_idx), D)
        derivative_idx = 1
        d_polynomials[:, :, derivative_idx + 1] .= d_polynomials[:, :, derivative_idx] * D

        # Loop over the remaining derivatives and compute them 
        if nderivatives > 1
            D_n = copy(D)  # we use a recursive formula, so we need the previous D^{n} derivative 
                           # to compute the current one, which will be stored in the same matrix
            for derivative_idx in 2:nderivatives
                # In this way we reuse previously computed values of D^{n-1}, so we just need to update in each step
                # mul!(view(d_polynomials, :, :, derivative_idx + 1), view(d_polynomials, :, :, derivative_idx), D)
                D_n = _derivative_matrix_next!(D_n, derivative_idx, D, polynomials.nodes)
                d_polynomials[:, :, derivative_idx + 1] .= d_polynomials[:, :, 1] * D_n
            end
        end
    end

    return Dict{Int,Matrix{Float64}}(i => d_polynomials[:,:,i+1] for i = 0:nderivatives)
    
end

function evaluate(polynomials::AbstractLagrangePolynomials, ξ::Float64, nderivatives::Int64)::Array{Float64}
    return evaluate(polynomials, [ξ], nderivatives)
end


@doc raw"""
    evaluate(polynomials::AbstractLagrangePolynomials, ξ::Vector{Float64})

Evaluate the polynomials ``B_{j}(\xi)``, ``j = 1, \dots, p+1``, in `polynomials` at `ξ` for 
``\xi \in [0.0, 1.0]``. 

# Arguments
- `polynomials::AbstractLagrangePolynomials`: ``(p+1)`` polynomials of degree ``p``,  
    ``B_{j}(\xi)`` with ``j = 1, \dots, p+1``, to evaluate.
- `ξ::Vector{Float64}`: vector of ``n`` evaluation points ``\xi \in [0.0, 1.0]``.

# Returns 
`polynomials::Array{Float64, 3}(n, p+1)` with the evaluation of `polynomials` at every point `ξ`. 
    `polynomials[i, j]` ``= B_{j}(\xi_{i})``.

See also [`evaluate(polynomial::AbstractLagrangePolynomials, ξ::Float64, nderivatives::Int64)`](@ref).
"""
function evaluate(polynomials::AbstractLagrangePolynomials, ξ::Vector{Float64})
    return evaluate(polynomials, ξ, 0)
end

@doc raw"""
    evaluate(polynomials::EdgeLobattoLegendre, ξ::Vector{Float64}, nderivatives::Int64)

Evaluate the polynomials ``B_{j}(\xi)``, ``j = 1, \dots, p+1``, in `polynomials` at `ξ` for 
``\xi \in [0.0, 1.0]``. 


# Arguments
- `polynomials::EdgeLobattoLegendre`: ``(p+1)`` polynomials of degree ``p``,  
    ``B_{j}^{p}(\xi)`` with ``j = 1, \dots, p+1``, to evaluate.
- `ξ::Vector{Float64}`: vector of ``n`` evaluation points ``\xi \in [0.0, 1.0]``.
- `nderivatives::Int64`: maximum order of derivatives to be computed (nderivatives ``\leq p``). Will compute
   the polynomial, first derivative, second derivative, etc, up to nderivatives.

# Returns 
`d_polynomials::Array{Float64, 3}(n, p+1, nderivatives)` with the evaluation of `polynomials` and its derivatives
up to degree `nderivatives` at every point `ξ`. `d_polynomials[i, j, k]` ``= \frac{\mathrm{d}^{k}B_{j}}{\mathrm{d}\xi^{k}}(\xi_{i})``.

See also [`evaluate(polynomial::AbstractLagrangePolynomials, ξ::Float64, nderivatives::Int64)`](@ref).
"""
function evaluate(polynomials::EdgeLobattoLegendre, ξ::Vector{Float64}, nderivatives::Int64)
    # The edge basis functions are given by, see documentation of EdgeLobattoLegendre:
    #   
    #   edge_{i}(x) = -\sum_{j=1}^{i} dh_{j}(x)/dx, i=1,...,p
    # 
    # This means that the derivatives are 
    #   
    #   d^{n}edge_{i}(x)/dx^{n} = -\sum_{j=1}^{i} d^{n+1}h_{j}(x)/dx^{n+1}, i=1,...,p
    # 

    # Compute nderivatives+1 of the Lagrange polynomials over Gauss-Lobatto-Legendre nodes
    ll_polynomials_eval = evaluate(polynomials._core_polynomials, ξ, nderivatives+1)
    

    return Dict{Int,Matrix{Float64}}(i => -cumsum(ll_polynomials_eval[i+1][:,1:(end-1)],dims=2) for i = 0:nderivatives)
end


@doc raw"""
    evaluate(polynomials::EdgeLobattoLegendre, ξ::Vector{Float64})

Evaluate the polynomials ``B_{j}(\xi)``, ``j = 1, \dots, p+1``, in `polynomials` at `ξ` for 
``\xi \in [0.0, 1.0]``. 

# Arguments
- `polynomials::EdgeLobattoLegendre`: ``(p+1)`` polynomials of degree ``p``,  
    ``B_{j}(\xi)`` with ``j = 1, \dots, p+1``, to evaluate.
- `ξ::Vector{Float64}`: vector of ``n`` evaluation points ``\xi \in [0.0, 1.0]``.

# Returns 
`polynomials::Array{Float64, 3}(n, p+1)` with the evaluation of `polynomials` at every point `ξ`. 
    `polynomials[i, j]` ``= B_{j}(\xi_{i})``.

See also [`evaluate(polynomial::EdgeLobattoLegendre, ξ::Float64, nderivatives::Int64)`](@ref).
"""
function evaluate(polynomials::EdgeLobattoLegendre, ξ::Vector{Float64})
    return evaluate(polynomials, ξ, 0)
end


@doc raw"""
    _derivative_matrix(nodes::Vector{Float64}; algorithm::Int64=1)

Returns the first derivative of the polynomial lagrange basis
functions at the nodal points.

The derivative of the Lagrange interpolating basis functions
(``l_{n}^{p}(x)``) are given by:
  
```math
\frac{dl_{n}(x)}{dx} = \sum_{i=1, i\neq n}^{p+1}\prod_{j=1, j \neq n, j\neq i}^{p+1}\frac{1}{x_{n}-x_{i}}\frac{x-x_{j}}{x_{n}-x_{j}}
```

For computation at the nodes a more efficient and accurate formula can
be used, see [Costa2000](@cite):

```math
d_{k,j} = \left\{
\begin{aligned}
&\frac{c_{k}}{c_{j}}\frac{1}{x_{k}-x_{j}}, \qquad k \neq j\\
&\sum_{l=1,l\neq k}^{p+1}\frac{1}{x_{k}-x_{l}}, \qquad k = j
\end{aligned}
\right.
```
with

```math
c_{k} = \prod_{l=1,l\neq k}^{p+1} (x_{k}-x_{l})   
```

It returns a 2-dimensional matrix, `D`, with the values of the derivative of
the polynomials, $B_{j}$, of order `p`

```math
D_{k,j} = \frac{\mathrm{d}B_{j}(x_{k})}{\mathrm{d}x}
```
# Arguments
- `nodes::Vector{Float64}`: ``(p+1)`` nodes that define a set of Lagrange polynomials of
  degree ``p``, ``B_{j}^{p}(\xi)``, for which to compute the derivative matrix. Note that 
  the polynomials are such that ``B_{j}^{p}(\xi_{i}) = \delta_{j,i}`` with ``j,i = 1, \dots, p+1``, 
  `\xi_{i} \in [0.0, 1.0]`.

# Keyword arguments
- `algorithm::Int64`: Flag to specify the algorithm to use
    1: <default> Stable algorithm using Eq. (7) in [1].
    2: Direct computation using Eq. (4) in [1].

# Returns 
- `D::Array{Float64, 2}` :: The derivatives of the `(p+1)` polynomials evaluated at the `(p+1)` nodal points.
   ``D_{k,j} = \frac{\mathrm{d}B_{j}(x_{k})}{\mathrm{d}x}``.
   (size: [p+1, p+1])
"""
function _derivative_matrix(nodes::Vector{Float64}; algorithm::Int64=1)
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
    
    D = zeros(Float64, p+1, p+1)  # allocate memory space for the differentiation matrix

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
        Δξ[diagind(Δξ)] .= exp(1.0)

        # The ratio c_{i} / c_{j} is computed with logs and exponentials to transform products into sums 
        # c = reshape(prod(Δξ, dims=1), :, 1) from Eq. (4) (direct computation)
        b = mapreduce(u->log(abs(u)), +, Δξ, dims=1)
        b .-= 1.0  # subtract the diagonal term that was added in the line above

        # Off diagonal elements
        for D_idx in CartesianIndices(D)
            # We also compute the diagonal terms, but with a wrong expression, for speed,
            # we compute them correctly in the following loop
            D[D_idx] = ((-1.0)^(D_idx[1] + D_idx[2])) * exp(b[D_idx[1]] - b[D_idx[2]]) / Δξ[D_idx]
        end

        # Diagonal elements

        # As mentioned above we now need to set the diagonal terms off Δξ to 0.0 because we need to
        # compute a sum over the axis, ignoring the diagonal
        Δξ[diagind(Δξ)] .= 1.0 

        # Now compute the diagonal terms of the differentiation matrix D
        D[diagind(D)] .-= sum(D, dims=2) #-transpose(mapreduce(inv, +, Δξ, dims=1)) .+ 1.0  # remove the 1.0 in the diagonal, since we have the - sign in the matrix we need to add a + sign to remove the diagonal 1.0

    elseif algorithm == 2
        # Algorithm 2 (direct computation) Eq. (4)

        # For the first part of computing the D matrix we need products of Δξ ignoring the diagonal terms
        # to avoid an if we set them to 1.0, later we will set them to 0.0 since we need a sum ignoring the
        # diagonal
        Δξ[diagind(Δξ)] = 1.0

        # Compute c_{i}
        c = reshape(prod(Δξ, dims=1), :, 1)  # just reshape it to be a column vector
        
        # Off diagonal elements
        for D_idx in CartesianIndices(D)
            # We also compute the diagonal terms, but with a wrong expression, for speed,
            # we compute them correctly in the following loop
            D[D_idx] = (c[D_idx[1]] / c[D_idx[2]]) / Δξ[D_idx]
        end

        # Diagonal elements

        # As mentioned above we now need to set the diagonal terms off Δξ to 0.0 because we need to
        # compute a sum over the axis, ignoring the diagonal
        Δξ[diagind(Δξ)] .= 0.0 

        # Now compute the diagonal terms of the differentiation matrix D
        D[diagind(D)] .= mapreduce(inv, +, Δξ, dims=1) 
    end

    return D
end

@doc raw"""
_derivative_matrix_next!(D_m::Array{Float64, 2}, m::Int64, D::Array{Float64, 2}, nodes::Vector{Float64})

Given the derivative matrix (of order 1), `D`, and the derivative matrix of order `n`, `D_m`,
compute the derivative of order `(n+1)`.

We follow the algorithm proposed in section 4 of [Costa2000](@cite).

# Arguments
- `D_m::Array{float64, 2}`: Derivative matrix of order `m` for the ``(p+1)`` `polynomials`` of degree ``p``,
   evaluated at the `(p+1)` nodal points, following the same format as the derivative matrix `D` below.
   (size: [p+1, p+1])
- `D::Array{Float64, 2}` :: The derivatives of the `(p+1)` polynomials evaluated at the `(p+1)` nodal points.
   ``D_{k,j} = \frac{\mathrm{d}B_{j}(x_{k})}{\mathrm{d}x}``.
   (size: [p+1, p+1])
- `nodes::Vector{Float64}`: ``(p+1)`` nodes that define a set of Lagrange polynomials of
   degree ``p``, ``B_{j}^{p}(\xi)``, for which to compute the derivative matrix. Note that 
   the polynomials are such that ``B_{j}^{p}(\xi_{i}) = \delta_{j,i}`` with ``j,i = 1, \dots, p+1``, 
   `\xi_{i} \in [0.0, 1.0]`.

# Returns 
- `D_m::Array{Float64, 2}` :: The derivatives or degree `(m+1)` of the `(p+1)` polynomials evaluated at the `(p+1)` nodal points.
   ``D^{(m)}_{k,j} = \frac{\mathrm{d}^{m}B_{j}(x_{k})}{\mathrm{d}x^{m}}``. `D_m` given as input argument is updated with the new value.
   (size: [p+1, p+1])

"""
function _derivative_matrix_next!(D_m::Array{Float64, 2}, m::Int64, D::Array{Float64, 2}, nodes::Vector{Float64})
    # Compute the differences ξ_{i} - ξ_{j}
    Δξ = broadcast(-, nodes, transpose(nodes)) 

    # We will need to compute inverses of Δξ, to avoid skipping the diagonal, we set it to 1.0
    # This does not affect the final result since the diagonal terms of the derivative matrix
    # are computed in another step 
    Δξ[diagind(Δξ)] .= 1.0

    # Since we wish to update D_n to avoid creating another matrix, we need to first extract the 
    # diagonal elements, since we need them in the computation 
    D_m_diag = D_m[diagind(D_m)]

    # Compute the off diagonal elements Eq. (13) from [1]
    for D_m_idx in CartesianIndices(D_m)
        D_m[D_m_idx] = m * (D_m_diag[D_m_idx[1]] * D[D_m_idx] - (D_m[D_m_idx] / Δξ[D_m_idx])) 
    end

    # The diagonal terms are computed by the row sum formula (9) of [1]
    D_m[diagind(D_m)] .-= sum(D_m, dims=2)  # this replaces the current value of the diagonal by the row sum without the diagonal

    return D_m
end

"""
    Bernstein <: AbstractCanonicalSpace

Concrete type for Bernstein polynomials.

# Fields
- `p::Int`: Degree of the Bernstein polynomial.
"""
struct Bernstein <: AbstractCanonicalSpace
    p::Int

    function Bernstein(p::Int)
        if p < 0
            msg = "Bernstein polynomials must be of degree at least 0. Got p = $p."
            throw(ArgumentError(msg))
        end

        return new(p)
    end
end

"""
    evaluate(polynomial::Bernstein, ξ::Vector{Float64}, nderivatives::Int=0)

Compute derivatives up to order `nderivatives` for all Bernstein polynomials of degree `p`
at `ξ` for ``\\xi \\in [0.0, 1.0]``.

# Arguments
- `polynomial::Bernstein`: Bernstein polynomial.
- `ξ::Vector{Float64}`: Vector of evaluation points ``\\in [0.0, 1.0]``.
- `nderivatives::Int=0`: Maximum order of derivatives to be computed (`nderivatives`
    ``\\leq p``). Defaults to `0`, i.e., only the values of the polynomials are computed.

# Returns
- `::Vector{Vector{Matrix{Float64}}}`: Nested vector containing the values.
"""
Memoization.@memoize function evaluate(
    polynomials::Bernstein, xi::Points.AbstractPoints{1}, nderivatives::Int=0
)
    # store the values and derivatives here
    neval = length(xi)

    # allocate space for derivatives
    # - ders[j+1][1] contains the matrix of evaluations of the j-th derivative
    ders = Vector{Vector{Matrix{Float64}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        ders[j + 1] = Vector{Matrix{Float64}}(undef, 1)
        ders[j + 1][1] = zeros(Float64, neval, polynomials.p + 1)
    end
    # loop over the evaluation points and evaluate all derivatives at each point
    for i in eachindex(xi)
        tmp = _evaluate(polynomials, xi[i][1], nderivatives)
        for cart_ind in CartesianIndices(size(tmp))
            (basis, j) = Tuple(cart_ind)
            ders[j][1][i, basis] = tmp[basis, j]
        end
    end

    return ders
end

"""
    _evaluate(polynomial::Bernstein, xi::Float64, nderivatives::Int64)

Compute derivatives up to order `nderivatives` for all Bernstein polynomials of degree `p`
at ``\\xi`` for ``\\xi \\in [0.0, 1.0]``.

Computes the values of the Bernstein polynomial and its derivative. This function is called
by all other implementations of the Bernstein polynomial.

# Arguments
- `polynomial::Bernstein`: Bernstein polynomial
- `xi::Float64`: evaluation point ``\\in [0.0, 1.0]``.
- `nderivatives::Int64`: maximum order of derivatives to be computed (nderivatives
    ``\\leq p``).
"""
function _evaluate(polynomial::Bernstein, xi::Float64, nderivatives::Int64)
    # degree
    p = get_polynomial_degree(polynomial)
    # store the values and derivatives here
    ders = zeros(Float64, p + 1, nderivatives + 1)
    for k in 0:nderivatives
        for i in 0:p
            ders[i + 1, k + 1] = _dbpoly(p, i, k, xi)
        end
    end
    return ders
end
function _bpoly(p::Int, i::Int, xi::Float64)
    if i < 0 || i > p
        return 0.0
    else
        return binomial(p, i) * xi^i * (1 - xi)^(p-i)
    end
end
function _dbpoly(p::Int, i::Int, k::Int, xi::Float64)
    if k == 0
        return _bpoly(p, i, xi)
    elseif k > p
        return 0.0
    else
        val = 0.0
        for r in max(0, i + k - p):min(i, k)
            val += (-1)^(r+k) * binomial(k, r) * _bpoly(p-k, i-r, xi)
        end
        return val * prod((p - k + 1):p)
    end
end

"""
    extract_monomial_to_bernstein(polynomial::Bernstein)

Computes transformation matrix T that transforms coefficients of
a polynomial in terms of the monomial basis into coefficients of
in terms of the Bernstein basis.

# Arguments
- `polynomial::Bernstein`: Bernstein polynomial
"""
function extract_monomial_to_bernstein(polynomial::Bernstein)
    # degree
    p = polynomial.p

    # arg checks
    if p < 0
        msg = "The Bernstein polynomials must be of degree at least 0."
        throw(ArgumentError(msg))
    end

    # build transformation matrix for mapping coefficients of a polynomial in monomial basis
    # to that of Bernstein
    T = zeros(Float64, p + 1, p + 1)
    for i in 0:p
        for j in 0:i
            T[i + 1, j + 1] = binomial(i, j) / binomial(p, j)
        end
    end

    return T
end

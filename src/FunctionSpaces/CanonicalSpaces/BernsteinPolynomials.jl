
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

    # arg checks
    if xi < 0.0 || xi > 1.0
        msg = "x = $xi is outside the interval [0.0, 1.0]."
        throw(ArgumentError(msg))
    end

    # compute upper and lower triangular values that combine to form values and derivatives
    ndu = zeros(Float64, p + 1, p + 1)
    left = zeros(Float64, p + 1)
    right = zeros(Float64, p + 1)
    ndu[1, 1] = 1
    for j in 1:p
        left[j + 1] = xi
        right[j + 1] = 1.0 - xi
        saved = 0.0
        for r in 0:(j - 1)
            #lower trinagle
            ndu[j + 1, r + 1] = right[r + 2] + left[j - r + 1]
            temp = ndu[r + 1, j] / ndu[j + 1, r + 1]

            # upper triangle
            ndu[r + 1, j + 1] = saved + right[r + 2] * temp
            saved = left[j - r + 1] * temp
        end
        ndu[j + 1, j + 1] = saved
    end

    # store the values and derivatives here
    ders = zeros(Float64, p + 1, nderivatives + 1)
    # values are contained in the last column of ndu
    ders[:, 1] = ndu[:, p + 1]

    # if nderivatves>0, the next section computes the derivatives
    if nderivatives > 0
        a = zeros(Int64, 2, nderivatives + 1)
        for r in 0:p
            s1 = 0
            s2 = 1
            a[1, 1] = 1
            # loop to compute kth derivative
            for k in 1:nderivatives
                d = 0
                rk = r - k
                pk = p - k
                if (r >= k)
                    a[s2 + 1, 1] = a[s1 + 1, 1] / ndu[pk + 2, rk + 1]
                    d = a[s2 + 1, 1] * ndu[rk + 1, pk + 1]
                end
                if (rk >= -1)
                    j1 = 1
                else
                    j1 = -rk
                end
                if (r - 1 <= pk)
                    j2 = k - 1
                else
                    j2 = p - r
                end

                for j in j1:j2
                    a[s2 + 1, j + 1] =
                        (a[s1 + 1, j + 1] - a[s1 + 1, j]) / ndu[pk + 2, rk + j + 1]
                    d = d + a[s2 + 1, j + 1] * ndu[rk + j + 1, pk + 1]
                end

                if (r <= pk)
                    a[s2 + 1, k + 1] = -a[s1 + 1, k] / ndu[pk + 2, r + 1]
                    d = d + a[s2 + 1, k + 1] * ndu[r + 1, pk + 1]
                end
                ders[r + 1, k + 1] = d
                j = s1
                s1 = s2
                s2 = j
            end
        end

        # multiply by the correct factors
        r = p
        for k in 1:nderivatives
            ders[:, k + 1] *= r
            r = r * (p - k)
        end
    end

    return ders
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

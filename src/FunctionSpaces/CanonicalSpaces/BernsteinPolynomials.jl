
"""
    Bernstein <: AbstractCanonicalSpace

Concrete type for Bernstein polynomials; see [DeBoor1978](@cite).

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
    neval = Points.get_num_points(xi)
    p = get_polynomial_degree(polynomials)

    # allocate space for derivatives
    # - ders[j+1][1] contains the matrix of evaluations of the j-th derivative
    ders = Vector{Vector{Matrix{eltype(xi)}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        ders[j + 1] = Vector{Matrix{eltype(xi)}}(undef, 1)
        ders[j + 1][1] = zeros(eltype(xi), neval, p + 1)
    end
    # loop over the evaluation points and evaluate all derivatives at each point
    @inbounds for derivative in 0:nderivatives
        for basis in 0:p
            for point in eachindex(xi)
                ders[derivative + 1][1][point, basis + 1] = _dbpoly(
                    p, basis, derivative, xi[point][1]
                )
            end
        end
    end

    return ders
end

# function evaluate!(
#     out::Vector{Vector{Vector{Matrix{T}}}},
#     polynomials::Bernstein,
#     xi::Points.AbstractPoints{1},
# ) where {T <: Number}
#     p = get_polynomial_degree(polynomials)

#     # loop over the evaluation points and evaluate all derivatives at each point
#     for derivative in eachindex(out) # from 1 to nderivatives + 1
#         for basis in 0:p
#             for point in eachindex(xi)
#                 out[derivative][1][1][point, basis + 1] = _dbpoly(
#                     p, basis, derivative - 1, xi[point][1]
#                 )
#             end
#         end
#     end

#     return out
# end

function _bpoly(p::Int, i::Int, xi::T) where {T <: Number}
    if i < 0 || i > p
        return 0.0
    else
        return binomial(p, i) * xi^i * (1 - xi)^(p - i)
    end
end

function _dbpoly(p::Int, i::Int, k::Int, xi::T) where {T <: Number}
    if k == 0
        return _bpoly(p, i, xi)
    elseif k > p
        return 0.0
    else
        val = 0.0
        for r in max(0, i + k - p):min(i, k)
            val += (-1)^(r + k) * binomial(k, r) * _bpoly(p - k, i - r, xi)
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

"""
    build_degree_elevation_matrix(canonical_space::Bernstein, degree_delta::Int)

Builds the degree elevation matrix for Bernstein polynomials.

# Arguments
- `canonical_space::Bernstein`: A Bernstein canonical space.
- `degree_delta::Int`: The amount to increase the degree of the canonical space by.

# Returns
- `::SparseMatrixCSC{Float64}`: A matrix mapping coefficients of the canonical space to
    coefficients of the elevated space.
"""
function build_degree_elevation_matrix(canonical_space::Bernstein, degree_delta::Int)
    if degree_delta < 0
        throw(ArgumentError("Degree delta must be non-negative. Got $degree_delta."))
    elseif degree_delta == 0
        p = get_polynomial_degree(canonical_space)
        return SparseArrays.sparse(Matrix{Float64}(LinearAlgebra.I, p + 1, p + 1))
    end

    p = get_polynomial_degree(canonical_space)
    
    # Elevate by 1 iteratively
    M = SparseArrays.sparse(Matrix{Float64}(LinearAlgebra.I, p + 1, p + 1))
    current_p = p
    for _ in 1:degree_delta
        E = zeros(Float64, current_p + 2, current_p + 1)
        for k in 1:(current_p + 1)
            E[k, k] = (current_p + 2 - k) / (current_p + 1)
            E[k + 1, k] = (k - 1) / (current_p + 1)
        end
        M = SparseArrays.sparse(E) * M
        current_p += 1
    end
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, M)
    return M
end

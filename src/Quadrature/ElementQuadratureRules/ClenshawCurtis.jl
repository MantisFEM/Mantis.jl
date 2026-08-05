"""
    clenshaw_curtis(p::Integer)

Find the roots and weights for Clenshaw-Curtis quadrature on the interval [0, 1]. The roots
include the endpoints 0 and 1 and Clenshaw-Curtis quadrature is exact for polynomials up to
degree `p`. However, in practise, this quadrature rule can obtain results comparable to
Gauss quadrature (in some cases), see [Trefethen2008](@cite) and [Trefethen2022](@cite).

# Arguments
- `p::Integer`: Degree of the quadrature rule.

# Returns
- `::CanonicalQuadratureRule{1}`: 1 dimensional quadrature rule containing the nodes and weights.
    There will be `p+1` nodes and weights.

# Notes
See [Waldvogel2006](@cite) for the algorithm based on fast fourier transforms. The algorithm
used here is a direct translation from the given MATLAB code on page 201.
"""
function clenshaw_curtis(p::Integer)
    if p <= 1
        throw(
            DomainError("""\
          Invalid degree: $p. The degree for Clenshaw-Curtis quadrature must be greater \
          than 1.\
          """)
        )
    end

    N = 1:2:(p - 1)
    l = length(N)
    m = p - l

    v0 = vcat(2.0 ./ N ./ (N .- 2), 1 / N[end], zeros(m))
    v2 = -v0[1:(end - 1)] .- v0[end:-1:2]

    g0 = -ones(p)
    g0[l + 1] = g0[l + 1] + p
    g0[m + 1] = g0[m + 1] + p
    g = g0 ./ (p^2 - 1 + p % 2)

    w = real(FFTW.ifft(v2 .+ g))
    w = vcat(w, w[1])

    rts = cospi.(collect(Float64, range(0, p)) ./ p)
    ξ = rts[end:-1:begin]

    # Map roots and weights to the interval [0, 1].
    @. ξ = (ξ + 1.0) / 2.0
    @. w = 0.5 * w
    nodes = Points.TensorProductPoints((ξ,))

    return CanonicalQuadratureRule(nodes, w, "Clenshaw-Curtis")
end

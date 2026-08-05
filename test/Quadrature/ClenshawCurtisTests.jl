module ClenshawCurtisQuadratureTests

using Mantis

import LinearAlgebra

using Test

# Defines test functions and tolerances for quadrature rules.
include("QuadratureTestsSetup.jl")

# Clenshaw-Curtis Quadrature ---------------------------------------------------------------
# The degree used for the Clenshaw-Curtis quadrature rule does not affect how the nodes and
# weights are computed. So testing with a few different degrees should be enough.

# Using a step of 3 to test a few different degrees, both even and odd.
for p in range(2, 20; step=3)
    quad_rule = Quadrature.clenshaw_curtis(p)

    ξ = Quadrature.get_nodes(quad_rule)
    ξ_const = Points.get_input_points(ξ)[1]
    w = Quadrature.get_weights(quad_rule)

    # Check that the rule type is correct.
    @test Quadrature.get_label(quad_rule) == "Clenshaw-Curtis"

    # Constructor tests.
    @test typeof(quad_rule) <: Quadrature.CanonicalQuadratureRule{1}

    # Invalid constructor tests.
    # Test that the constructor throws an error for invalid degrees.
    @test_throws DomainError Quadrature.clenshaw_curtis(1)
    @test_throws DomainError Quadrature.clenshaw_curtis(0)
    @test_throws DomainError Quadrature.clenshaw_curtis(-1)

    # Property tests.
    @test isapprox(sum(w), 1.0, atol=atol)
    @test all(w .> 0.0)
    @test isapprox(w, reverse(w), atol=atol)
    @test sum((ξ_const .< 0.0) .& (ξ_const .> 1.0)) == 0

    # Value tests on [0,1].
    # Test that the quadrature rule is exact for polynomials of degree up to p (with p the
    # degree, meaning that Clenshaw-Curtis quadrature will use p+1 nodes).
    for degree in 0:p
        f = monomial(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(degree, 1.0) - integrated_monomial(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)

        f = chebyshev(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev(degree, 1.0) - integrated_chebyshev(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)
    end

    # Test that the quadrature rule is not exact for polynomials of degree p+1.

    # The monomials can be exactly integrated for degrees up to 2p. This is because they
    # have a numerical order of 0.5p, see Trefethen2022.
    f = monomial(2 * p, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_monomial(2 * p, 1.0) - integrated_monomial(2 * p, 0.0)
    @test !isapprox(I_num, I; atol=atol)

    # The higher accuracy does not hold for the Chebyshev polynomials. However, Clenshaw-
    # Curtis quadrature is exact for Chebyshev polynomials when p is even, see
    # Trefethen2022.
    f = chebyshev(p + 1, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_chebyshev(p + 1, 1.0) - integrated_chebyshev(p + 1, 0.0)
    if p % 2 == 0
        @test isapprox(I_num, I, atol=atol) # Exact for even p.
    else
        @test !isapprox(I_num, I; atol=atol) # Not exact for odd p.
    end
end

end

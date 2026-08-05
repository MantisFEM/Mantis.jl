module GaussQuadratureTests

using Mantis

import LinearAlgebra

using Test

# Defines test functions and tolerances for quadrature rules.
include("QuadratureTestsSetup.jl")

# The Gauss quadrature rules use the FastGaussQuadrature.jl package, so we will focus on
# testing our modifications. A few additional tests for the correctness of the rules are
# added in case there is a bug in the FastGaussQuadrature.jl package.

# Gauss-Lobatto ----------------------------------------------------------------------------
for N in range(2, 17; step=3)
    quad_rule = Quadrature.gauss_lobatto(N)

    ξ = Quadrature.get_nodes(quad_rule)
    ξ_const = Points.get_input_points(ξ)[1]
    w = Quadrature.get_weights(quad_rule)

    # Check that the rule type is correct.
    @test Quadrature.get_label(quad_rule) == "Gauss-Lobatto"

    # Constructor tests.
    @test typeof(quad_rule) <: Quadrature.CanonicalQuadratureRule{1}

    # Invalid constructor tests.
    # Test that the constructor throws an error for invalid degrees.
    @test_throws DomainError Quadrature.gauss_lobatto(1)
    @test_throws DomainError Quadrature.gauss_lobatto(0)
    @test_throws DomainError Quadrature.gauss_lobatto(-1)

    # Property tests.
    @test isapprox(sum(w), 1.0, atol=atol)
    @test all(w .> 0.0)
    @test isapprox(w, reverse(w), atol=atol)
    @test sum((ξ_const .< 0.0) .& (ξ_const .> 1.0)) == 0
    # Test that the endpoints are included.
    @test isapprox(ξ_const[1], 0.0, atol=atol)
    @test isapprox(ξ_const[end], 1.0, atol=atol)

    # Value tests on [0,1].
    # Test that the quadrature rule is exact for polynomials of degree up to 2N-3 (with N
    # the number of nodes).
    for degree in 0:(2 * N - 3)
        f = monomial(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(degree, 1.0) - integrated_monomial(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)

        f = chebyshev(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev(degree, 1.0) - integrated_chebyshev(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)
    end

    # Test that the quadrature rule is not exact for polynomials of degree 2N-2.

    # The monomials can be exactly integrated for degrees up to 4*N-6. This is because they
    # have a numerical order of 0.5p, see Trefethen2022.
    f = monomial(4 * N - 6, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_monomial(4 * N - 6, 1.0) - integrated_monomial(4 * N - 6, 0.0)
    @test !isapprox(I_num, I; atol=atol)

    # The higher accuracy does not hold for the Chebyshev polynomials.
    f = chebyshev(2 * N - 2, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_chebyshev(2 * N - 2, 1.0) - integrated_chebyshev(2 * N - 2, 0.0)
    @test !isapprox(I_num, I; atol=atol)
end

# Gauss-Legendre ---------------------------------------------------------------------------
for N in range(1, 16; step=3)
    quad_rule = Quadrature.gauss_legendre(N)

    ξ = Quadrature.get_nodes(quad_rule)
    ξ_const = Points.get_input_points(ξ)[1]
    w = Quadrature.get_weights(quad_rule)

    # Check that the rule type is correct.
    @test Quadrature.get_label(quad_rule) == "Gauss-Legendre"

    # Constructor tests.
    @test typeof(quad_rule) <: Quadrature.CanonicalQuadratureRule{1}

    # Invalid constructor tests.
    # Test that the constructor throws an error for invalid degrees.
    @test_throws DomainError Quadrature.gauss_legendre(0)
    @test_throws DomainError Quadrature.gauss_legendre(-1)

    # Property tests.
    @test isapprox(sum(w), 1.0, atol=atol)
    @test all(w .> 0.0)
    @test isapprox(w, reverse(w), atol=atol)
    @test sum((ξ_const .< 0.0) .& (ξ_const .> 1.0)) == 0

    # Value tests on [0,1].
    # Test that the quadrature rule is exact for polynomials of degree up to 2N-1 (with N
    # the number of nodes).
    for degree in 0:(2 * N - 1)
        f = monomial(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(degree, 1.0) - integrated_monomial(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)

        f = chebyshev(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev(degree, 1.0) - integrated_chebyshev(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)
    end

    # Test that the quadrature rule is not exact for polynomials of
    # degree 2N.

    # The monomials can be exactly integrated for degrees up to 4*N. This is because they
    # have a numerical order of 0.5p, see Trefethen2022.
    f = monomial(4 * N, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_monomial(4 * N, 1.0) - integrated_monomial(4 * N, 0.0)
    @test !isapprox(I_num, I; atol=atol)

    # The higher accuracy does not hold for the Chebyshev polynomials.
    f = chebyshev(2 * N, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_chebyshev(2 * N, 1.0) - integrated_chebyshev(2 * N, 0.0)
    @test !isapprox(I_num, I; atol=atol)
end

end

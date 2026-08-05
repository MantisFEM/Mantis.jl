module NewtonCotesQuadratureTests

using Mantis

import LinearAlgebra

using Test

# Defines test functions and tolerances for quadrature rules.
include("QuadratureTestsSetup.jl")

# The degree used for the Newton-Cotes quadrature rule does not affect how the nodes and
# weights are computed. So testing with a few different degrees should be enough.

# Closed rules -----------------------------------------------------------------------------
for N in range(2, 12; step=3)
    quad_rule = Quadrature.newton_cotes(N, "closed")

    ξ = Quadrature.get_nodes(quad_rule)
    ξ_const = Points.get_input_points(ξ)[1]
    w = Quadrature.get_weights(quad_rule)

    # Check that the rule type is correct.
    @test Quadrature.get_label(quad_rule) == "Newton-Cotes (closed)"

    # Constructor tests.
    @test typeof(quad_rule) <: Quadrature.CanonicalQuadratureRule{1}

    # Invalid constructor tests.
    # Test that the constructor throws an error for invalid degrees.
    @test_throws DomainError Quadrature.newton_cotes(1, "closed")
    @test_throws DomainError Quadrature.newton_cotes(0, "closed")
    @test_throws DomainError Quadrature.newton_cotes(-1, "closed")

    @test_throws ArgumentError Quadrature.newton_cotes(1, "test")

    # Property tests.
    @test isapprox(sum(w), 1.0, atol=atol)
    @test isapprox(w, reverse(w), atol=atol)
    @test sum((ξ_const .< 0.0) .& (ξ_const .> 1.0)) == 0
    # Test that the endpoints are included.
    @test isapprox(ξ_const[1], 0.0, atol=atol)
    @test isapprox(ξ_const[end], 1.0, atol=atol)

    # Value tests on [0,1].
    # Test that the quadrature rule is exact for polynomials of degree up to N-1 (with N the
    # number of nodes).
    for degree in 0:(N - 1)
        f = monomial(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(degree, 1.0) - integrated_monomial(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)

        f = chebyshev(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev(degree, 1.0) - integrated_chebyshev(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)
    end

    # Test that the quadrature rule is not exact for polynomials of degree N.

    # For Newton-Cotes rules, the error for the integration of a monomial of degree N will
    # be zero when N is odd.
    f = monomial(N, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_monomial(N, 1.0) - integrated_monomial(N, 0.0)
    if N % 2 != 0
        @test isapprox(I_num, I, atol=atol)
    else
        @test !isapprox(I_num, I; atol=atol)
    end

    f = chebyshev(N + 1, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_chebyshev(N + 1, 1.0) - integrated_chebyshev(N + 1, 0.0)
    @test !isapprox(I_num, I; atol=atol)
end

# Open rules -------------------------------------------------------------------------------
for N in range(1, 10; step=3)
    quad_rule = Quadrature.newton_cotes(N, "open")

    ξ = Quadrature.get_nodes(quad_rule)
    ξ_const = Points.get_input_points(ξ)[1]
    w = Quadrature.get_weights(quad_rule)

    # Check that the rule type is correct.
    @test Quadrature.get_label(quad_rule) == "Newton-Cotes (open)"

    # Constructor tests.
    @test typeof(quad_rule) <: Quadrature.CanonicalQuadratureRule{1}

    # Invalid constructor tests.
    # Test that the constructor throws an error for invalid degrees.
    @test_throws DomainError Quadrature.newton_cotes(0, "open")
    @test_throws DomainError Quadrature.newton_cotes(-1, "open")

    @test_throws ArgumentError Quadrature.newton_cotes(1, "test")

    # Property tests.
    @test isapprox(sum(w), 1.0, atol=atol)
    @test isapprox(w, reverse(w), atol=atol)
    @test sum((ξ_const .< 0.0) .& (ξ_const .> 1.0)) == 0

    # Value tests on [0,1].
    # Test that the quadrature rule is exact for polynomials of degree up to N-1 (with N
    # the number of nodes).
    for degree in 0:(N - 1)
        f = monomial(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(degree, 1.0) - integrated_monomial(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)

        f = chebyshev(degree, ξ_const)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev(degree, 1.0) - integrated_chebyshev(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)
    end

    # Test that the quadrature rule is not exact for polynomials of degree N.

    # For Newton-Cotes rules, the error for the integration of a monomial of degree N will
    # be zero when N is odd.
    f = monomial(N, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_monomial(N, 1.0) - integrated_monomial(N, 0.0)
    if N % 2 != 0
        @test isapprox(I_num, I, atol=atol)
    else
        @test !isapprox(I_num, I; atol=atol)
    end

    f = chebyshev(N + 1, ξ_const)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_chebyshev(N + 1, 1.0) - integrated_chebyshev(N + 1, 0.0)
    @test !isapprox(I_num, I; atol=atol)
end

end

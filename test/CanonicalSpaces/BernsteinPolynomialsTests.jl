"""
Tests for the Bernstein polynomials. These tests are based on the 
standard properties of the Bernstein polynomials, see 
https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties.
"""


import Mantis

using Test

# Degree of the polynomial on which the tests are performed.
degrees_to_test = 0:25

for p in degrees_to_test
    # Gauss-Legendre quadrature rule of degree q
    q = max(2, ceil(Int, (p+1)/2))
    x, w = Mantis.Quadrature.gauss_legendre(q)
    
    sum_all = zeros(size(x))
    sum_all2 = zeros(size(x))

    # Bernstein polynomials of degree p ...
    b = Mantis.FunctionSpaces.Bernstein(p)
    # ... evaluated (values, 1st and 2nd derivatives) at quadrature nodes
    b_eval = Mantis.FunctionSpaces.evaluate(b, x, 2)

    # Check positivity of the polynomials
    @test minimum(b_eval[:,:,1]) >= 0.0

    # Check constant definite integral
    @test isapprox(maximum(abs.(w'*b_eval[:,:,1] .- 1.0/(p + 1))), 0.0, atol=1e-15)

    # Check partition of unity
    @test all(isapprox.(sum(b_eval[:,:,1], dims=2), 1.0))

    # Check zero sum of derivatives
    @test all(isapprox.(abs.(sum(b_eval[:,:,2], dims=2)), 0.0, atol=1e-14))

    # Check that Greville points represent the polynomial \xi
    @test all(isapprox.(b_eval[:,:,1] * LinRange(0,p,p+1), p.*x))

    if p > 0
        # Test correctness of evaluation and derivatives
        # Check the first and second derivatives with respect to polynomial
        #   f = x^{p} + x^{p-1}
        #   df/dx = p x^{p-1} + (p-1) x^{p-2}
        #   d2f/dx2 = p(p-1) x^{p-2} + (p-1)(p-2) x^{p-3}
        f = (x::Float64,p::Int64,m::Int64) -> (p - m >= 0 ? 1.0 : 0.0) * prod(LinRange(p:-1:(p-m+1))) * (p-m > 0 ? x^(p-m) : 1.0) + (p - 1 - m >= 0 ? 1.0 : 0.0)* prod(LinRange((p-1):-1:(p-m))) * (p-1-m > 0 ? x^(p-1-m) : 1.0)
        f_eval = f.(x, p, 0)  # the polynomial at evaluation points
        df_dx_eval = f.(x, p, 1)  # the first derivative at evaluation points
        d2f_dx2_eval = f.(x, p, 2)  # the second derivative at evaluation points
        # Coefficients of f in terms of the monomial basis ...
        coeff_m = [zeros(p-1); 1.0; 1.0]
        # ... and in terms of the Bernstein basis
        coeff_b = Mantis.FunctionSpaces.extract_monomial_to_bernstein(b) * coeff_m
        # Check that the values match f ...
        @test isapprox(maximum(abs.(b_eval[:,:,1] * coeff_b .- f_eval)), 0.0, atol = 1e-15)
        # ... the first order derivative matches df/dx ...
        @test isapprox(maximum(abs.(b_eval[:,:,2] * coeff_b .- df_dx_eval)), 0.0, atol = 2e-14)
        # ... and the second order derivative matches d2f/dx2.
        if p > 1
            @test isapprox(maximum(abs.(b_eval[:,:,3] * coeff_b .- d2f_dx2_eval)), 0.0, atol = 2e-13)
        end
    end
end

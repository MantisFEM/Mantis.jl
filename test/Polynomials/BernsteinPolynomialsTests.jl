"""
Tests for the Bernstein polynomials. These tests are based on the 
standard properties of the Bernstein polynomials, see 
https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties.
"""


import Mantis

using Test

# Degree of the polynomial on which the tests are performed. Since the 
# Bernstein implementation is explicitly written out for p = 0 to p = 3,
# those are all tested. A few extra for the general case are also 
# included.
degrees_to_test = 0:25

for p in degrees_to_test
    # Lobatto quadrature includes the endpoints, which can also be used
    # for additional test. The degree is more than large enough anyway.
    # The + is needed to have more than 0 roots when p = 0. Note that 
    # the standard quadrature rule is defined on [-1, 1], while we need 
    # it on [0, 1].
    q = max(2, ceil(Int, (p+1)/2))
    x, w = Mantis.Quadrature.gauss_legendre(q)
    
    sum_all = zeros(size(x))
    sum_all2 = zeros(size(x))

    b = Mantis.Polynomials.Bernstein(p)
    b_eval = Mantis.Polynomials.evaluate(b, x, 1)

    # Positivity of the polynomials
    @test minimum(b_eval[:,:,1]) >= 0.0

    # Constant definite integral
    @test isapprox(maximum(abs.(w'*b_eval[:,:,1] .- 1.0/(p + 1))), 0.0, atol=1e-15)

    # Partition of unity
    @test all(isapprox.(sum(b_eval[:,:,1], dims=2), 1.0))

    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(b_eval[:,:,2], dims=2)), 0.0, atol=1e-14))

    # Other
    @test all(isapprox.(b_eval[:,:,1] * LinRange(0,p,p+1), p.*x))

    # # Symmetry
    # b_plp = [Mantis.Polynomials.polynomial_bernstein(p, p-l, 1.0-xi) for xi in x]
    # @test isapprox(b_lp, b_plp)

    # # Root at zero
    # if l == 0
    #     # The zero-th polynomial is one at x=0.0
    #     @test isapprox(b_lp[1], 1.0)
    # else
    #     @test isapprox(b_lp[1], 0.0)
    # end

    # # Root at one
    # if l == p
    #     # The last polynomial is one at x=1.0
    #     @test isapprox(b_lp[end], 1.0)
    # else
    #     @test isapprox(b_lp[end], 0.0)
    # end

end

module BSplineTests

"""
Tests for the univariate spline spaces. These tests are based on the
standard properties of Bezier curves. See
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Properties.
"""

using Mantis

using Test

deg1 = 3
breakpoints = LinRange(0.0, 1.0, 3)
geometry = Geometry.CartesianGeometry(breakpoints)
B1_univariate_bs = FunctionSpaces.BSplineSpace(geometry, deg1, [-1, 1, -1])
quad_rule = Quadrature.gauss_legendre(deg1 + 1)
x1 = Quadrature.get_nodes(quad_rule)

@test FunctionSpaces.get_component_spaces(B1_univariate_bs) == (B1_univariate_bs,)
for el in 1:1:FunctionSpaces.get_num_elements(B1_univariate_bs)
    # check extraction coefficients
    ex_coeffs = FunctionSpaces.get_extraction_coefficients(B1_univariate_bs, el)
    # Test for non-negativity
    @test all(ex_coeffs .>= 0.0)
    # Test for partition of unity
    @test all(isapprox.(sum(ex_coeffs; dims=2) .- 1.0, 0.0, atol=1e-14))

    # check B-spline evaluation
    B1_eval, _ = FunctionSpaces.evaluate(B1_univariate_bs, el, x1, 1)
    # Positivity of the polynomials
    @test all(B1_eval[1][1][1] .>= 0.0)
    # Partition of unity
    @test all(isapprox.(sum(B1_eval[1][1][1]; dims=2), 1.0))
    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(B1_eval[2][1][1]; dims=2)), 0.0, atol=1e-14))
end

breakpoints2 = [0.0, 0.5, 0.6, 1.0]
deg2 = 4
geometry2 = Geometry.CartesianGeometry(breakpoints2)
B2_univariate_bs = FunctionSpaces.BSplineSpace(geometry2, deg2, [-1, 1, 3, -1])
quad_rule2 = Quadrature.gauss_legendre(deg2 + 1)
x2 = Quadrature.get_nodes(quad_rule2)
@test FunctionSpaces.get_component_spaces(B2_univariate_bs) == (B2_univariate_bs,)
for el in 1:1:FunctionSpaces.get_num_elements(B2_univariate_bs)
    # check extraction coefficients
    ex_coeffs = FunctionSpaces.get_extraction_coefficients(B2_univariate_bs, el)

    # Test for non-negativity
    @test all(ex_coeffs .>= 0.0)
    # Test for partition of unity
    @test all(isapprox.(sum(ex_coeffs; dims=2) .- 1.0, 0.0, atol=1e-14))

    # check B-spline evaluation
    B2_eval, _ = FunctionSpaces.evaluate(B2_univariate_bs, el, x2, 1)
    # Positivity of the polynomials
    @test minimum(B2_eval[1][1][1]) >= 0.0
    # Partition of unity
    @test all(isapprox.(sum(B2_eval[1][1][1]; dims=2), 1.0))
    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(B2_eval[2][1][1]; dims=2)), 0.0, atol=1e-14))
end

breakpoints3 = [0.0, 0.5, 0.6, 1.0]
deg3 = 2
geometry3 = Geometry.CartesianGeometry(breakpoints3)
Bsp_univariate = FunctionSpaces.BSplineSpace(geometry3, deg3, [-1, 1, 1, -1])
weights = [1.0, 2.0, 2.0, 3.0, 1.0]
Nurbs_univariate = FunctionSpaces.RationalFESpace(Bsp_univariate, weights)
quad_rule = Quadrature.gauss_legendre(deg3 + 1)
x3 = Quadrature.get_nodes(quad_rule)
@test FunctionSpaces.get_component_spaces(Bsp_univariate) == (Bsp_univariate,)
for el in 1:1:FunctionSpaces.get_num_elements(Nurbs_univariate)
    # check Nurbs evaluation
    Nurbs_eval, _ = FunctionSpaces.evaluate(Nurbs_univariate, el, x3, 0)
    # Positivity of the polynomials
    @test minimum(Nurbs_eval[1][1][1]) >= 0.0
end

# Test Ck-smooth GeneralizedExponential spline space ---------------------------------------
deg4 = 5
Wt = 10.0
b = FunctionSpaces.GeneralizedExponential(deg4, Wt, 0.25)
breakpoints4 = [0.0, 0.25, 0.5, 0.75]
geometry4 = Geometry.CartesianGeometry(breakpoints4)
B = FunctionSpaces.BSplineSpace(geometry4, b, [-1, deg4 - 1, deg4 - 1, -1])
nbasis = FunctionSpaces.get_num_basis(B)
nel = FunctionSpaces.get_num_elements(B)
@test FunctionSpaces.get_component_spaces(B) == (B,)
for el in 1:1:nel
    # check extraction coefficients
    ex_coeffs = FunctionSpaces.get_extraction_coefficients(B, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs; dims=2) .- 1.0, 0.0, atol=5e-14)) # Test for partition of unity
end

# interpolate an exponential
x = range(; start=0.1, stop=0.9, length=deg4 + 1)
xi = Points.CartesianPoints((x,))
npts = deg4 + 1
LHS = zeros(nel * npts, nbasis)
RHS_P = zeros(nel * npts)
RHS_N = zeros(nel * npts)
for el in 1:1:FunctionSpaces.get_num_elements(B)
    B_eval, inds = FunctionSpaces.evaluate(B, el, xi, 0)
    LHS[(el - 1) * npts .+ (1:npts), inds] = B_eval[1][1][1]
    RHS_P[(el - 1) * npts .+ (1:npts)] = exp.(Wt .* (x .+ (el - 1)) .* 0.25)
    RHS_N[(el - 1) * npts .+ (1:npts)] = exp.(-Wt .* (x .+ (el - 1)) .* 0.25)
end
coeffs_P = LHS \ RHS_P
coeffs_N = LHS \ RHS_N
@test all(isapprox.(abs.(LHS * coeffs_P - RHS_P), 0.0, atol=1e-10))
@test all(isapprox.(abs.(LHS * coeffs_N - RHS_N), 0.0, atol=1e-14))

end

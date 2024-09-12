"""
Tests for the univariate spline spaces. These tests are based on the 
standard properties of Bezier curves. See 
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Properties.
"""

import Mantis

using Test

deg1 = 3
breakpoints = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints)
B1_univariate_bs = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, 1, -1])
quad_rule = Mantis.Quadrature.gauss_legendre(deg1+1)
x = Mantis.Quadrature.get_quadrature_nodes(quad_rule)[1]
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(B1_univariate_bs)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(B1_univariate_bs, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity

    # check B-spline evaluation
    B1_eval, _ = Mantis.FunctionSpaces.evaluate(B1_univariate_bs, el, (x,), 1)
    # Positivity of the polynomials
    @test minimum(B1_eval[1][1]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(B1_eval[1][1], dims=2), 1.0))

    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(B1_eval[2][1], dims=2)), 0.0, atol=1e-14))
end

breakpoints = [0.0, 0.5, 0.6, 1.0]
deg2 = 4
patch2 = Mantis.Mesh.Patch1D(breakpoints)
B2_univariate_bs = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, 1, 3, -1])
quad_rule2 = Mantis.Quadrature.gauss_legendre(deg2+1)
x = Mantis.Quadrature.get_quadrature_nodes(quad_rule2)[1]
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(B2_univariate_bs)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(B2_univariate_bs, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity

    # check B-spline evaluation
    B2_eval, _ = Mantis.FunctionSpaces.evaluate(B2_univariate_bs, el, (x,), 1)
    # Positivity of the polynomials
    @test minimum(B2_eval[1][1]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(B2_eval[1][1], dims=2), 1.0))

    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(B2_eval[2][1], dims=2)), 0.0, atol=1e-14))
end

GB = Mantis.FunctionSpaces.GTBSplineSpace((B1_univariate_bs, B2_univariate_bs), [1, -1])
quad_rule3 = Mantis.Quadrature.gauss_legendre(max(deg1,deg2)+1)
x = Mantis.Quadrature.get_quadrature_nodes(quad_rule3)[1]
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(GB, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity
    # check GTB-spline evaluation
    GB_eval, _ = Mantis.FunctionSpaces.evaluate(GB, el, (x,), 1)
end

# Test C1-smooth TrigonometricSplineSpace ----------------------------------------------------
deg = 2
npatch = 4
Wt = 2.0*pi/npatch
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B_C1_smooth_trig = ntuple( i -> b, npatch)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B_C1_smooth_trig, ones(Int,npatch))
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(GB, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity
end

# interpolate a cosine and a sine
x = [0.5]
LHS = zeros(npatch,npatch)
RHS_C = zeros(npatch)
RHS_S = zeros(npatch)
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    GB_eval, inds = Mantis.FunctionSpaces.evaluate(GB, el, (x,), 0)
    LHS[el,inds] = GB_eval[1][1]
    RHS_C[el] = cos(Wt * (x[1] + el - 1))
    RHS_S[el] = sin(Wt * (x[1] + el - 1))
end
coeffs_C = LHS \ RHS_C
coeffs_S = LHS \ RHS_S
x = Mantis.Quadrature.get_quadrature_nodes(quad_rule3)[1]
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    GB_eval, inds = Mantis.FunctionSpaces.evaluate(GB, el, (x,), 0)
    @test all(isapprox.(maximum(abs.(GB_eval[1][1]*coeffs_C[inds] - cos.(Wt * (x .+ (el - 1))))), 0.0, atol=1e-14))
    @test all(isapprox.(maximum(abs.(GB_eval[1][1]*coeffs_S[inds] - sin.(Wt * (x .+ (el - 1))))), 0.0, atol=1e-14))
end

breakpoints = [0.0, 0.5, 0.6, 1.0]
deg = 2
patch = Mantis.Mesh.Patch1D(breakpoints)
Bsp_univariate = Mantis.FunctionSpaces.BSplineSpace(patch, deg, [-1, 1, 1, -1])
weights = [1.0, 2.0, 2.0, 3.0, 1.0]
Nurbs_univariate = Mantis.FunctionSpaces.RationalFiniteElementSpace(Bsp_univariate, weights)
quad_rule = Mantis.Quadrature.gauss_legendre(deg+1)
x = Mantis.Quadrature.get_quadrature_nodes(quad_rule)[1]
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(Nurbs_univariate)
    # check Nurbs evaluation
    Nurbs_eval, _ = Mantis.FunctionSpaces.evaluate(Nurbs_univariate, el, (x,), 0)
    # Positivity of the polynomials
    @test minimum(Nurbs_eval[1][1]) >= 0.0
end
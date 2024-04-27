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
B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, 1, -1])
x, _ = Mantis.Quadrature.gauss_legendre(deg1+1)
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(B1)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(B1, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity

    # check B-spline evaluation
    B1_eval, _ = Mantis.FunctionSpaces.evaluate(B1, el, x, 1)
    # Positivity of the polynomials
    @test minimum(B1_eval[0]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(B1_eval[0], dims=2), 1.0))

    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(B1_eval[1], dims=2)), 0.0, atol=1e-14))
end

breakpoints = [0.0, 0.5, 0.6, 1.0]
deg2 = 4
patch2 = Mantis.Mesh.Patch1D(breakpoints)
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, 1, 3, -1])
x, _ = Mantis.Quadrature.gauss_legendre(deg2+1)
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(B2)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(B2, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity

    # check B-spline evaluation
    B2_eval, _ = Mantis.FunctionSpaces.evaluate(B2, el, x, 1)
    # Positivity of the polynomials
    @test minimum(B2_eval[0]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(B2_eval[0], dims=2), 1.0))

    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(B2_eval[1], dims=2)), 0.0, atol=1e-14))
end

GB = Mantis.FunctionSpaces.GTBSplineSpace((B1, B2), [1, -1])
x, _ = Mantis.Quadrature.gauss_legendre(max(deg1,deg2)+1)
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(GB, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity
    # check GTB-spline evaluation
    GB_eval, _ = Mantis.FunctionSpaces.evaluate(GB, el, x, 1)
end

# Test C1-smooth TrigonometricSplineSpace ----------------------------------------------------
deg = 2
Wt = pi/2
breakpoints = [0.0, 1.0]
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
x, _ = Mantis.Quadrature.gauss_legendre(max(deg1,deg2)+1)
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(GB, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity
    # check GTB-spline evaluation
    GB_eval, _ = Mantis.FunctionSpaces.evaluate(GB, el, x, 1)
end
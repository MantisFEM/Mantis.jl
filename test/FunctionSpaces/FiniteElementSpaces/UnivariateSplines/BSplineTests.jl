module BSplineTests

"""
Tests for the univariate spline spaces. These tests are based on the
standard properties of Bezier curves. See
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Properties.
"""

using Mantis

using Test

function basic_tests(bspline, answers)
    @test FunctionSpaces.get_manifold_dim(bspline) == answers[1]
    @test FunctionSpaces.get_num_components(bspline) == answers[2]
    @test FunctionSpaces.get_num_patches(bspline) == answers[3]

    @test FunctionSpaces.get_component_spaces(bspline) == answers[4]

    @test FunctionSpaces.get_polynomials(bspline) == answers[5]
    @test FunctionSpaces.get_polynomial_degree(bspline) == answers[6]
    @test FunctionSpaces.get_num_basis(bspline) == answers[7]
    @test FunctionSpaces.get_num_basis(bspline, 1) == answers[8]
    @test FunctionSpaces.get_max_local_dim(bspline) == answers[9]
    @test all(FunctionSpaces.get_multiplicity_vector(bspline) .== answers[10])
    @test all(FunctionSpaces.get_support(bspline, 1) .== answers[11])
    @test FunctionSpaces.get_geometry(bspline) == answers[12]
    @test FunctionSpaces.get_parametric_geometry(bspline) == answers[13]

    return nothing
end

# C1 cubic on two element. LinRange CartesiangGeometry input.
const deg1 = 3
const breakpoints = LinRange(0.0, 1.0, 3)
const geometry = Geometry.CartesianGeometry(breakpoints)
const B1_univariate_bs = FunctionSpaces.BSplineSpace(geometry, deg1, [-1, 1, -1])
const quad_rule = Quadrature.gauss_legendre(deg1 + 1)
const x1 = Quadrature.get_nodes(quad_rule)
# Equivalent ways to construct the same space.
const B1_univariate_bs_alt1 = FunctionSpaces.BSplineSpace(
    geometry, geometry, FunctionSpaces.Bernstein(deg1), [-1, 1, -1], 1, 1
)
const B1_univariate_bs_alt2 = FunctionSpaces.BSplineSpace(
    geometry, FunctionSpaces.Bernstein(deg1), [-1, 1, -1]
)
const B1_univariate_bs_alt3 = FunctionSpaces.BSplineSpace(geometry, deg1, 1)
const B1_univariate_bs_alt4 = FunctionSpaces.BSplineSpace(
    geometry, FunctionSpaces.Bernstein(deg1), 1
)
const mapping = Geometry.Mapping(Val(1), Val(1), x -> x[1], x -> zero(x[1]))  # identity map
const mapped_geo = Geometry.MappedGeometry(geometry, mapping)
const B1_univariate_bs_alt5 = FunctionSpaces.BSplineSpace(
    geometry, mapping, FunctionSpaces.Bernstein(deg1), [-1, 1, -1]
)
const B1_univariate_bs_alt6 = FunctionSpaces.BSplineSpace(
    mapped_geo, geometry, FunctionSpaces.Bernstein(deg1), [-1, 1, -1], 1, 1
)
const B1_univariate_alts = (
    B1_univariate_bs,
    B1_univariate_bs_alt1,
    B1_univariate_bs_alt2,
    B1_univariate_bs_alt3,
    B1_univariate_bs_alt4,
    B1_univariate_bs_alt5,
    B1_univariate_bs_alt6,
)

for i in eachindex(B1_univariate_alts)
    space = B1_univariate_alts[i]
    if i >= 6
        basic_tests(
            space,
            (
                1,  # manifold_dim
                1,  # num_components
                1,  # num_patches
                (space,),  # component spaces
                FunctionSpaces.Bernstein(deg1),  # polynomial
                deg1,  # degree
                6,  # num basis
                4,  # num basis on element 1
                deg1 + 1,  # max_local_dim
                [deg1 + 1, deg1 - 1, deg1 + 1],  # multiplicity vector
                [1],  # elements on which the 1st basis function is supported
                mapped_geo,  # physical geometry
                geometry,  # parametric geometry
            ),
        )
    else
        basic_tests(
            space,
            (
                1,  # manifold_dim
                1,  # num_components
                1,  # num_patches
                (space,),  # component spaces
                FunctionSpaces.Bernstein(deg1),  # polynomial
                deg1,  # degree
                6,  # num basis
                4,  # num basis on element 1
                deg1 + 1,  # max_local_dim
                [deg1 + 1, deg1 - 1, deg1 + 1],  # multiplicity vector
                [1],  # elements on which the 1st basis function is supported
                geometry,  # physical geometry
                geometry,  # parametric geometry
            ),
        )
    end
end

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

# 3-element quartic with mixed regularity. Vector CartesianGeometry input.
const breakpoints2 = [0.0, 0.5, 0.6, 1.0]
const deg2 = 4
const geometry2 = Geometry.CartesianGeometry(breakpoints2)
const B2_univariate_bs = FunctionSpaces.BSplineSpace(geometry2, deg2, [-1, 1, 3, -1])
const quad_rule2 = Quadrature.gauss_legendre(deg2 + 1)
const x2 = Quadrature.get_nodes(quad_rule2)

basic_tests(
    B2_univariate_bs,
    (
        1,  # manifold_dim
        1,  # num_components
        1,  # num_patches
        (B2_univariate_bs,),  # component spaces
        FunctionSpaces.Bernstein(deg2),  # polynomial
        deg2,  # degree
        9,  # num basis
        5,  # num basis on element 1
        deg2 + 1,  # max_local_dim
        [deg2 + 1, deg2 - 1, deg2 - 3, deg2 + 1],  # multiplicity vector
        [1],  # elements on which the 1st basis function is supported
        geometry2,  # physical geometry
        geometry2,  # parametric geometry
    ),
)

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

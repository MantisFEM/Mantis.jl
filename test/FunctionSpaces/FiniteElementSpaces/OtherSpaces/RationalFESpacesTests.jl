module RationalFESpacesTests

using Mantis
using Test

function basic_tests(bspline, answers)
    @test FunctionSpaces.get_manifold_dim(bspline) == answers[1]
    @test FunctionSpaces.get_num_components(bspline) == answers[2]
    @test FunctionSpaces.get_num_patches(bspline) == answers[3]

    @test FunctionSpaces.get_component_spaces(bspline) == answers[4]

    @test FunctionSpaces.get_polynomial_degree(bspline, 1) == answers[5]
    @test FunctionSpaces.get_num_basis(bspline) == answers[6]
    @test FunctionSpaces.get_num_basis(bspline, 1) == answers[7]
    @test FunctionSpaces.get_max_local_dim(bspline) == answers[8]
    @test FunctionSpaces.get_geometry(bspline) == answers[9]
    @test FunctionSpaces.get_parametric_geometry(bspline) == answers[10]

    return nothing
end

# 1D, 3-element C1 quadratic NURBS. Vector CartesianGeometry input.
const breakpoints = [0.0, 0.5, 0.6, 1.0]
const deg = 2
const geometry = Geometry.CartesianGeometry(breakpoints)
const Bsp_univariate = FunctionSpaces.BSplineSpace(geometry, deg, [-1, 1, 1, -1])
const weights = [1.0, 2.0, 2.0, 3.0, 1.0]
const Nurbs_univariate = FunctionSpaces.RationalFESpace(Bsp_univariate, weights)
const quad_rule = Quadrature.gauss_legendre(deg + 1)
const x = Quadrature.get_nodes(quad_rule)

basic_tests(
    Nurbs_univariate, (
        1,  # manifold_dim
        1,  # num_components
        1,  # num_patches
        (Nurbs_univariate,),  # component spaces
        deg,  # polynomial degree on the first element
        5,  # num basis
        3,  # num basis on element 1
        deg + 1,  # max_local_dim
        geometry,  # physical geometry
        geometry,  # parametric geometry
    )
)

for el in 1:1:FunctionSpaces.get_num_elements(Nurbs_univariate)
    # check Nurbs evaluation
    Nurbs_eval, _ = FunctionSpaces.evaluate(Nurbs_univariate, el, x, 0)
    # Positivity of the polynomials
    @test minimum(Nurbs_eval[1][1][1]) >= 0.0
end

end

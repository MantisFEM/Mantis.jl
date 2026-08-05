module TensorProductTests

"""
Tests for tensor-product spline spaces.
"""

using Mantis
using LinearAlgebra
using Test

###
### Basic tensor-product tests
###

# patch breakpoints in x and y
breakpoints1 = [0.0, 0.5, 0.8, 0.9, 1.0]
patch1 = Geometry.CartesianGeometry(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Geometry.CartesianGeometry(breakpoints2)
num_derivatives = 3
for deg1 in 0:5
    for deg2 in 0:5
        # first B-spline patch
        local B1 = FunctionSpaces.BSplineSpace(
            patch1, deg1, [-1, deg1 - 1, min(deg1 - 1, 0), deg1 - 1, -1]
        )
        # second B-spline patch
        local B2 = FunctionSpaces.BSplineSpace(
            patch2, deg2, [-1, min(deg2 - 1, 1), deg2 - 1, -1]
        )
        # tensor-product B-spline patch
        TP = FunctionSpaces.TensorProductSpace((B1, B2))
        TP1 = FunctionSpaces.TensorProductSpace((
            FunctionSpaces.TensorProductSpace((B1, B2)), B1
        ))
        TP2 = FunctionSpaces.TensorProductSpace((
            B1, FunctionSpaces.TensorProductSpace((B2, B1))
        ))
        TP3 = FunctionSpaces.TensorProductSpace((B1, B2, B1))
        # evaluation points
        qrule = Quadrature.tensor_product_rule(
            (deg1 + 1, deg2 + 1), Quadrature.gauss_legendre
        )
        qrule3 = Quadrature.tensor_product_rule(
            (deg1 + 1, deg2 + 1, deg1 + 1), Quadrature.gauss_legendre
        )
        x_all = Quadrature.get_nodes(qrule)
        x_all3 = Quadrature.get_nodes(qrule3)
        for el in 1:1:FunctionSpaces.get_num_elements(TP)
            # check B-spline evaluation
            TP_eval, _ = FunctionSpaces.evaluate(TP, el, x_all, 0)
            TP1_eval, _ = FunctionSpaces.evaluate(TP1, el, x_all3, num_derivatives)
            TP2_eval, _ = FunctionSpaces.evaluate(TP1, el, x_all3, num_derivatives)
            TP3_eval, _ = FunctionSpaces.evaluate(TP3, el, x_all3, num_derivatives)
            # Positivity of the polynomials
            @test minimum(TP_eval[1][1][1]) >= 0.0
            @test minimum(TP1_eval[1][1][1]) >= 0.0
            @test minimum(TP2_eval[1][1][1]) >= 0.0
            @test minimum(TP3_eval[1][1][1]) >= 0.0

            # Partition of unity
            @test all(isapprox.(sum(TP_eval[1][1][1]; dims=2), 1.0))
            @test all(isapprox.(sum(TP1_eval[1][1][1]; dims=2), 1.0))
            @test all(isapprox.(sum(TP2_eval[1][1][1]; dims=2), 1.0))
            @test all(isapprox.(sum(TP3_eval[1][1][1]; dims=2), 1.0))

            # Consistency of the evaluation
            for der in 1:num_derivatives
                for der_idx in eachindex(TP1_eval[der])
                    @test isapprox(TP1_eval[der][der_idx], TP2_eval[der][der_idx])
                    @test isapprox(TP2_eval[der][der_idx], TP3_eval[der][der_idx])
                end
            end
        end

        # tests for number of basis functions
        @test FunctionSpaces.get_num_basis(TP1) == FunctionSpaces.get_num_basis(TP2)
        @test FunctionSpaces.get_num_basis(TP2) == FunctionSpaces.get_num_basis(TP3)

        # tests for dof partitioning
        @test FunctionSpaces.get_dof_partition(TP1) == FunctionSpaces.get_dof_partition(TP2)
        @test FunctionSpaces.get_dof_partition(TP2) == FunctionSpaces.get_dof_partition(TP3)
        @test sum(length, FunctionSpaces.get_dof_partition(TP3)[1]) ==
            FunctionSpaces.get_num_basis(TP3)
    end
end

###
### Combination of tensor-product and multi-patch tests
###

# first B-spline patch
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Geometry.CartesianGeometry(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Geometry.CartesianGeometry(breakpoints2)
deg1 = 3
B1_tensor_prod = FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1 - 1, -1])
# second B-spline patch
deg2 = 4
B2_tensor_prod = FunctionSpaces.BSplineSpace(
    patch2, deg2, [-1, min(deg2 - 1, 1), deg2 - 1, -1]
)
# first multi-patch object
MP1 = FunctionSpaces.GTBSplineSpace((B1_tensor_prod, B2_tensor_prod), [1, -1])

# third B-spline patch
deg1 = 4
B3_tensor_prod = FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1 - 1, -1])
# second B-spline patch
deg2 = 3
B4_tensor_prod = FunctionSpaces.BSplineSpace(
    patch2, deg2, [-1, min(deg2 - 1, 1), deg2 - 1, -1]
)
# First multi-patch object
MP2 = FunctionSpaces.GTBSplineSpace((B3_tensor_prod, B4_tensor_prod), [1, -1])

# tensor-product B-spline patch
TP = FunctionSpaces.TensorProductSpace((MP1, MP2))
# evaluation points
x1 = LinRange(0.0, 1.0, 11)
x2 = LinRange(0.0, 1.0, 11)
for el in 1:1:FunctionSpaces.get_num_elements(TP)
    # check B-spline evaluation
    TP_eval, _ = FunctionSpaces.evaluate(TP, el, Points.TensorProductPoints((x1, x2)))
    # Positivity of the polynomials
    @test minimum(TP_eval[1][1][1]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(TP_eval[1][1][1]; dims=2), 1.0))
end

# Constructor, property, and getters and setters tests -------------------------------------
function basic_tests(space, answers, element_id=1, component_id=1)
    @test FunctionSpaces.get_manifold_dim(space) == answers[1]
    @test FunctionSpaces.get_num_components(space) == answers[2]
    @test FunctionSpaces.get_num_patches(space) == answers[3]

    @test all(FunctionSpaces.get_component_spaces(space) .== answers[4])
    @test FunctionSpaces.get_extraction(space, element_id, component_id) == answers[5]
    @test FunctionSpaces.get_extraction_coefficients(space, element_id, component_id) ==
        answers[6]
    @test FunctionSpaces.get_basis_indices(space, element_id) == answers[7]
    @test FunctionSpaces.get_basis_permutation(space, element_id, component_id) ==
        answers[8]
    @test FunctionSpaces.get_num_basis(space) == answers[9]
    @test FunctionSpaces.get_num_basis(space, element_id) == answers[10]
    @test FunctionSpaces.get_dof_partition(space) == answers[11]
    @test FunctionSpaces.get_max_local_dim(space) == answers[12]
    @test FunctionSpaces.get_geometry(space) == answers[13]
    @test FunctionSpaces.get_parametric_geometry(space) == answers[14]

    return nothing
end

# Reduction test, single-patch, single element, 1D, Cartesian, degree 0 BSplines.
geometry1 = Geometry.CartesianGeometry(([-1, 1],))
B0 = FunctionSpaces.BSplineSpace(
    geometry1, geometry1, FunctionSpaces.Bernstein(0), [-1, -1], 0, 0
)
TP_B0 = FunctionSpaces.TensorProductSpace((B0,))
answers_0 = (
    1,
    1,
    1,
    (TP_B0,),
    FunctionSpaces.get_extraction(B0, 1, 1),
    FunctionSpaces.get_extraction_coefficients(B0, 1, 1),
    [1],
    1:1,
    1,
    1,
    [[[], [1], []]],
    1,
    Geometry.TensorProductGeometry((geometry1,)),
    Geometry.TensorProductGeometry((geometry1,)),
)
basic_tests(TP_B0, answers_0)

# Reduction test, single-patch, single element, 1D, Cartesian, degree 1 BSplines.
B1 = FunctionSpaces.BSplineSpace(
    geometry1, geometry1, FunctionSpaces.Bernstein(1), [-1, -1], 1, 1
)
TP_B1_a = FunctionSpaces.TensorProductSpace((B1,))
answers_1a = (
    1,
    1,
    1,
    (TP_B1_a,),
    FunctionSpaces.get_extraction(B1, 1, 1),
    FunctionSpaces.get_extraction_coefficients(B1, 1, 1),
    [1, 2],
    1:2,
    2,
    2,
    [[[1], [], [2]]],
    2,
    Geometry.TensorProductGeometry((geometry1,)),
    Geometry.TensorProductGeometry((geometry1,)),
)
basic_tests(TP_B1_a, answers_1a)

# Reduction test, single-patch, single element, 1D, TensorProduct, degree 1 BSplines.
TP_B1_b = FunctionSpaces.TensorProductSpace((B1,), Geometry.CartesianGeometry)
answers_1b = (
    1,
    1,
    1,
    (TP_B1_b,),
    FunctionSpaces.get_extraction(B1, 1, 1),
    FunctionSpaces.get_extraction_coefficients(B1, 1, 1),
    [1, 2],
    1:2,
    2,
    2,
    [[[1], [], [2]]],
    2,
    geometry1,
    geometry1,
)
basic_tests(TP_B1_b, answers_1b)

# Reduction test, single-patch, single element, 1D, Cartesian, degree 3 Lagrange.
nodes = Points.get_input_points(Quadrature.get_nodes(Quadrature.gauss_lobatto(4)))[1]
ll_polynomial = FunctionSpaces.Lagrange(nodes)
L3 = FunctionSpaces.BSplineSpace(
    geometry1, geometry1, ll_polynomial, [-1, -1]
)
TP_L3 = FunctionSpaces.TensorProductSpace((L3,))
answers_L3 = (
    1,
    1,
    1,
    (TP_L3,),
    FunctionSpaces.get_extraction(L3, 1, 1),
    LinearAlgebra.I,
    [1, 2, 3, 4], # basis indices on element_id
    1:4, # basis permutation on element_id
    4, # total num basis
    4, # num basis on element_id
    [[[1], [2, 3], [4]]], # dof partition
    4, # max local dim
    Geometry.TensorProductGeometry((geometry1,)),
    Geometry.TensorProductGeometry((geometry1,)),
)
basic_tests(TP_L3, answers_L3)

# Single-patch, multi-element, 1D, TensorProduct, degree 4 BSplines.
geometry2 = Geometry.CartesianGeometry(([-1, -0.9, -0.2, 0.1, 0.3, 0.85, 1],))
Bmulti = FunctionSpaces.BSplineSpace(
    geometry2, geometry2, FunctionSpaces.Bernstein(4), [-1, 3, 3, 3, 3, 2, -1], 1, 1
)
TP_Bmulti = FunctionSpaces.TensorProductSpace((Bmulti,))
answers_TP_Bmulti = (
    1,
    1,
    1,
    (TP_Bmulti,),
    FunctionSpaces.get_extraction(Bmulti, 3, 1),
    FunctionSpaces.get_extraction_coefficients(Bmulti, 3, 1),
    [3, 4, 5, 6, 7], # basis indices on element_id
    1:5, # basis permutation on element_id
    11, # total num basis
    5, # num basis on element_id
    [[[1], [2, 3, 4, 5, 6, 7, 8, 9, 10], [11]]], # dof partition
    5, # max local dim
    Geometry.TensorProductGeometry((geometry2,)),
    Geometry.TensorProductGeometry((geometry2,)),
)
basic_tests(TP_Bmulti, answers_TP_Bmulti, 3, 1)

# Reduction test, single-patch, single element, 2D, TensorProduct, degree 0 BSplines.
TP_B0B0 = FunctionSpaces.TensorProductSpace((B0, B0))
answers_TP_B0B0 = (
    2,
    1,
    1,
    (TP_B0B0,),
    (LinearAlgebra.I, 1:1),
    LinearAlgebra.I,
    [1], # basis indices on element_id
    1:1, # basis permutation on element_id
    1, # total num basis
    1, # num basis on element_id
    [[[], [], [], [], [1], [], [], [], []]], # dof partition
    1, # max local dim
    Geometry.TensorProductGeometry((geometry1, geometry1)),
    Geometry.TensorProductGeometry((geometry1, geometry1)),
)
basic_tests(TP_B0B0, answers_TP_B0B0, 1, 1)

# Reduction test, single-patch, single element, 2D, TensorProduct, degree 1 BSplines.
TP_B1B1 = FunctionSpaces.TensorProductSpace((B1, B1))
answers_TP_B1B1 = (
    2,
    1,
    1,
    (TP_B1B1,),
    (LinearAlgebra.I, 1:4),
    LinearAlgebra.I,
    [1, 2, 3, 4], # basis indices on element_id
    1:4, # basis permutation on element_id
    4, # total num basis
    4, # num basis on element_id
    [[[1], [], [2], [], [], [], [3], [], [4]]], # dof partition
    4, # max local dim
    Geometry.TensorProductGeometry((geometry1, geometry1)),
    Geometry.TensorProductGeometry((geometry1, geometry1)),
)
basic_tests(TP_B1B1, answers_TP_B1B1, 1, 1)

# General methods that should error
# Since v1.12, accessing a non-defined field will return a FieldError. In earlier versions
# this was an ErrorException, so we have to account for that in (CI) testing.
@static if Base.VERSION >= v"1.12"
    const fielderror = FieldError
else
    const fielderror = ErrorException
end
@test_throws fielderror FunctionSpaces.get_extraction_operator(TP_B1_a)

end

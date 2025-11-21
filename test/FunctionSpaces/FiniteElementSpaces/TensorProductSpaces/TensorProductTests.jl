module TensorProductTests

"""
Tests for tensor-product spline spaces.
"""

using Mantis

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
    TP_eval, _ = FunctionSpaces.evaluate(TP, el, Points.CartesianPoints((x1, x2)))
    # Positivity of the polynomials
    @test minimum(TP_eval[1][1][1]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(TP_eval[1][1][1]; dims=2), 1.0))
end

end

"""
Tests for tensor-product spline spaces.
"""

import Mantis

using Test


###
### Basic tensor-product tests
###

# patch breakpoints in x and y
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
for deg1 in 0:5
    for deg2 in 0:5
        # first B-spline patch
        local B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
        # second B-spline patch
        local B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
        # tensor-product B-spline patch
        TP = Mantis.FunctionSpaces.TensorProductSpace(B1,B2,Dict())
        # evaluation points
        x1, _ = Mantis.Quadrature.gauss_legendre(deg1+1)
        x2, _ = Mantis.Quadrature.gauss_legendre(deg2+1)
        for el in 1:1:Mantis.FunctionSpaces.get_num_elements(TP)
            # check B-spline evaluation
            TP_eval, _ = Mantis.FunctionSpaces.evaluate(TP, el, (x1,x2),0)
            # Positivity of the polynomials
            @test minimum(TP_eval[0,0]) >= 0.0

            # Partition of unity
            @test all(isapprox.(sum(TP_eval[0,0], dims=2), 1.0))
        end
    end
end

###
### Combination of tensor-product and multi-patch tests
###

# first B-spline patch
deg1 = 3;
B1_tensor_prod = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
# second B-spline patch
deg2 = 4;
B2_tensor_prod = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
# first multi-patch object
MP1 = Mantis.FunctionSpaces.GTBSplineSpace((B1_tensor_prod, B2_tensor_prod), [1, -1])

# third B-spline patch
deg1 = 4;
B3_tensor_prod = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
# second B-spline patch
deg2 = 3;
B4_tensor_prod = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
# First multi-patch object
MP2 = Mantis.FunctionSpaces.GTBSplineSpace((B3_tensor_prod, B4_tensor_prod), [1, -1])

# tensor-product B-spline patch
TP = Mantis.FunctionSpaces.TensorProductSpace(MP1, MP2, Dict())
# evaluation points
x1 = collect(LinRange(0.0,1.0,11))
x2 = collect(LinRange(0.0,1.0,11))
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(TP)
    # check B-spline evaluation
    TP_eval, _ = Mantis.FunctionSpaces.evaluate(TP, el, (x1,x2))
    # Positivity of the polynomials
    @test minimum(TP_eval[0,0]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(TP_eval[0,0], dims=2), 1.0))
end
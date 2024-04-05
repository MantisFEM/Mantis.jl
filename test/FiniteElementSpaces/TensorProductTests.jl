"""
Tests for tensor-product spline spaces.
"""

import Mantis

using Test

# patch breakpoints in x and y
breakpoints1 = [0.0, 0.5, 1.0]
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
for deg1 in 0:5
    for deg2 in 0:5
        # first B-spline patch
        patch1 = Mantis.Mesh.Patch1D(breakpoints1)
        B1 = Mantis.FiniteElementSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
        # second B-spline patch
        patch2 = Mantis.Mesh.Patch1D(breakpoints2)
        B2 = Mantis.FiniteElementSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
        # tensor-product B-spline patch
        TP = Mantis.FiniteElementSpaces.TensorProductSpace((B1,B2), Dict())
        # evaluation points
        x1, _ = Mantis.Quadrature.gauss_legendre(deg1+1)
        x2, _ = Mantis.Quadrature.gauss_legendre(deg2+1)
        for el in 1:1:Mantis.FiniteElementSpaces.get_num_elements(TP)
            # check B-spline evaluation
            TP_eval, _ = Mantis.FiniteElementSpaces.evaluate(TP, el, (x1,x2))
            # Positivity of the polynomials
            @test minimum(TP_eval[:,:,1]) >= 0.0

            # Partition of unity
            @test all(isapprox.(sum(TP_eval[:,:,1], dims=2), 1.0))
        end
    end
end
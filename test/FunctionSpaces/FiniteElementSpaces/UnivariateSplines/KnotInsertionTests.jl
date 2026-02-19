module KnotInsertionTests

using Mantis

using Test

# Piece-wise degree of the basis functions on which the tests are performed.
degrees_for_test = 0:5#20
subdivisions_to_test = 2:8

# Tests for validity of knot insertion algorithm
nq = 30
coeff_factor = 5

child_x = Points.CartesianPoints((range(0, 1, nq + 1),))

for p in degrees_for_test
    nel = p + 2
    parent_regularity = fill(p - 1, nel + 1)
    parent_regularity[1] = parent_regularity[end] = -1
    parent_bspline = FunctionSpaces.BSplineSpace(
        Geometry.CartesianGeometry(LinRange(0.0, 1.0, nel + 1)), p, parent_regularity
    )

    parent_coeffs =
        (rand(FunctionSpaces.get_num_basis(parent_bspline)) .* 2 .- 1) .* coeff_factor

    for nsubdivision in subdivisions_to_test
        parent_x = Points.CartesianPoints((range(0, 1, nq * nsubdivision + 1),))

        twoscale_operator, child_bspline = FunctionSpaces.build_two_scale_operator(
            parent_bspline, nsubdivision
        )

        child_coeffs = FunctionSpaces.get_child_basis_coefficients(
            parent_coeffs, twoscale_operator
        )

        for child_el in 1:size(child_bspline.knot_vector.patch_1d)
            parent_el = FunctionSpaces.get_element_parent(child_el, nsubdivision)
            parent_spline_eval = FunctionSpaces.evaluate(
                parent_bspline, parent_el, parent_x, 0, parent_coeffs
            )
            parent_idx = (child_el - 1) % nsubdivision + 1

            child_spline_eval = FunctionSpaces.evaluate(
                child_bspline, child_el, child_x, 0, child_coeffs
            )
            @test all(
                isapprox.(
                    child_spline_eval[1][1][1] .-
                    parent_spline_eval[1][1][1][(1 + (parent_idx - 1) * nq):(1 + parent_idx * nq)],
                    0.0,
                    atol=1e-13,
                ),
            )
        end
    end
end

end

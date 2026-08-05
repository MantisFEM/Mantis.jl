module DirectSumSpaceTests

using Mantis
import LinearAlgebra
using Test

function basic_tests(space, answers)
    # Type-based getters
    @test FunctionSpaces.get_manifold_dim(space) == answers[1]
    @test FunctionSpaces.get_num_components(space) == answers[2]
    @test FunctionSpaces.get_num_patches(space) == answers[3]

    # Full-space properties
    @test all(FunctionSpaces.get_component_spaces(space) .== answers[4])
    @test FunctionSpaces.get_num_elements_per_patch(space) == answers[5]
    @test FunctionSpaces.get_num_basis(space) == answers[6]
    @test FunctionSpaces.get_num_elements(space) == answers[7]
end

function test_direct_sum_space(breakpoints)
    # 1D, multi-component, single patch ----------------------------------------------------
    num_elements_B1 = length(breakpoints) - 1
    num_elements_B2 = length(breakpoints) - 1
    patch = Geometry.CartesianGeometry(breakpoints)
    regularity1 = vcat(-1, ones(Int, length(breakpoints) - 2), -1)
    regularity2 = vcat(-1, fill(2, length(breakpoints) - 2), -1)
    B1 = FunctionSpaces.BSplineSpace(patch, 2, regularity1)
    B2 = FunctionSpaces.BSplineSpace(patch, 3, regularity2)

    D1 = FunctionSpaces.DirectSumSpace((B1, B2))

    # Verify that a different number of elements in each component space throws an error.
    breakpoints_3 = [0.0, 1.0, 2.0]
    patch_3 = Geometry.CartesianGeometry(breakpoints_3)
    B3 = FunctionSpaces.BSplineSpace(patch_3, 3, [-1, 0, -1])
    @test_throws ArgumentError FunctionSpaces.DirectSumSpace((B1, B3))
    @test_throws ArgumentError FunctionSpaces.DirectSumSpace((B1, B2, B3))

    num_basis_B1 = FunctionSpaces.get_num_basis(B1)
    num_basis_B2 = FunctionSpaces.get_num_basis(B2)
    num_basis_D1 = num_basis_B1 + num_basis_B2
    basic_tests(D1, (1, 2, 1, (B1, B2), (num_elements_B1,), num_basis_D1, num_elements_B1))

    # Verify that the evaluation of the direct sum space is indeed the evaluation of the
    # component spaces per component and zero elsewhere.
    xi = Points.TensorProductPoints(([0.0, 0.5, 1.0],))
    D1_eval, _ = FunctionSpaces.evaluate(D1, num_elements_B1, xi, 1)
    B1_eval, _ = FunctionSpaces.evaluate(B1, num_elements_B1, xi, 1)
    B2_eval, _ = FunctionSpaces.evaluate(B2, num_elements_B1, xi, 1)
    n_nonzero_basis_B1 = FunctionSpaces.get_num_basis(B1, num_elements_B1)
    @test all(
        isapprox(D1_eval[1][1][1][:, 1:n_nonzero_basis_B1], B1_eval[1][1][1]; rtol=1e-14)
    )
    @test all(D1_eval[1][1][1][:, (n_nonzero_basis_B1 + 1):end] .== 0.0)
    @test all(
        isapprox(
            D1_eval[1][1][2][:, (n_nonzero_basis_B1 + 1):end], B2_eval[1][1][1]; rtol=1e-14
        ),
    )
    @test all(D1_eval[1][1][2][:, 1:n_nonzero_basis_B1] .== 0.0)

    # 1D, multi-component, multi-patch -----------------------------------------------------
    GB = FunctionSpaces.GTBSplineSpace((B1, B2), [1, -1])
    D2 = FunctionSpaces.DirectSumSpace((GB, GB))
    num_basis_D2 = 2 * FunctionSpaces.get_num_basis(GB)
    elems_per_patch_D2 = (num_elements_B1, num_elements_B2)
    num_elems_D2 = sum(elems_per_patch_D2)
    basic_tests(D2, (1, 2, 2, (GB, GB), elems_per_patch_D2, num_basis_D2, num_elems_D2))

    # Verify that the evaluation of the direct sum space is indeed the evaluation of the
    # component spaces per component and zero elsewhere.
    D2_eval, _ = FunctionSpaces.evaluate(D2, num_elems_D2, xi, 1)
    GB_eval, _ = FunctionSpaces.evaluate(GB, num_elems_D2, xi, 1)
    n_nonzero_basis_GB = FunctionSpaces.get_num_basis(GB, num_elems_D2)
    @test all(
        isapprox(D2_eval[1][1][1][:, 1:n_nonzero_basis_GB], GB_eval[1][1][1]; rtol=1e-14)
    )
    @test all(D2_eval[1][1][1][:, (n_nonzero_basis_GB + 1):end] .== 0.0)
    @test all(
        isapprox(
            D2_eval[1][1][2][:, (n_nonzero_basis_GB + 1):end], GB_eval[1][1][1]; rtol=1e-14
        ),
    )
    @test all(D2_eval[1][1][2][:, 1:n_nonzero_basis_GB] .== 0.0)

    # 2D, multi-component, single patch ----------------------------------------------------
    TP1 = FunctionSpaces.TensorProductSpace((B1, B1))
    TP2 = FunctionSpaces.TensorProductSpace((B2, B2))
    TP3 = FunctionSpaces.TensorProductSpace((B1, B2))

    D3 = FunctionSpaces.DirectSumSpace((TP1, TP2, TP3))

    num_basis_D3 =
        FunctionSpaces.get_num_basis(TP1) +
        FunctionSpaces.get_num_basis(TP2) +
        FunctionSpaces.get_num_basis(TP3)
    num_elements_TP1 = num_elements_B1^2
    num_elems_D3 = num_elements_TP1
    basic_tests(D3, (2, 3, 1, (TP1, TP2, TP3), (num_elems_D3,), num_basis_D3, num_elems_D3))

    # Verify that the evaluation of the direct sum space is indeed the evaluation of the
    # component spaces per component and zero elsewhere.
    xi = Points.TensorProductPoints(([0.0, 0.5, 1.0], [0.1, 0.7]))
    D3_eval, _ = FunctionSpaces.evaluate(D3, num_elems_D3, xi, 1)
    TP1_eval, _ = FunctionSpaces.evaluate(TP1, num_elems_D3, xi, 1)
    TP2_eval, _ = FunctionSpaces.evaluate(TP2, num_elems_D3, xi, 1)
    TP3_eval, _ = FunctionSpaces.evaluate(TP3, num_elems_D3, xi, 1)
    n_nonzero_basis_TP1 = FunctionSpaces.get_num_basis(TP1, num_elems_D3)
    n_nonzero_basis_TP2 = FunctionSpaces.get_num_basis(TP2, num_elems_D3)
    @test all(
        isapprox(D3_eval[1][1][1][:, 1:n_nonzero_basis_TP1], TP1_eval[1][1][1]; rtol=1e-14)
    )
    @test all(D3_eval[1][1][1][:, (n_nonzero_basis_TP1 + 1):end] .== 0.0)
    @test all(
        D3_eval[1][1][1][:, (n_nonzero_basis_TP1 + n_nonzero_basis_TP2 + 1):end] .== 0.0
    )

    @test all(
        isapprox(
            D3_eval[1][1][2][
                :, (n_nonzero_basis_TP1 + 1):(n_nonzero_basis_TP1 + n_nonzero_basis_TP2)
            ],
            TP2_eval[1][1][1];
            rtol=1e-14,
        ),
    )
    @test all(D3_eval[1][1][2][:, 1:n_nonzero_basis_TP1] .== 0.0)
    @test all(
        D3_eval[1][1][2][:, (n_nonzero_basis_TP1 + n_nonzero_basis_TP2 + 1):end] .== 0.0
    )

    @test all(
        isapprox(
            D3_eval[1][1][3][:, (n_nonzero_basis_TP1 + n_nonzero_basis_TP2 + 1):end],
            TP3_eval[1][1][1];
            rtol=1e-14,
        ),
    )
    @test all(D3_eval[1][1][3][:, 1:n_nonzero_basis_TP1] .== 0.0)
    @test all(
        D3_eval[1][1][3][
            :, (n_nonzero_basis_TP1 + 1):(n_nonzero_basis_TP1 + n_nonzero_basis_TP2)
        ] .== 0.0,
    )

    # 3D, multi-component, single patch ----------------------------------------------------
    TP4 = FunctionSpaces.TensorProductSpace((B1, B1, B1))
    TP5 = FunctionSpaces.TensorProductSpace((B2, B2, B2))
    TP6 = FunctionSpaces.TensorProductSpace((B1, B2, B2))

    D4 = FunctionSpaces.DirectSumSpace((TP4, TP5, TP6))

    num_basis_D4 =
        FunctionSpaces.get_num_basis(TP4) +
        FunctionSpaces.get_num_basis(TP5) +
        FunctionSpaces.get_num_basis(TP6)
    num_elements_TP4 = num_elements_B1^3
    num_elements_TP6 = num_elements_B1 * num_elements_B2 * num_elements_B2
    num_elems_D4 = num_elements_TP4

    # Verify that the evaluation of the direct sum space is indeed the evaluation of the
    # component spaces per component and zero elsewhere.
    xi = Points.TensorProductPoints(([0.0, 0.5, 1.0], [0.1, 0.7], [0.22]))
    D4_eval, _ = FunctionSpaces.evaluate(D4, num_elems_D4, xi, 1)

    TP4_eval, _ = FunctionSpaces.evaluate(TP4, num_elems_D4, xi, 1)
    TP5_eval, _ = FunctionSpaces.evaluate(TP5, num_elems_D4, xi, 1)
    TP6_eval, _ = FunctionSpaces.evaluate(TP6, num_elems_D4, xi, 1)
    n_nonzero_basis_TP4 = FunctionSpaces.get_num_basis(TP4, num_elems_D4)
    n_nonzero_basis_TP5 = FunctionSpaces.get_num_basis(TP5, num_elems_D4)
    @test all(
        isapprox(D4_eval[1][1][1][:, 1:n_nonzero_basis_TP4], TP4_eval[1][1][1]; rtol=1e-14)
    )
    @test all(D4_eval[1][1][1][:, (n_nonzero_basis_TP4 + 1):end] .== 0.0)
    @test all(
        D4_eval[1][1][1][:, (n_nonzero_basis_TP4 + n_nonzero_basis_TP5 + 1):end] .== 0.0
    )

    @test all(
        isapprox(
            D4_eval[1][1][2][
                :, (n_nonzero_basis_TP4 + 1):(n_nonzero_basis_TP4 + n_nonzero_basis_TP5)
            ],
            TP5_eval[1][1][1];
            rtol=1e-14,
        ),
    )
    @test all(D4_eval[1][1][2][:, 1:n_nonzero_basis_TP4] .== 0.0)
    @test all(
        D4_eval[1][1][2][:, (n_nonzero_basis_TP4 + n_nonzero_basis_TP5 + 1):end] .== 0.0
    )

    @test all(
        isapprox(
            D4_eval[1][1][3][:, (n_nonzero_basis_TP4 + n_nonzero_basis_TP5 + 1):end],
            TP6_eval[1][1][1];
            rtol=1e-14,
        ),
    )
    @test all(D4_eval[1][1][3][:, 1:n_nonzero_basis_TP4] .== 0.0)
    @test all(
        D4_eval[1][1][3][
            :, (n_nonzero_basis_TP4 + 1):(n_nonzero_basis_TP4 + n_nonzero_basis_TP5)
        ] .== 0.0,
    )

    # 3D, single-component, single patch ---------------------------------------------------
    D5 = FunctionSpaces.DirectSumSpace((TP6,))

    num_basis_D5 = FunctionSpaces.get_num_basis(TP6)
    num_elems_D5 = num_elements_TP6
    basic_tests(D5, (3, 1, 1, (TP6,), (num_elems_D5,), num_basis_D5, num_elems_D5))

    # Verify that the evaluation of the direct sum space is indeed the evaluation of the
    # component spaces per component and zero elsewhere.
    D5_eval, D5_ind = FunctionSpaces.evaluate(D5, num_elems_D5, xi, 1)
    @test all(isapprox(D5_eval[1][1][1], TP6_eval[1][1][1]; rtol=1e-14))
end

# Loop over three sets of breakpoints to ensure correct behaviour when the number of
# elements reduces to 1 and to test different types of breakpoints.
test_direct_sum_space([0.0, 2.0])
test_direct_sum_space([0.0, 0.25, 0.5, 0.75, 1.0])
test_direct_sum_space(LinRange(0.0, 1.0, 5))
test_direct_sum_space([0.0, 0.2, 0.7, 1.0])

end

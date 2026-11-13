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

"""
Verify that the evaluation of the direct sum space is, per component, the evaluation of the
corresponding component space, and that the basis indices of each component are the (offset)
basis indices of the corresponding component space.
"""
function test_direct_sum_evaluation(space, component_spaces, element_id, xi, nderivatives)
    num_components = length(component_spaces)
    space_eval = FunctionSpaces.evaluate(space, element_id, xi, nderivatives)
    @test length(space_eval[1][1]) == num_components

    offset = 0
    union_indices = Int[]
    for (component_id, component_space) in enumerate(component_spaces)
        component_eval = FunctionSpaces.evaluate(
            component_space, element_id, xi, nderivatives
        )
        component_indices =
            FunctionSpaces.get_basis_indices(component_space, element_id) .+ offset
        @test FunctionSpaces.get_basis_indices(space, element_id, component_id) ==
            component_indices
        @test FunctionSpaces.get_num_basis(space, element_id, component_id) ==
            length(component_indices)
        for der_order in eachindex(space_eval)
            for der_idx in eachindex(space_eval[der_order])
                @test size(space_eval[der_order][der_idx][component_id]) ==
                    (Points.get_num_points(xi), length(component_indices))
                @test isapprox(
                    space_eval[der_order][der_idx][component_id],
                    component_eval[der_order][der_idx][1];
                    rtol=1e-14,
                )
            end
        end
        append!(union_indices, component_indices)
        offset += FunctionSpaces.get_num_basis(component_space)
    end
    @test FunctionSpaces.get_basis_indices(space, element_id) == union_indices
    @test FunctionSpaces.get_num_basis(space, element_id) == length(union_indices)

    # The extraction only consists of identity blocks on the diagonal.
    @test FunctionSpaces.get_block_pattern(space) ==
        ntuple(component_id -> (component_id, component_id), num_components)
    @test all(
        block -> block == LinearAlgebra.I,
        FunctionSpaces.get_extraction_blocks(space, element_id),
    )
    for input_component_id in 1:num_components
        for output_component_id in 1:num_components
            if input_component_id == output_component_id
                @test FunctionSpaces.get_extraction_coefficients(
                    space, element_id, input_component_id, output_component_id
                ) == LinearAlgebra.I
            else
                @test_throws ArgumentError FunctionSpaces.get_extraction_coefficients(
                    space, element_id, input_component_id, output_component_id
                )
            end
        end
    end

    return nothing
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

    xi = Points.TensorProductPoints(([0.0, 0.5, 1.0],))
    for element_id in 1:num_elements_B1
        test_direct_sum_evaluation(D1, (B1, B2), element_id, xi, 1)
    end

    # The input component must be given explicitly for multi-component spaces.
    @test_throws MethodError FunctionSpaces.get_local_basis(D1, 1, xi, 1)

    # 1D, multi-component, multi-patch -----------------------------------------------------
    GB = FunctionSpaces.GTBSplineSpace((B1, B2), [1, -1])
    D2 = FunctionSpaces.DirectSumSpace((GB, GB))
    num_basis_D2 = 2 * FunctionSpaces.get_num_basis(GB)
    elems_per_patch_D2 = (num_elements_B1, num_elements_B2)
    num_elems_D2 = sum(elems_per_patch_D2)
    basic_tests(D2, (1, 2, 2, (GB, GB), elems_per_patch_D2, num_basis_D2, num_elems_D2))

    for element_id in 1:num_elems_D2
        test_direct_sum_evaluation(D2, (GB, GB), element_id, xi, 1)
    end

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

    xi = Points.TensorProductPoints(([0.0, 0.5, 1.0], [0.1, 0.7]))
    for element_id in 1:num_elems_D3
        test_direct_sum_evaluation(D3, (TP1, TP2, TP3), element_id, xi, 1)
    end

    # 3D, multi-component, single patch ----------------------------------------------------
    TP4 = FunctionSpaces.TensorProductSpace((B1, B1, B1))
    TP5 = FunctionSpaces.TensorProductSpace((B2, B2, B2))
    TP6 = FunctionSpaces.TensorProductSpace((B1, B2, B2))

    D4 = FunctionSpaces.DirectSumSpace((TP4, TP5, TP6))

    num_elements_TP4 = num_elements_B1^3
    num_elements_TP6 = num_elements_B1 * num_elements_B2 * num_elements_B2
    num_elems_D4 = num_elements_TP4

    xi = Points.TensorProductPoints(([0.0, 0.5, 1.0], [0.1, 0.7], [0.22]))
    test_direct_sum_evaluation(D4, (TP4, TP5, TP6), num_elems_D4, xi, 1)

    # 3D, single-component, single patch ---------------------------------------------------
    D5 = FunctionSpaces.DirectSumSpace((TP6,))

    num_basis_D5 = FunctionSpaces.get_num_basis(TP6)
    num_elems_D5 = num_elements_TP6
    basic_tests(D5, (3, 1, 1, (TP6,), (num_elems_D5,), num_basis_D5, num_elems_D5))

    test_direct_sum_evaluation(D5, (TP6,), num_elems_D5, xi, 1)
    # Single-component direct sums do not need an explicit component.
    @test FunctionSpaces.get_local_basis(D5, num_elems_D5, xi, 1) ==
        FunctionSpaces.get_local_basis(D5, num_elems_D5, xi, 1, 1)
end

# Loop over three sets of breakpoints to ensure correct behaviour when the number of
# elements reduces to 1 and to test different types of breakpoints.
test_direct_sum_space([0.0, 2.0])
test_direct_sum_space([0.0, 0.25, 0.5, 0.75, 1.0])
test_direct_sum_space(LinRange(0.0, 1.0, 5))
test_direct_sum_space([0.0, 0.2, 0.7, 1.0])

# Example from issue #424: each component only stores its own basis functions.
B = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (1,), (1,), (0,))
D = FunctionSpaces.DirectSumSpace((B, B))
xi = Points.TensorProductPoints((collect(LinRange(0.0, 1.0, 4)),))
D_eval = FunctionSpaces.evaluate(D, 1, xi)
B_expected = [1.0 0.0; 2 / 3 1 / 3; 1 / 3 2 / 3; 0.0 1.0]
@test FunctionSpaces.get_basis_indices(D, 1) == [1, 2, 3, 4]
@test FunctionSpaces.get_basis_indices(D, 1, 1) == [1, 2]
@test FunctionSpaces.get_basis_indices(D, 1, 2) == [3, 4]
@test size(D_eval[1][1][1]) == (4, 2)
@test size(D_eval[1][1][2]) == (4, 2)
@test isapprox(D_eval[1][1][1], B_expected; atol=1e-14)
@test isapprox(D_eval[1][1][2], B_expected; atol=1e-14)

end

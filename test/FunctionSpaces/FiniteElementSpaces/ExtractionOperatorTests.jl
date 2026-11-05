module ExtractionOperatorTests

using Mantis
import LinearAlgebra
using Test

# Local basis of the linear Bernstein polynomials, B = [1-x x], evaluated at `x`.
linear_basis(x) = [1 .- x x]
const x = collect(LinRange(0.0, 1.0, 4))

"""
Computes M_co = ∑ B_ci E_{ci, co} for each output component, by looping over the stored blocks.
"""
function extract(extraction_op, element_id, local_bases)
    num_output_components = FunctionSpaces.get_num_output_components(extraction_op)
    num_points = size(first(local_bases), 1)
    M = [
        zeros(num_points, FunctionSpaces.get_num_basis(extraction_op, element_id, co)) for
        co in 1:num_output_components
    ]
    blocks = FunctionSpaces.get_extraction_blocks(extraction_op, element_id)
    for (block_id, (ci, co)) in enumerate(FunctionSpaces.get_block_pattern(extraction_op))
        LinearAlgebra.mul!(M[co], local_bases[ci], blocks[block_id], true, true)
    end

    return M
end

@testset "Indices" begin
    # Single component.
    indices = FunctionSpaces.Indices([3, 4, 5])
    @test FunctionSpaces.get_num_components(indices) == 1
    @test FunctionSpaces.get_basis_indices(indices) == [3, 4, 5]
    @test FunctionSpaces.get_basis_indices(indices, 1) ===
        FunctionSpaces.get_basis_indices(indices)
    @inferred FunctionSpaces.get_basis_indices(indices)
    @inferred FunctionSpaces.get_basis_indices(indices, 1)

    # Multiple components, with an index shared by both components.
    indices = FunctionSpaces.Indices([1, 2, 3], ([1, 2], [2, 3]))
    @test FunctionSpaces.get_num_components(indices) == 2
    @test FunctionSpaces.get_basis_indices(indices) == [1, 2, 3]
    @test FunctionSpaces.get_basis_indices(indices, 1) == [1, 2]
    @test FunctionSpaces.get_basis_indices(indices, 2) == [2, 3]
    @inferred FunctionSpaces.get_basis_indices(indices, 2)

    # Ranges are allowed.
    indices = FunctionSpaces.Indices(1:4, (1:2, 3:4))
    @test FunctionSpaces.get_basis_indices(indices, 2) == 3:4

    # Invalid indices.
    @test_throws ArgumentError FunctionSpaces.Indices([1.0, 2.0])
    @test_throws ArgumentError FunctionSpaces.Indices([1, 1, 2])
    @test_throws ArgumentError FunctionSpaces.Indices([1, 2], ([1], [3]))
end

@testset "Single component" begin
    coefficients = [rand(3, 3) for _ in 1:2]
    indices = [1:3, 2:4]
    extraction_op = FunctionSpaces.ExtractionOperator(coefficients, indices, 2, 4)
    @test extraction_op isa
        FunctionSpaces.ExtractionOperator{1, 1, 1, Matrix{Float64}, UnitRange{Int}}

    # Type-level information.
    @test FunctionSpaces.get_num_input_components(extraction_op) == 1
    @test FunctionSpaces.get_num_output_components(extraction_op) == 1
    @test FunctionSpaces.get_num_components(extraction_op) == 1
    @test FunctionSpaces.get_num_blocks(extraction_op) == 1
    @test FunctionSpaces.get_block_pattern(extraction_op) == ((1, 1),)
    @test FunctionSpaces.get_EI_types(extraction_op) == (Matrix{Float64}, UnitRange{Int})
    @test FunctionSpaces.get_extraction_type(extraction_op) == Matrix{Float64}
    @test FunctionSpaces.get_index_type(extraction_op) == UnitRange{Int}

    # Extraction coefficients.
    for element_id in 1:2
        @test FunctionSpaces.get_extraction_blocks(extraction_op, element_id) ===
            (coefficients[element_id],)
        @test FunctionSpaces.get_extraction_coefficients(extraction_op, element_id) ===
            coefficients[element_id]
        @test FunctionSpaces.get_extraction_coefficients(extraction_op, element_id, 1) ===
            coefficients[element_id]
        @test FunctionSpaces.get_extraction_coefficients(
            extraction_op, element_id, 1, 1
        ) === coefficients[element_id]
    end
    @test_throws ArgumentError FunctionSpaces.get_extraction_coefficients(
        extraction_op, 1, 1, 2
    )
    @test_throws ArgumentError FunctionSpaces.get_extraction_coefficients(
        extraction_op, 1, 2, 1
    )

    # Basis indices and dimensions.
    @test FunctionSpaces.get_basis_indices(extraction_op, 2) == 2:4
    @test FunctionSpaces.get_basis_indices(extraction_op, 2, 1) == 2:4
    @test FunctionSpaces.get_num_basis(extraction_op) == 4
    @test FunctionSpaces.get_num_basis(extraction_op, 1) == 3
    @test FunctionSpaces.get_num_basis(extraction_op, 1, 1) == 3
    @test FunctionSpaces.get_num_elements(extraction_op) == 2

    # Invalid operators: wrong number of columns, and lengths not matching the number of
    # elements.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator([rand(3, 2)], [1:3], 1, 3)
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        coefficients, indices, 3, 4
    )
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        coefficients, indices[1:1], 2, 4
    )

    # Hot-path getters are type stable.
    @inferred FunctionSpaces.get_extraction_blocks(extraction_op, 1)
    @inferred FunctionSpaces.get_extraction_coefficients(extraction_op, 1)
    @inferred FunctionSpaces.get_extraction_coefficients(extraction_op, 1, 1)
    @inferred FunctionSpaces.get_extraction_coefficients(extraction_op, 1, 1, 1)
    @inferred FunctionSpaces.get_basis_indices(extraction_op, 1)
    @inferred FunctionSpaces.get_basis_indices(extraction_op, 1, 1)
end

@testset "Direct sum (identity blocks)" begin
    extraction_op = FunctionSpaces.ExtractionOperator(
        ((1, 1), (2, 2)),
        [(LinearAlgebra.I, LinearAlgebra.I)],
        [FunctionSpaces.Indices([1, 2, 3, 4], ([1, 2], [3, 4]))],
        1,
        4,
    )
    @test FunctionSpaces.get_num_input_components(extraction_op) == 2
    @test FunctionSpaces.get_num_output_components(extraction_op) == 2
    @test FunctionSpaces.get_num_blocks(extraction_op) == 2
    @test FunctionSpaces.get_basis_indices(extraction_op, 1) == 1:4
    @test FunctionSpaces.get_basis_indices(extraction_op, 1, 1) == [1, 2]
    @test FunctionSpaces.get_basis_indices(extraction_op, 1, 2) == [3, 4]
    @test FunctionSpaces.get_num_basis(extraction_op, 1) == 4
    @test FunctionSpaces.get_num_basis(extraction_op, 1, 2) == 2
    # The off-diagonal blocks are structurally zero, and thus not stored.
    @test_throws ArgumentError FunctionSpaces.get_extraction_coefficients(
        extraction_op, 1, 1, 2
    )
    @test_throws ArgumentError FunctionSpaces.get_extraction_coefficients(
        extraction_op, 1, 2, 1
    )
    @test FunctionSpaces.get_extraction_coefficients(extraction_op, 1, 2, 2) ==
        LinearAlgebra.I
    # There is no single extraction block.
    @test_throws MethodError FunctionSpaces.get_extraction_coefficients(extraction_op, 1)

    M = extract(extraction_op, 1, (linear_basis(x), linear_basis(x)))
    @test M[1] ≈ linear_basis(x)
    @test M[2] ≈ linear_basis(x)

    @inferred FunctionSpaces.get_extraction_coefficients(extraction_op, 1, 2, 2)
    @inferred FunctionSpaces.get_basis_indices(extraction_op, 1, 2)
end

@testset "Overlapping component indices" begin
    # Global basis function 2 has a non-zero first and second component.
    E11 = [1.0 0.0; 0.0 1.0]  # B_1 -> I_1 = [1, 2]
    E22 = [0.5 0.0; 0.5 1.0]  # B_2 -> I_2 = [2, 3]
    indices = [FunctionSpaces.Indices([1, 2, 3], ([1, 2], [2, 3]))]
    extraction_op = FunctionSpaces.ExtractionOperator(
        ((1, 1), (2, 2)), [(E11, E22)], indices, 1, 3
    )
    @test FunctionSpaces.get_num_basis(extraction_op, 1) == 3
    @test FunctionSpaces.get_extraction_coefficients(extraction_op, 1, 1, 1) === E11
    @test FunctionSpaces.get_extraction_coefficients(extraction_op, 1, 2, 2) === E22

    M = extract(extraction_op, 1, (linear_basis(x), linear_basis(x)))
    @test M[1] ≈ linear_basis(x)
    @test M[2] ≈ linear_basis(x) * E22

    # The number of columns must match the number of indices of the output component.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        ((1, 1), (2, 2)),
        [(E11, E22)],
        [FunctionSpaces.Indices([1, 2, 3], ([1, 2], [3]))],
        1,
        3,
    )
end

@testset "Coupled components, different number of input and output components" begin
    # A single input component that contributes to two output components.
    E11 = [1.0 0.0; 0.0 1.0]
    E12 = [2.0; 3.0;;]
    indices = [FunctionSpaces.Indices([1, 2, 3], ([1, 2], [3]))]
    extraction_op = FunctionSpaces.ExtractionOperator(
        ((1, 1), (1, 2)), [(E11, E12)], indices, 1, 3
    )
    @test extraction_op isa
        FunctionSpaces.ExtractionOperator{1, 2, 2, Matrix{Float64}, Vector{Int}}
    @test FunctionSpaces.get_num_input_components(extraction_op) == 1
    @test FunctionSpaces.get_num_output_components(extraction_op) == 2
    @test FunctionSpaces.get_extraction_coefficients(extraction_op, 1, 1, 2) === E12

    M = extract(extraction_op, 1, (linear_basis(x),))
    @test M[1] ≈ linear_basis(x)
    @test M[2] ≈ linear_basis(x) * E12

    # Two input components that contribute to a single output component.
    indices = [FunctionSpaces.Indices([1, 2])]
    extraction_op = FunctionSpaces.ExtractionOperator(
        ((1, 1), (2, 1)), [(E11, 2.0 .* E11)], indices, 1, 2
    )
    @test FunctionSpaces.get_num_input_components(extraction_op) == 2
    @test FunctionSpaces.get_num_output_components(extraction_op) == 1
    M = extract(extraction_op, 1, (linear_basis(x), linear_basis(x)))
    @test M[1] ≈ 3.0 .* linear_basis(x)

    # Blocks of the same input component must have the same number of rows.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        ((1, 1), (1, 2)),
        [(E11, [2.0; 3.0; 4.0;;])],
        [FunctionSpaces.Indices([1, 2, 3], ([1, 2], [3]))],
        1,
        3,
    )
end

@testset "Block pattern validation" begin
    block = [1.0;;]
    indices_1 = [FunctionSpaces.Indices([1])]
    indices_2 = [FunctionSpaces.Indices([1, 2], ([1], [2]))]
    # Empty pattern.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator((), [()], indices_1, 1, 1)
    # Non-positive input component.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        ((0, 1),), [(block,)], indices_1, 1, 1
    )
    # Output component out of range.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        ((1, 2),), [(block,)], indices_1, 1, 1
    )
    # Duplicate blocks.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        ((1, 1), (1, 1)), [(block, block)], indices_1, 1, 1
    )
    # Input component 1 is not part of any block.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        ((2, 1), (2, 2)), [(block, block)], indices_2, 1, 2
    )
    # Output component 2 is not part of any block.
    @test_throws ArgumentError FunctionSpaces.ExtractionOperator(
        ((1, 1),), [(block,)], indices_2, 1, 2
    )

    # Block lookup in a pattern.
    @test FunctionSpaces._get_block_id(((1, 1), (2, 1), (2, 2)), 2, 1) == 2
    @test_throws ArgumentError FunctionSpaces._get_block_id(((1, 1), (2, 2)), 1, 2)
end

end

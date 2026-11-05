module PolarSplineTests

using Mantis
import LinearAlgebra
using Test

# 1D spaces in radial and poloidal directions
num_elements_p = 5
num_elements_r = 2
deg_p = 2
deg_r = 3
regularity_p = deg_p - 1
regularity_r = deg_r - 1
n_p = num_elements_p * (deg_p + 1) - num_elements_p * (regularity_p + 1)
n_r = num_elements_r * (deg_r + 1) - (num_elements_r - 1) * (regularity_r + 1)

#################################################################
# PolarSplineSpace
#################################################################

# build scalar polar spline space

geometry, geom_coeffs_tp = FunctionSpaces.create_polar_geometry_data(
    (num_elements_p, num_elements_r), (deg_p, deg_r), (regularity_p, regularity_r)
)
P_scalar = FunctionSpaces.create_scalar_polar_spline_space(
    (num_elements_p, num_elements_r),
    (deg_p, deg_r),
    (regularity_p, regularity_r),
    geometry;
    geom_coeffs_tp,
)

@test FunctionSpaces.get_num_basis(P_scalar) == n_p * (n_r - 2) + 3
@test FunctionSpaces.get_num_elements(P_scalar) == num_elements_p * num_elements_r
@test FunctionSpaces.get_num_elements_per_patch(P_scalar)[1] ==
    num_elements_p * num_elements_r

# evaluate basis functions
xi = Points.TensorProductPoints(([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.33, 0.66, 1.0]))
@test FunctionSpaces.get_block_pattern(P_scalar) == ((1, 1),)
for element_id in 1:FunctionSpaces.get_num_elements(P_scalar)
    ex_coeffs = FunctionSpaces.get_extraction_coefficients(P_scalar, element_id)
    @test ex_coeffs ==
        FunctionSpaces.get_extraction_coefficients(P_scalar, element_id, 1, 1)
    @test all(isapprox.(sum(ex_coeffs; dims=2) .- 1.0, 0.0, atol=1e-14))

    evaluations = FunctionSpaces.evaluate(P_scalar, element_id, xi)
    basis_indices = FunctionSpaces.get_basis_indices(P_scalar, element_id)
    @test FunctionSpaces.get_basis_indices(P_scalar, element_id, 1) == basis_indices
    @test size(evaluations[1][1][1]) == (Points.get_num_points(xi), length(basis_indices))
    @test all(evaluations[1][1][1] .>= 0.0)
    @test all(isapprox.(sum(evaluations[1][1][1]; dims=2) .- 1.0, 0.0, atol=1e-14))
end

#################################################################
# VectorPolarSplineSpace
#################################################################

# build vector polar spline space
P_vector = FunctionSpaces.create_vector_polar_spline_space(
    (num_elements_p, num_elements_r),
    (deg_p, deg_r),
    (regularity_p, regularity_r),
    geometry;
    geom_coeffs_tp,
)

@test FunctionSpaces.get_num_basis(P_vector) == 2 * n_p * (n_r - 2) + 2
@test FunctionSpaces.get_num_elements(P_vector) == num_elements_p * num_elements_r
@test FunctionSpaces.get_num_elements_per_patch(P_vector)[1] ==
    num_elements_p * num_elements_r

# evaluate basis functions
xi = Points.TensorProductPoints(([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.33, 0.66, 1.0]))
# Each tensor-product space only contributes to its own component.
@test FunctionSpaces.get_block_pattern(P_vector) == ((1, 1), (2, 2))
# Number of elements in the first ring around the pole.
num_pole_elements = num_elements_p
for element_id in 1:FunctionSpaces.get_num_elements(P_vector)
    @test_throws ArgumentError FunctionSpaces.get_extraction_coefficients(
        P_vector, element_id, 1, 2
    )
    @test_throws ArgumentError FunctionSpaces.get_extraction_coefficients(
        P_vector, element_id, 2, 1
    )

    evaluations = FunctionSpaces.evaluate(P_vector, element_id, xi, 1)
    basis_indices = FunctionSpaces.get_basis_indices(P_vector, element_id)
    @test allunique(basis_indices)
    component_basis_indices = ntuple(
        component_id ->
            FunctionSpaces.get_basis_indices(P_vector, element_id, component_id),
        2,
    )
    # The basis functions of the space on this element are the union of those of each
    # component.
    @test issetequal(basis_indices, union(component_basis_indices...))
    for component_id in 1:2
        ex_coeffs = FunctionSpaces.get_extraction_coefficients(
            P_vector, element_id, component_id, component_id
        )
        @test size(ex_coeffs, 2) == length(component_basis_indices[component_id])
        @test issubset(component_basis_indices[component_id], basis_indices)
        for der_order in eachindex(evaluations)
            for der_idx in eachindex(evaluations[der_order])
                @test size(evaluations[der_order][der_idx][component_id]) == (
                    Points.get_num_points(xi), length(component_basis_indices[component_id])
                )
            end
        end
        # Only the basis functions with a non-zero component are stored.
        @test all(col -> !all(iszero, col), eachcol(evaluations[1][1][component_id]))
    end
    # Near the pole, (some) basis functions have both components non-zero.
    if element_id <= num_pole_elements
        @test !isempty(intersect(component_basis_indices...))
    end
end

end

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
for element_id in 1:FunctionSpaces.get_num_elements(P_scalar)
    ex_coeffs, _ = FunctionSpaces.get_extraction(P_scalar, element_id, 1)
    @test all(isapprox.(sum(ex_coeffs; dims=2) .- 1.0, 0.0, atol=1e-14))

    evaluations, basis_indices = FunctionSpaces.evaluate(P_scalar, element_id, xi)
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
for element_id in 1:1#FunctionSpaces.get_num_elements(P_vector)
    ex_coeffs, basis_indices = FunctionSpaces.get_extraction(P_vector, element_id, 1)
    ex_coeffs, basis_indices = FunctionSpaces.get_extraction(P_vector, element_id, 2)
    evaluations, basis_indices = FunctionSpaces.evaluate(P_vector, element_id, xi)
end

end

module TensorProductGeometryTests

using Mantis

# Refer to the following file for method and variable definitions
include("GeometryTestsHelpers.jl")

import ReadVTK
using Test

# Test Square Tensor Product Geometry -----------------------------------------
# Generate a tensor product geometry by combining two lines

# Line geometries
line_1_geometry = Geometry.create_cartesian_box((0.0,), (1.0,), (10,))
line_2_geometry = Geometry.create_cartesian_box((2.0,), (1.0,), (10,))

# Tensor product geometry
tensor_prod_geometry = Geometry.TensorProductGeometry((line_1_geometry, line_2_geometry))

# Set file name and path
file_name = "tensor_product_geometry.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
# Generate the vtk file
Plot.plot(
    tensor_prod_geometry;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)

# Read the cell data from the reference file
reference_points, reference_cells = get_point_cell_data(reference_directory_tree, file_name)
# Read the cell data from the output file
output_points, output_cells = get_point_cell_data(output_file_path)
# Check if cell data is identical
@test all(isapprox.(reference_points, output_points; rtol=rtol))
@test all(isequal.(reference_cells, output_cells))
# -----------------------------------------------------------------------------

# Test Cylinder Tensor Product Geometry ---------------------------------------
deg = 2
nθ_elements = 4
Wt = 2.0 * pi / nθ_elements
b = FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = collect(LinRange(0.0, nθ_elements, nθ_elements + 1))
patch = Mesh.Patch1D(breakpoints)
B = FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])
GB = FunctionSpaces.GTBSplineSpace((B,), [1])

# control points for geometry
# radius of cylinder is 1.0
geom_coeffs_circle = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
cylinder_circle_geometry = Geometry.FEGeometry(GB, geom_coeffs_circle)
dx_cylinder_line = 0.1
nz_elements = 10
cylinder_line_geometry = Geometry.create_cartesian_box((0.0,), (1.0,), (nz_elements,))

# Tensor product geometry
cylinder_tensor_prod_geometry = Geometry.TensorProductGeometry((
    cylinder_circle_geometry, cylinder_line_geometry
))

# Set file name and path
file_name = "tensor_product_cylinder_geometry.vtu"
output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
# Generate the vtk file
Plot.plot(
    cylinder_tensor_prod_geometry;
    vtk_filename=output_file_path[1:(end - 4)], #remove the file extension
    n_subcells=1,
    degree=4,
    ascii=false,
    compress=false,
)
# Read the point and cell data from the reference file
reference_points, reference_cells = get_point_cell_data(reference_directory_tree, file_name)
# Read the point and cell data from the output file
output_points, output_cells = get_point_cell_data(output_file_path)
# Check if point and cell data is identical
@test all(isapprox.(reference_points, output_points; atol=atol))
@test all(isequal.(reference_cells, output_cells))

# Test Jacobian with single point evaluation
# We check the Jacobian
#    J^{k}_{ij} = \partial{\Phi^{i}(\boldsymbol{x}_{k})}{\partial x^{0}_{j}}
# at four points k at different z levels

for element_row_idx in 1:nz_elements
    # Compute Jacobian at x_{1} = [1.0, 0.0, z]
    # This corresponds to the point with local coordinates [0.0, 0.0] on the first element of
    # row element_row_idx
    ξ = Points.CartesianPoints(([0.0], [0.0]))
    J_cylinder_reference = [[
        0.0 0.0
        0.5*π 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 1, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))

    # Compute Jacobian at x_{1} = [0.0, 1.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the first element
    # of row element_row_idx
    ξ = Points.CartesianPoints(([1.0], [0.0]))
    J_cylinder_reference = [[
        -0.5*π 0.0
        0.0 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 1, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))

    # Compute Jacobian at x_{1} = [-1.0, 0.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the second element
    # of row element_row_idx
    ξ = Points.CartesianPoints(([1.0], [0.0]))
    J_cylinder_reference = [[
        0.0 0.0
        -0.5*π 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 2, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))

    # Compute Jacobian at x_{1} = [0.0, -1.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the third element
    # of row element_row_idx
    ξ = Points.CartesianPoints(([1.0], [0.0]))
    J_cylinder_reference = [[
        0.5*π 0.0
        0.0 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 3, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))

    # Compute Jacobian again at x_{1} = [1.0, 0.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the fourth element
    # of row element_row_idx
    ξ = Points.CartesianPoints(([1.0], [0.0]))
    J_cylinder_reference = [[
        0.0 0.0
        0.5*π 0.0
        0.0 dx_cylinder_line
    ]]
    J_cylinder = Geometry.jacobian(
        cylinder_tensor_prod_geometry, (element_row_idx - 1) * nθ_elements + 4, ξ
    )
    @test all(isapprox.(J_cylinder, J_cylinder_reference; atol=atol))
end

# -----------------------------------------------------------------------------

end

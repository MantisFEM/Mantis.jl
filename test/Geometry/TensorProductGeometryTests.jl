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

# Constructor, property, and getters and setters tests -------------------------------------
function basic_tests(geometry, answers)
    @test all(Geometry.get_constituent_num_elements(geometry) .== answers[1])

    @test Geometry.get_num_patches(geometry) == answers[2]
    @test Geometry.get_num_elements(geometry) == answers[3]
    @test Geometry.get_manifold_dim(geometry) == answers[4]
    @test Geometry.get_image_dim(geometry) == answers[5]
    @test Geometry.get_num_elements_per_patch(geometry) == answers[6]
    @test Geometry.get_num_elements(geometry, 1) == answers[7]
    @test Geometry.get_element_lengths(geometry, 1) == answers[8]
    @test all(isapprox.(Geometry.get_element_measure(geometry, 1), answers[9], rtol=1e-14))

    patch_id, local_element_id = Geometry.get_patch_and_local_element_id(
        geometry, answers[11]
    )
    @test (patch_id, local_element_id) == answers[10]
    @test Geometry.get_global_element_id(geometry, patch_id, local_element_id) ==
        answers[11]

    @test all(
        all.([
            isapprox.(
                Geometry.get_element_vertices(geometry, 1)[i], answers[12][i], rtol=1e-14
            ) for i in eachindex(answers[12])
        ]),
    )

    return nothing
end

# Reduction test, single-patch, single element, single geometry, 1D.
cg1 = Geometry.CartesianGeometry(([-1, 1],))
tpgeometry1 = Geometry.TensorProductGeometry((cg1,))
answers_1 = ((1,), 1, 1, 1, 1, (1,), 1, (2.0,), 2.0, (1, 1), 1, ((-1, 1),))
basic_tests(tpgeometry1, answers_1)

# All cartesian. 3D: 2D (2 patches) tensored with 1D (3 patches).
cg1d = Geometry.CartesianGeometry((
    (LinRange(0.0, 1.0, 2),), (LinRange(1.0, 2.0, 3),), (LinRange(2.0, 3.0, 4),)
))
cg2d = Geometry.CartesianGeometry((
    (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 5)), # First patch
    (LinRange(1.0, 2.0, 6), LinRange(0.0, 1.0, 7)), # Second patch
))
tpgeometry2 = Geometry.TensorProductGeometry((cg2d, cg1d))
answers_2 = (
    (42, 6),
    6,
    252,
    3,
    3,
    (12, 30, 24, 60, 36, 90),
    12,
    (1.0 / 3, 0.25, 1.0),
    1.0 / 12.0,
    (5, 36),
    162,
    ((0.0, 1.0 / 3.0), (0.0, 0.25), (0.0, 1.0)),
)
basic_tests(tpgeometry2, answers_2)

end

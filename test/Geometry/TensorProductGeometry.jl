
import Mantis

import ReadVTK
using Printf
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Geometry")
output_data_folder = joinpath(data_folder, "output", "Geometry")


# Test Square Tensor Product Geometry -----------------------------------------
# Generate a tensor product geometry by combining two lines

# Line 1
breakpoints_1 = (Vector(0.0:0.1:1.0),)
line_1_geo = Mantis.Geometry.CartesianGeometry(breakpoints_1)

# Line 2
breakpoints_2 = (Vector(2.0:0.1:3.0),)
line_2_geo = Mantis.Geometry.CartesianGeometry(breakpoints_2)

# Tensor product geometry 
tensor_prod_geo = Mantis.Geometry.TensorProductGeometry(line_1_geo, line_2_geo)

# Generate the plot
output_filename = "tensor_product_geometry.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(tensor_prod_geo; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Test geometry 
# Read the cell data from the reference file
reference_file = joinpath(input_data_folder, output_filename)
vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# Read the cell data from the output file
vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------


# Test Cylinder Tensor Product Geometry ---------------------------------------
deg = 2
nθ_elements = 4
Wt = pi/2
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, nθ_elements)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])

# control points for geometry
# radius of cylinder is 1.0
geom_coeffs_circle =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]

cylinder_circle_geo = Mantis.Geometry.FEMGeometry(GB, geom_coeffs_circle)

dx_cylinder_line = 0.1
nz_elements = 10
breakpoints_cylinder_line = (Vector(LinRange(0.0, 1.0, nz_elements + 1)),)
cylinder_line_geo = Mantis.Geometry.CartesianGeometry(breakpoints_cylinder_line)

# Tensor product geometry 
cylinder_tensor_prod_geo = Mantis.Geometry.TensorProductGeometry(cylinder_circle_geo, cylinder_line_geo)
# Generate the plot
output_filename = "tensor_product_cylinder_geometry.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(cylinder_tensor_prod_geo; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Test geometry 
# Read the cell data from the reference file
reference_file = joinpath(input_data_folder, output_filename)
vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# Read the cell data from the output file
vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells

# Test evaluation with NTuple input and automatic tensor product 



# Test Jacobian with single point evaluation
# We check the Jacobian 
#    J^{k}_{ij} = \partial{\Phi^{i}(\boldsymbol{x}_{k})}{\partial x^{0}_{j}}
# at four points k at different z levels

for element_row_idx in 1:nz_elements
    # Compute Jacobian at x_{1} = [1.0, 0.0, z]
    # This corresponds to the point with local coordinates [0.0, 0.0] on the first element of row element_row_idx
    ξ = [0.0, 0.0]
    J_cylinder_reference = [0.0 0.5*π 0.0;;; 0.0 0.0 dx_cylinder_line] 
    J_cylinder = Mantis.Geometry.jacobian(cylinder_tensor_prod_geo, (element_row_idx - 1)*nθ_elements + 1, ξ)
    @test maximum(abs.(J_cylinder - J_cylinder_reference)) ≈ 0.0 atol = 1e-14

    # Compute Jacobian at x_{1} = [0.0, 1.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the first element of row element_row_idx
    ξ = [1.0, 0.0]
    J_cylinder_reference = [-0.5*π 0.0 0.0;;; 0.0 0.0 dx_cylinder_line] 
    J_cylinder = Mantis.Geometry.jacobian(cylinder_tensor_prod_geo, (element_row_idx - 1)*nθ_elements + 1, ξ)
    @test maximum(abs.(J_cylinder - J_cylinder_reference)) ≈ 0.0 atol = 1e-14

    # Compute Jacobian at x_{1} = [-1.0, 0.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the second element of row element_row_idx
    ξ = [1.0, 0.0]
    J_cylinder_reference = [0.0 -0.5*π 0.0;;; 0.0 0.0 dx_cylinder_line] 
    J_cylinder = Mantis.Geometry.jacobian(cylinder_tensor_prod_geo, (element_row_idx - 1)*nθ_elements + 2, ξ)
    @test maximum(abs.(J_cylinder - J_cylinder_reference)) ≈ 0.0 atol = 1e-14

    # Compute Jacobian at x_{1} = [0.0, -1.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the third element of row element_row_idx
    ξ = [1.0, 0.0]
    J_cylinder_reference = [0.5*π 0.0 0.0;;; 0.0 0.0 dx_cylinder_line] 
    J_cylinder = Mantis.Geometry.jacobian(cylinder_tensor_prod_geo, (element_row_idx - 1)*nθ_elements + 3, ξ)
    @test maximum(abs.(J_cylinder - J_cylinder_reference)) ≈ 0.0 atol = 1e-14

    # Compute Jacobian again at x_{1} = [1.0, 0.0, 0.0]
    # This corresponds to the point with local coordinates [1.0, 0.0] on the fourth element of row element_row_idx
    ξ = [1.0, 0.0]
    J_cylinder_reference = [0.0 0.5*π 0.0;;; 0.0 0.0 dx_cylinder_line] 
    J_cylinder = Mantis.Geometry.jacobian(cylinder_tensor_prod_geo, (element_row_idx - 1)*nθ_elements + 4, ξ)
    @test maximum(abs.(J_cylinder - J_cylinder_reference)) ≈ 0.0 atol = 1e-14
end

# -----------------------------------------------------------------------------


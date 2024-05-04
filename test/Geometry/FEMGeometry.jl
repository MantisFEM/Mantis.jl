
import Mantis

import ReadVTK
using Printf
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Geometry")
output_data_folder = joinpath(data_folder, "output", "Geometry")

# Test FEMGeometry (Annulus) --------------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP = Mantis.FunctionSpaces.TensorProductSpace(GB, b1, Dict())
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0
               geom_coeffs_0.*r1]
geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
# Generate the plot
output_filename = "fem_geometry_annulus_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

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

# # Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry - LagrangexBernstein (Square w/ hole) ----------------------
deg = 1
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.LobattoLegendre(deg))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [0,0,0,0])
b1 = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP = Mantis.FunctionSpaces.TensorProductSpace(GB, b1, Dict())
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0
               geom_coeffs_0.*r1]
geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
# Generate the plot
output_filename = "fem_geometry_lagrange_square_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1, ascii = false, compress = false)

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

# # Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry (Spiral) ---------------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, -1])
# control points for geometry
geom_coeffs =   [0.0 -1.0 0.0
1.0  -1.0 0.25
1.0   1.0 0.5
-1.0   1.0 0.75
-1.0  -1.0 1.0
0.0  -1.0 1.25]
spiral_geom = Mantis.Geometry.FEMGeometry(GB, geom_coeffs)

# Generate the plot
output_filename = "fem_geometry_spiral_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(spiral_geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

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

# # Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry (wavy surface) ---------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt))
B = ntuple( i -> b, 4)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.CanonicalFiniteElementSpace(Mantis.FunctionSpaces.Bernstein(1))
TP = Mantis.FunctionSpaces.TensorProductSpace(GB, b1, Dict())
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
    1.0   1.0
    -1.0   1.0
    -1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0 -[+1.0, -1.0, +1.0, -1.0]
               geom_coeffs_0.*r1 [+1.0, -1.0, +1.0, -1.0]]
wavy_surface_geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)

# Generate the plot
output_filename = "fem_geometry_wavy_surface_test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(wavy_surface_geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

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

# # Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------
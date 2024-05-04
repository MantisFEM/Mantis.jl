
import Mantis

import ReadVTK
using Printf
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Geometry")
output_data_folder = joinpath(data_folder, "output", "Geometry")

# Test CartesianGeometry ------------------------------------------------------
for nx = 1:3
    for ny = 1:3
        breakpoints = (collect(LinRange(0.0, 1.0, nx+1)), collect(LinRange(0.0,2.0,ny+1)))
        geom = Mantis.Geometry.CartesianGeometry(breakpoints)
        
        # Generate the plot
        output_filename = @sprintf "cartesian_test_nx_%d_ny_%d.vtu" nx ny
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
    end
end
# -----------------------------------------------------------------------------
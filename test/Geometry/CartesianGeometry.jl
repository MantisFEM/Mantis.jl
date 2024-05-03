
import Mantis

import Mmap
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
        Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1)

        # Test geometry 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
    end
end
# -----------------------------------------------------------------------------
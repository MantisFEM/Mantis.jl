module PlotTests

import Mantis

import Mmap
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Plot")
output_data_folder = joinpath(data_folder, "output")

# Test AnalGeometry

# Rectangle based geometries input parameters
n_elements = (2, 2)
xy_start = [0.0, 0.0]
xy_end = [1.0, 1.0]

# Rectangle geometry
rectangle = Mantis.Geometry.Rectangle(n_elements, xy_start, xy_end)

# Generate the plot 
output_file = joinpath(output_data_folder, "rectangle_test.vtu")
Mantis.Plot.plot(rectangle; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1)

# Test plotting 
input_file = joinpath(input_data_folder, "rectangle_test.vtu")
@test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))

# Mapped rectangle geometry
function mapping(x::Vector{Float64})
    return [(x[1] + 0.2)*cos(x[2]), (x[1] + 0.2)*sin(x[2])]
end

mapped_rectangle = Mantis.Geometry.MappedRectangle(n_elements, xy_start, xy_end, mapping)

# Generate the plot 
output_file = joinpath(output_data_folder, "mapped_rectangle_test.vtu")
Mantis.Plot.plot(mapped_rectangle; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1)

# Test plotting 
input_file = joinpath(input_data_folder, "mapped_rectangle_test.vtu")
@test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))


end
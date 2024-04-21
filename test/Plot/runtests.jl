module PlotTests

import Mantis

import Mmap
using Printf
using Test

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
input_data_folder = joinpath(data_folder, "reference", "Plot")
output_data_folder = joinpath(data_folder, "output")

# Test AnalGeometry -----------------------------------------------------------
n_subcells_to_test = 1:4
degrees_to_test = 1:4
for n_subcells in n_subcells_to_test 
    for degree in degrees_to_test
        # Rectangle based geometries input parameters
        n_elements = (2, 2)
        xy_start = [0.0, 0.0]
        xy_end = [1.0, 1.0]

        # Rectangle geometry
        rectangle = Mantis.Geometry.Rectangle(n_elements, xy_start, xy_end)

        # Generate the plot
        output_filename = @sprintf "rectangle_test_n_%d_p_%d.vtu" n_subcells degree
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(rectangle; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree)

        # Test plotting 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))

        # Mapped custom rectangle geometry
        function mapping(x::Vector{Float64})
            return [(x[1] + 0.2)*cos(x[2]), (x[1] + 0.2)*sin(x[2])]
        end

        function dmapping(x::Vector{Float64})
            return [cos(x[2]) -(x[1] + 0.2)*sin(x[2]); sin(x[2]) (x[1] + 0.2)*cos(x[2])]
        end

        mapped_rectangle = Mantis.Geometry.MappedRectangle(n_elements, xy_start, xy_end, mapping, dmapping)

        # Generate the plot 
        output_filename = @sprintf "mapped_rectangle_test_n_%d_p_%d.vtu" n_subcells degree
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(mapped_rectangle; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree)

        # Test plotting 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
        # -----------------------------------------------------------------------------


        # Test Mapped geometry --------------------------------------------------------
        # This results in the same geometry as the custom mapped_rectangle, the difference is that is uses composition
        # via the MappedGeometry struct: we provide the straight geometry rectangle and then a mapping and it generates
        # the resulting geometry, which is a composition of the rectangle geometry and the mapping 
        dimension = (2, 2)

        curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)

        mapped_geometry = Mantis.Geometry.MappedGeometry(rectangle, curved_mapping)

        # Generate the plot 
        output_filename = @sprintf "mapped_geometry_test_n_%d_p_%d.vtu" n_subcells degree
        output_file = joinpath(output_data_folder, output_filename)
        Mantis.Plot.plot(mapped_geometry; vtk_filename = output_file[1:end-4], n_subcells = n_subcells, degree = degree)

        # Test plotting 
        input_file = joinpath(input_data_folder, output_filename)
        @test Mmap.mmap(open(input_file)) == Mmap.mmap(open(output_file))
    end
end
# -----------------------------------------------------------------------------

end
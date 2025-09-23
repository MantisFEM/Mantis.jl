module MappedGeometryTests

import Mantis

import ReadVTK
using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")

# Test MappedCartesianGeometry ------------------------------------------------
for nx in 1:3
    for ny in 1:3
        breakpoints = (
            collect(LinRange(0.0, 1.0, nx + 1)), collect(LinRange(0.0, 2.0, ny + 1))
        )
        geom = Mantis.Geometry.CartesianGeometry(breakpoints)

        # Define the mapping ϕ of the geometry and its derivative.
        # ϕ(x,y) = [(x + 0.2)*cos(y), (x + 0.2)*sin(y)\
        function mapping(x::AbstractVector)
            return [(x[1] + 0.2) * cos(x[2]), (x[1] + 0.2) * sin(x[2])]
        end
        function dmapping(x::AbstractVector)
            return [cos(x[2]) -(x[1] + 0.2)*sin(x[2]); sin(x[2]) (x[1] + 0.2)*cos(x[2])]
        end

        dimension = (2, 2)
        curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
        mapped_geometry = Mantis.Geometry.MappedGeometry(geom, curved_mapping)

        # Generate the plot
        file_name = "mapped_cartesian_test_nx_$(nx)_ny_$(ny).vtu"
        output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
        Mantis.Plot.plot(
            mapped_geometry;
            vtk_filename=output_file_path[1:(end - 4)],
            n_subcells=1,
            degree=3,
            ascii=false,
            compress=false,
        )

        # Test geometry
        # Read the cell data from the reference file
        reference_points, reference_cells = get_point_cell_data(
            reference_directory_tree, file_name
        )
        # Read the cell data from the output file
        output_points, output_cells = get_point_cell_data(output_file_path)
        # Check if cell data is identical
        @test all(isapprox.(reference_points, output_points; rtol=rtol))
        @test all(isequal.(reference_cells, output_cells))
    end
end
# -----------------------------------------------------------------------------
end

module CartesianGeometryTests

using Mantis

import ReadVTK
using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")

# Constructor, property, and getters and setters tests -------------------------------------
# Vector{Float64} input.
geometryVF = Geometry.CartesianGeometry((
    [0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [-0.75, 0.0, 0.25, 0.75]
))
@test Geometry.get_breakpoints(geometryVF) ==
    ([0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [-0.75, 0.0, 0.25, 0.75])
@test Geometry.get_constituent_num_elements(geometryVF) == ((2, 2, 3),)

@test Geometry.get_num_elements(geometryVF) == 12
@test Geometry.get_manifold_dim(geometryVF) == 3
@test Geometry.get_image_dim(geometryVF) == 3

# LinRange input.
geometryLR = Geometry.CartesianGeometry((LinRange(0.5, 2.5, 5), LinRange(-0.75, 0.75, 3)))
@test Geometry.get_breakpoints(geometryLR) == ([0.5, 1.0, 1.5, 2.0, 2.5], [-0.75, 0.0, 0.75])
@test Geometry.get_constituent_num_elements(geometryLR) == ((4, 2),)

@test Geometry.get_num_elements(geometryLR) == 8
@test Geometry.get_manifold_dim(geometryLR) == 2
@test Geometry.get_image_dim(geometryLR) == 2

# Multi-patch input.
geometryMP = Geometry.CartesianGeometry((
    (LinRange(0.5, 2.5, 5), LinRange(-0.75, 0.75, 3)),
    (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 6)),
))
@test Geometry.get_breakpoints(geometryMP, 1) == (([0.5, 1.0, 1.5, 2.0, 2.5], [-0.75, 0.0, 0.75]))
@test Geometry.get_breakpoints(geometryMP, 2) == (([0.0, 1/3, 2/3, 1.0], [0.0, 1/5, 2/5, 3/5, 4/5, 1.0]))
@test Geometry.get_constituent_num_elements(geometryMP) == ((4, 2), (3, 5))

@test Geometry.get_num_elements(geometryMP) == 23
@test Geometry.get_manifold_dim(geometryMP) == 2
@test Geometry.get_image_dim(geometryMP) == 2

for i in 1:Geometry.get_num_elements(geometryMP)
    jac = Geometry.jacobian(geometryMP, i, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0])))
    if i <= 8
        for p in axes(jac, 1)
            @test all(isapprox.(jac[p, :, :], [0.5 0.0; 0.0 0.75], rtol=1e-14))
        end
    else
        for p in axes(jac, 1)
            @test all(isapprox.(jac[p, :, :], [1/3 0.0; 0.0 1/5], rtol=1e-14))
        end
    end
end

# Test 2D CartesianGeometry ----------------------------------------------------------------
for nx in 1:3
    for ny in 1:3
        geometry = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 2.0), (nx, ny))

        # Set file name and path
        file_name = "cartesian_test_nx_$(nx)_ny_$(ny).vtu"
        output_file_path = Mantis.GeneralHelpers.export_path(
            output_directory_tree, file_name
        )
        # Generate the vtk file
        Plot.plot(
            geometry;
            vtk_filename=output_file_path[1:(end - 4)],  # Remove the file extension.
            n_subcells=1,
            degree=1,
            ascii=false,
            compress=false,
        )

        # Read the cell data from the reference file.
        reference_points, reference_cells = get_point_cell_data(
            reference_directory_tree, file_name
        )
        # Read the cell data from the output file.
        output_points, output_cells = get_point_cell_data(output_file_path)

        # Check if cell data is point-wise identical.
        @test all(isapprox.(reference_points, output_points; rtol=rtol))
        @test all(isequal.(reference_cells, output_cells))
    end
end

end

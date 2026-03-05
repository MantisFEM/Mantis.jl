# Helper file which defines a few test functions for geometry tests.

# Tolerance for comparing floating point numbers.
rtol = 10 * eps(Float64)
atol = 1e-14

# Compute base directories for data input and output
reference_directory_tree = [
    dirname(dirname(pathof(Mantis))), "test", "data", "reference", "Geometry"
]
output_directory_tree = [
    dirname(dirname(pathof(Mantis))), "test", "data", "output", "Geometry"
]

function get_point_cell_data(directory_tree::Vector{String}, file_name::String)
    # Read the cell data from the reference file
    file_path = Mantis.GeneralHelpers.export_path(directory_tree, file_name)
    vtk_file = ReadVTK.VTKFile(file_path)

    point_data = ReadVTK.get_data(ReadVTK.get_data_section(vtk_file, "Points")["Points"])
    cell_data = ReadVTK.get_data(
        ReadVTK.get_data_section(vtk_file, "Cells")["connectivity"]
    )

    return point_data, cell_data
end

function get_point_cell_data(file_path::String)
    # Read the cell data from the reference file
    vtk_file = ReadVTK.VTKFile(ReadVTK.get_example_file(file_path))

    point_data = ReadVTK.get_data(ReadVTK.get_data_section(vtk_file, "Points")["Points"])
    cell_data = ReadVTK.get_data(
        ReadVTK.get_data_section(vtk_file, "Cells")["connectivity"]
    )

    return point_data, cell_data
end

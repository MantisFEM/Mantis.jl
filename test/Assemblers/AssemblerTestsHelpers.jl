
# Tolerance for comparing floating point numbers.
rtol = 10 * eps(Float64)
atol = 1e-14

# Compute base directories for data input and output
const reference_directory_tree = [
    dirname(dirname(pathof(Mantis))), "test", "data", "reference", "Assemblers"
]
const output_directory_tree = [
    dirname(dirname(pathof(Mantis))), "test", "data", "output", "Assemblers"
]

function write_data(sub_dir::String, file_name::String, data)
    # Write the solution to an output file
    file_path = Mantis.GeneralHelpers.export_path(
        [output_directory_tree; sub_dir], file_name
    )

    return DelimitedFiles.writedlm(file_path, data, ',')
end

function read_data(sub_dir::String, file_name::String)
    # Read the solution from the reference file
    file_path = Mantis.GeneralHelpers.export_path(
        [reference_directory_tree; sub_dir], file_name
    )

    return DelimitedFiles.readdlm(file_path, ',')
end

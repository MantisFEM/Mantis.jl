"""
    export_geometry_to_vtk(
    	geo::Geometry.AbstractGeometry,
    	filename::String;
    	n_subcells::Int=1,
    	degree::Int=4,
    	output_directory_tree=[pwd()],
    )

Export the geometry to a VTK file.

# Arguments

  - `geo::Geometry.AbstractGeometry`: The geometry to be exported.
  - `filename::String`: The name of the output file.
  - `n_subcells::Int`: The number of subcells to be used in the visualization.
  - `degree::Int`: The degree of the basis functions used in the visualization.
  - `output_directory_tree`: A vector of strings representing the directory tree. Defaults to
    the current working directory.
"""
function export_geometry_to_vtk(
    geo::Geometry.AbstractGeometry,
    filename::String;
    n_subcells::Int=1,
    degree::Int=4,
    output_directory_tree=[pwd()],
)
    output_file = export_path(output_directory_tree, filename)
    plot(
        geo;
        vtk_filename=output_file,
        n_subcells=n_subcells,
        degree=degree,
        ascii=false,
        compress=false,
    )

    return nothing
end

"""
    export_form_fields_to_vtk(
    	form_sols,
    	var_names,
    	filename;
    	n_subcells::Int=1,
    	degree::Int=4,
    	output_directory_tree=[pwd()],
    )

Export the form solutions to VTK files.

# Arguments

  - `form_sols::Vector{Forms.AbstractForm}`: The form solutions to be exported.
  - `var_names::Vector{String}`: The names of the form solutions.
  - `filename::String`: The name of the output file.
  - `n_subcells::Int`: The number of subcells to be used in the visualization.
  - `degree::Int`: The degree of the basis functions used in the visualization.
  - `output_directory_tree`: A vector of strings representing the directory tree. Defaults to
    the current working directory.
"""
function export_form_fields_to_vtk(
    form_sols,
    var_names,
    filename;
    n_subcells::Int=1,
    degree::Int=4,
    output_directory_tree=[pwd()],
)
    for (form_sol, var_name) in zip(form_sols, var_names)
        println("Writing form '$var_name' to file ...")
        output_file = export_path(output_directory_tree, "$filename-$var_name")
        plot(
            form_sol;
            vtk_filename=output_file,
            n_subcells=n_subcells,
            degree=degree,
            ascii=false,
            compress=false,
        )
    end

    return nothing
end

"""
    export_form_fields_to_vtk(
    	form_sols, filename; n_subcells::Int=1, degree::Int=4, output_directory_tree=[pwd()]
    )

Export the form solutions to VTK files.

# Arguments

  - `form_sols::Vector{Forms.AbstractForm}`: The form solutions to be exported.
  - `filename::String`: The name of the output file.
  - `n_subcells::Int`: The number of subcells to be used in the visualization.
  - `degree::Int`: The degree of the basis functions used in the visualization.
    output_directory_tree=[pwd()],
"""
function export_form_fields_to_vtk(
    form_sols, filename; n_subcells::Int=1, degree::Int=4, output_directory_tree=[pwd()]
)
    for form in form_sols
        label = Forms.get_label(form)
        sanitised_label = replace(label, "\$" => "")
        sanitised_label = replace(sanitised_label, "\\" => "")
        println("Writing form '$sanitised_label' to file ...")
        output_file = export_path(output_directory_tree, "$filename-$sanitised_label")
        plot(
            form;
            vtk_filename=output_file,
            n_subcells=n_subcells,
            degree=degree,
            ascii=false,
            compress=false,
        )
    end

    return nothing
end

# Only usable if Makie is also loaded.
function plot_solution end
function plot_basis end

# Pad for 1D.
function _pad_point(point::NTuple{1, T}, pad_value=zero(T)) where {T}
    return (point[1], pad_value)
end
# Otherwise, no padding needed.
function _pad_point(point, pad_value=0.0)
    return point
end

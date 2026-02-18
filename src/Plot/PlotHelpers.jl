"""
    visualize_geometry(geo::Geometry.AbstractGeometry, filename::String; n_subcells::Int = 1, degree::Int = 4, output_directory_tree::Vector{String} = ["examples", "data", "output"])

Export the geometry to a VTK file.

# Arguments
- `geo::Geometry.AbstractGeometry`: The geometry to be exported.
- `filename::String`: The name of the output file.
- `n_subcells::Int`: The number of subcells to be used in the visualization.
- `degree::Int`: The degree of the basis functions used in the visualization.
- `output_directory_tree::Vector{String}`: A vector of strings representing the directory tree.
"""
function export_geometry_to_vtk(
    geo::Geometry.AbstractGeometry,
    filename::String;
    n_subcells::Int=1,
    degree::Int=4,
    output_directory_tree::Vector{String}=["examples", "data", "output"],
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
    export_form_fields_to_vtk(form_sols::Vector{Forms.AbstractForm}, var_names::Vector{String}, filename::String; n_subcells::Int = 1, degree::Int = 4, output_directory_tree::Vector{String} = ["examples", "data", "output"])

Export the form solutions to VTK files.

# Arguments
- `form_sols::Vector{Forms.AbstractForm}`: The form solutions to be exported.
- `var_names::Vector{String}`: The names of the form solutions.
- `filename::String`: The name of the output file.
- `n_subcells::Int`: The number of subcells to be used in the visualization.
- `degree::Int`: The degree of the basis functions used in the visualization.
- `output_directory_tree::Vector{String}`: A vector of strings representing the directory tree.
"""
function export_form_fields_to_vtk(
    form_sols,
    var_names,
    filename;
    n_subcells::Int=1,
    degree::Int=4,
    output_directory_tree::Vector{String}=["examples", "data", "output"],
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
    export_form_fields_to_vtk(form_sols::Vector{Forms.AbstractForm}, var_names::Vector{String}, filename::String; n_subcells::Int = 1, degree::Int = 4, output_directory_tree::Vector{String} = ["examples", "data", "output"])

Export the form solutions to VTK files.

# Arguments
- `form_sols::Vector{Forms.AbstractForm}`: The form solutions to be exported.
- `filename::String`: The name of the output file.
- `n_subcells::Int`: The number of subcells to be used in the visualization.
- `degree::Int`: The degree of the basis functions used in the visualization.
- `output_directory_tree::Vector{String}`: A vector of strings representing the directory tree.
"""
function export_form_fields_to_vtk(
    form_sols,
    filename;
    n_subcells::Int=1,
    degree::Int=4,
    output_directory_tree::Vector{String}=["examples", "data", "output"],
)
    for form in form_sols
        label = form.label
        println("Writing form '$label' to file ...")
        output_file = export_path(output_directory_tree, "$filename-$label")
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

function plot_solution(
    fields::T, num_plot_points_per_element=25; title="Solution", xlabel="x", ylabel="phi(x)"
) where {n_fields, T <: NTuple{n_fields, Forms.AbstractFormField{1}}}
    fig = Figure()
    ax = Axis(fig[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel)

    geometry = Forms.get_geometry(fields[1])

    n_elements = Geometry.get_num_elements(geometry)
    xi = Points.CartesianPoints((LinRange(0.0, 1.0, num_plot_points_per_element),))

    colors = [:blue, :green, :red, :purple, :orange, :black, :pink, :brown]
    for field_id in eachindex(fields)
        field = fields[field_id]
        color_i = colors[field_id]
        for element_idx in 1:n_elements
            form_eval, _ = Forms.evaluate(field, element_idx, xi)
            x = Geometry.evaluate(geometry, element_idx, xi)

            lines!(ax, x[:], form_eval[1]; color=color_i, label=field.label)
        end
    end
    fig[1, 2] = Legend(fig, ax; marge=true, unique=true)

    return fig
end

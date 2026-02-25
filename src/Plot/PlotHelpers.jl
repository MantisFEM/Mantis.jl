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

# Only usable if GLMakie is also loaded.
function plot_solution end
function plot_solution(
    fields::T,
    num_plot_points_per_element=25;
    title="Solution",
    xlabel="x",
    ylabel=L"\phi(x)",
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

function plot_topology(geometry; edge_color=:darkolivegreen3, vertex_color=:orange)
    fig = GLMakie.Figure()
    ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    topology = Geometry.get_topology(geometry)
    manifold_dim = Topology.get_manifold_dim(topology)
    image_dim = Geometry.get_image_dim(geometry)

    TPoint = GLMakie.Point{image_dim, Float32}

    if manifold_dim > 2
        error("not implemented")
    end

    # First plot all element lines per patch.
    xi_element = Points.CartesianPoints((LinRange(0.0, 1.0, 2), LinRange(0.0, 1.0, 2)))
    for element_id in 1:Geometry.get_num_elements(geometry)
        element_vertices = Geometry.evaluate(geometry, element_id, xi_element)
        element_vertices_points = [
            GLMakie.Point2f(element_vertices[i, :]...) for i in axes(element_vertices, 1)
        ]
        GLMakie.linesegments!(element_vertices_points; color=:black)
        GLMakie.linesegments!(
            [
                element_vertices_points[1],
                element_vertices_points[3],
                element_vertices_points[2],
                element_vertices_points[4],
            ];
            color=:black,
        )
    end

    # Then plot the edges (with label).
    edge_alignment = (
        (:left, :center), (:right, :center), (:center, :bottom), (:center, :top)
    )
    edge_coordinates = Geometry.get_edge_coordinates(TPoint, geometry)
    for edge_id in 1:size(topology, 2)
        for patch_id in topology[2, manifold_dim + 1][edge_id]
            # Go through all patches so that we know the global and local ids.
            local_edge_id = Topology.get_local_id(topology, patch_id, edge_id, 2)

            starting_coordinate, final_coordinate = edge_coordinates[edge_id]
            GLMakie.lines!([starting_coordinate, final_coordinate]; color=edge_color)
            edge_midpoint = (
                (starting_coordinate[1] + final_coordinate[1]) / 2,
                (starting_coordinate[2] + final_coordinate[2]) / 2,
            )

            GLMakie.text!(
                edge_midpoint;
                text="($patch_id, $local_edge_id)",
                align=edge_alignment[local_edge_id],
                color=edge_color,
            )

            if patch_id == topology[2, manifold_dim + 1][edge_id][1]
                # Also add the global edge number
                GLMakie.text!(
                    edge_midpoint;
                    text="Edge $edge_id",
                    align=(:center, :center),
                    color=edge_color,
                    offset=(0, 20),
                )
            end
        end
    end

    # Then plot the vertices (with global and local label) on top of this to make them more
    # visible.
    alignment = ((:left, :bottom), (:right, :bottom), (:right, :top), (:left, :top))
    vertex_coordinates = Geometry.get_vertex_coordinates(TPoint, geometry)
    for vertex_id in eachindex(vertex_coordinates)
        for patch_id in topology[1, manifold_dim + 1][vertex_id]
            # Go through all patches so that we know the global and local ids.
            local_vertex_id = Topology.get_local_id(topology, patch_id, vertex_id, 1)

            coordinate = vertex_coordinates[vertex_id]
            GLMakie.scatter!(coordinate; marker=:circle, markersize=10, color=vertex_color)
            GLMakie.text!(
                coordinate;
                text="($patch_id, $local_vertex_id)",
                align=alignment[local_vertex_id],
                color=vertex_color,
            )
            if patch_id == topology[1, manifold_dim + 1][vertex_id][1]
                # Also add the global vertex number
                GLMakie.text!(
                    coordinate;
                    text="Vertex $vertex_id",
                    align=(:center, :center),
                    color=vertex_color,
                    offset=(0, 20),
                )
            end
        end
    end

    return fig
end

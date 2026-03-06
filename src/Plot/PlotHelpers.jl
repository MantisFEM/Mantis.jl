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

# 1D
function plot_topology(
    geometry::Geometry.AbstractGeometry{manifold_dim, 1};
    edge_color=:darkolivegreen3,
    vertex_color=:orange,
    draw_elements=false,
) where {manifold_dim}
    fig = GLMakie.Figure()
    ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    vertex_alignment = ((:left, :bottom), (:right, :bottom))
    edge_alignment = ((:center, :bottom),)

    fig = _plot_topology!(
        fig,
        ax,
        geometry,
        vertex_alignment,
        edge_alignment;
        edge_color=edge_color,
        vertex_color=vertex_color,
        draw_elements=draw_elements,
    )

    return fig
end

# 2D
function plot_topology(
    geometry::Geometry.AbstractGeometry{manifold_dim, 2};
    edge_color=:darkolivegreen3,
    vertex_color=:orange,
    face_color=(:purple, 0.2),
    draw_elements=false,
    draw_faces=false,
    plot_points_per_element=5,
) where {manifold_dim}
    fig = GLMakie.Figure()
    ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y")

    vertex_alignment = ((:left, :bottom), (:right, :bottom), (:right, :top), (:left, :top))
    edge_alignment = (
        (:left, :center), (:right, :center), (:center, :bottom), (:center, :top)
    )

    fig = _plot_topology!(
        fig,
        ax,
        geometry,
        vertex_alignment,
        edge_alignment;
        edge_color=edge_color,
        vertex_color=vertex_color,
        face_color=face_color,
        draw_elements=draw_elements,
        draw_faces=draw_faces,
        plot_points_per_element=plot_points_per_element,
    )

    return fig
end

# 3D
function plot_topology(
    geometry::Geometry.AbstractGeometry{manifold_dim, 3};
    edge_color=:darkolivegreen3,
    vertex_color=:orange,
    face_color=(:purple, 0.2),
    draw_elements=false,
    draw_faces=true,
    plot_points_per_element=5,
) where {manifold_dim}
    fig = GLMakie.Figure()
    ax = GLMakie.Axis3(fig[1, 1]; viewmode=:fit)

    vertex_alignment = (
        (:left, :bottom),
        (:right, :bottom),
        (:right, :top),
        (:left, :top),
        (:left, :bottom),
        (:right, :bottom),
        (:right, :top),
        (:left, :top),
    )
    edge_alignment = (
        (:left, :center),
        (:right, :center),
        (:center, :bottom),
        (:center, :top),
        (:left, :center),
        (:right, :center),
        (:center, :bottom),
        (:center, :top),
        (:left, :center),
        (:right, :center),
        (:center, :bottom),
        (:center, :top),
    )

    fig = _plot_topology!(
        fig,
        ax,
        geometry,
        vertex_alignment,
        edge_alignment;
        edge_color=edge_color,
        vertex_color=vertex_color,
        face_color=face_color,
        draw_elements=draw_elements,
        draw_faces=draw_faces,
        plot_points_per_element=plot_points_per_element,
    )

    return fig
end

function _plot_topology!(
    fig,
    ax,
    geometry,
    vertex_alignment,
    edge_alignment;
    vertex_color=:orange,
    edge_color=:darkolivegreen3,
    face_color=(:purple, 0.2),
    draw_elements=false,
    draw_faces=true,
    plot_points_per_element=5,
)
    topology = Geometry.get_topology(geometry)
    manifold_dim = Topology.get_manifold_dim(topology)
    image_dim = Geometry.get_image_dim(geometry)

    TPoint = GLMakie.Point{image_dim, Float32}

    # First plot all element lines per patch, if desired.
    if draw_elements
        xi_element = Points.CartesianPoints(
            ntuple(manifold_dim) do i
                return LinRange(0.0, 1.0, 2)
            end,
        )
        if image_dim == manifold_dim
            permutations = ([1, 3, 2, 4], [1, 5, 2, 6, 3, 7, 4, 8])
        elseif image_dim == manifold_dim + 1
            permutations = ([1, 3, 2, 4], [1, 3, 2, 4])
        end
        for element_id in 1:Geometry.get_num_elements(geometry)
            element_vertices = Geometry.evaluate(geometry, element_id, xi_element)
            element_vertices_points = [
                _pad_point(TPoint(element_vertices[i, :]...)) for
                i in axes(element_vertices, 1)
            ]
            # Element lines in the x direction.
            GLMakie.linesegments!(element_vertices_points; color=:black)
            if image_dim == 1
                # Also add a vertical bar for the edges, otherwise they are invisble.
                GLMakie.scatter!(
                    element_vertices_points; marker=:vline, markersize=15, color=:black
                )
            end

            # Element lines in the other directions.
            for dim in 1:(manifold_dim - 1)
                GLMakie.linesegments!(
                    element_vertices_points[permutations[dim]]; color=:black
                )
            end
        end
    end

    # Then plot the edges (with label).
    edge_coordinates = Geometry.get_edge_coordinates(TPoint, geometry)
    for edge_id in 1:size(topology, 2)
        if image_dim == 1
            # Edges and patches are equal, so are their ids.
            patch_ids = edge_id:edge_id
        else
            patch_ids = topology[2, manifold_dim + 1][edge_id]
        end
        for patch_id in patch_ids
            # Go through all patches so that we know the global and local ids.
            if image_dim == 1
                local_edge_id = 1
            else
                local_edge_id = Topology.get_local_id(topology, patch_id, edge_id, 2)
            end

            # Recompute the edge coordinates using the current patch_id to get the
            # orientation on the patch.
            starting_coordinate_local_raw, final_coordinate_local_raw = Geometry.get_edge_coordinates(
                TPoint, geometry, patch_id, local_edge_id
            )
            starting_coordinate_local = _pad_point(starting_coordinate_local_raw)
            final_coordinate_local = _pad_point(final_coordinate_local_raw)

            # Compute the edge as a curve
            elements_on_edge = Geometry.get_elements(geometry, patch_id, local_edge_id, 1)
            xi_elements = Geometry.get_canonical_points(
                eltype(TPoint), geometry, local_edge_id, 1, plot_points_per_element
            )
            for element_id in elements_on_edge
                curved_edge_coordinates = Geometry.evaluate(
                    geometry, element_id, xi_elements
                )
                curved_edge_points = [
                    _pad_point(TPoint(curved_edge_coordinates[i, :]...)) for
                    i in axes(curved_edge_coordinates, 1)
                ]
                GLMakie.lines!(curved_edge_points; color=edge_color)
            end

            num_elements_on_edge = length(elements_on_edge)
            if isodd(num_elements_on_edge)
                middle_element = elements_on_edge[div(num_elements_on_edge, 2) + 1]
                xi = Geometry.get_canonical_points(
                    eltype(TPoint), geometry, local_edge_id, 1, 3
                )
            else
                middle_element = elements_on_edge[div(num_elements_on_edge, 2)]
                xi = Geometry.get_canonical_points(
                    eltype(TPoint), geometry, local_edge_id, 1, 2
                )
            end
            curved_midpoint_coordinates = Geometry.evaluate(geometry, middle_element, xi)
            edge_midpoint = _pad_point(TPoint(curved_midpoint_coordinates[2, :]...))
            # dir = Geometry.jacobian(geometry, middle_element, xi)
            # t_tup = Geometry.get_tangent_vector(eltype(TPoint), geometry, local_edge_id, 1)
            # t = [(t_tup)...]
            # dir_midpoint = _pad_point(TPoint((dir[2] * t)...))
            dir_midpoint = final_coordinate_local - starting_coordinate_local

            # if isapprox(t_tup[1], zero(eltype(TPoint)))
            #     dir_midpoint_raw = _pad_point(TPoint((dir[2] * t)...))
            #     dir_midpoint = TPoint(0.0, dir_midpoint_raw[2], dir_midpoint_raw[3])
            # else
            #     dir_midpoint = _pad_point(TPoint((dir[2] * t)...))
            # end

            GLMakie.text!(
                edge_midpoint;
                text="($patch_id, $local_edge_id)",
                align=edge_alignment[local_edge_id],
                color=edge_color,
            )
            GLMakie.arrows2d!(
                edge_midpoint,
                dir_midpoint;
                align=:center,
                lengthscale=0.25,
                color=edge_color,
            )

            if patch_id == patch_ids[1]
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
    vertex_coordinates = Geometry.get_vertex_coordinates(TPoint, geometry)
    for vertex_id in eachindex(vertex_coordinates)
        for patch_id in topology[1, manifold_dim + 1][vertex_id]
            # Go through all patches so that we know the global and local ids.
            local_vertex_id = abs(Topology.get_local_id(topology, patch_id, vertex_id, 0))

            coordinate_raw = vertex_coordinates[vertex_id]
            coordinate = _pad_point(coordinate_raw)
            GLMakie.scatter!(coordinate; marker=:circle, markersize=10, color=vertex_color)
            GLMakie.text!(
                coordinate;
                text="($patch_id, $local_vertex_id)",
                align=vertex_alignment[local_vertex_id],
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

    # Plot the faces
    if manifold_dim >= 2 && draw_faces
        for face_id in 1:size(topology, 3)
            if manifold_dim == 2
                patch_ids = [face_id]
            else
                patch_ids = topology[3, manifold_dim + 1][face_id]
            end
            for patch_id in patch_ids
                # Go through all patches so that we know the global and local ids.
                if manifold_dim == 2
                    local_face_id = 1
                else
                    local_face_id = Topology.get_local_id(topology, patch_id, face_id, 3)
                end

                # Compute the surface
                elements_on_face = Geometry.get_elements(
                    geometry, patch_id, local_face_id, 2
                )
                xi_elements = Geometry.get_canonical_points(
                    eltype(TPoint), geometry, local_face_id, 2, plot_points_per_element
                )
                for element_id in elements_on_face
                    curved_face_coordinates = Geometry.evaluate(
                        geometry, element_id, xi_elements
                    )
                    x = reshape(
                        curved_face_coordinates[:, 1],
                        (plot_points_per_element, plot_points_per_element),
                    )
                    y = reshape(
                        curved_face_coordinates[:, 2],
                        (plot_points_per_element, plot_points_per_element),
                    )
                    if image_dim > 2
                        z = reshape(
                            curved_face_coordinates[:, 3],
                            (plot_points_per_element, plot_points_per_element),
                        )
                        GLMakie.surface!(
                            x,
                            y,
                            z;
                            color=fill(
                                face_color, plot_points_per_element, plot_points_per_element
                            ),
                            shading=false,
                            transparency=true,
                        )
                    else
                        GLMakie.surface!(
                            x,
                            y;
                            color=fill(
                                face_color, plot_points_per_element, plot_points_per_element
                            ),
                            shading=false,
                            transparency=true,
                        )
                    end
                end
            end
        end
    end

    return fig
end

# Pad for 1D
function _pad_point(point::GLMakie.Point{1, T}, pad_value=zero(T)) where {T}
    return GLMakie.Point{2, T}(point[1], pad_value)
end
function _pad_point(point::NTuple{1, T}, pad_value=zero(T)) where {T}
    return (point[1], pad_value)
end
# Otherwise, no padding needed.
function _pad_point(point, pad_value=0.0)
    return point
end

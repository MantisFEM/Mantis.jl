module MakieExt

using Mantis
using Makie

function Mantis.Plot.plot_solution(
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

function Mantis.Plot.plot_solution(
    field::Forms.AbstractFormField{2},
    plot_points_per_element=25;
    draw_patch_wireframe=true,
    colorrange = (-1.0, 1.0),
    colormap = :viridis
)
    fig = Figure()
    ax = Axis3(fig[1, 1]; viewmode=:fit)

    geometry = Forms.get_geometry(field)

    TPoint = Point{2, Float32}

    # First plot the geometry wireframe if desired.
    if draw_patch_wireframe
        _draw_patch_wireframe!(ax, geometry, plot_points_per_element, TPoint)
    end

    num_elements = Geometry.get_num_elements(geometry)
    xi = Points.CartesianPoints(ntuple(2) do i
        return LinRange(0.0, 1.0, plot_points_per_element)
    end)

    for element_id in 1:num_elements
        position_coordinates = Geometry.evaluate(Forms.get_geometry(field), element_id, xi)

        x = reshape(
            position_coordinates[:, 1],
            (plot_points_per_element, plot_points_per_element),
        )
        y = reshape(
            position_coordinates[:, 2],
            (plot_points_per_element, plot_points_per_element),
        )

        function_values = Forms.evaluate(field, element_id, xi)[1]
        z = reshape(
            function_values[1],
            (plot_points_per_element, plot_points_per_element),
        )
        surface!(
            ax,
            x,
            y,
            z;
            shading=true,
            transparency=false,
            colorrange=colorrange,
            colormap=colormap,
        )
    end

    Colorbar(fig[1, 2], limits=colorrange, colormap=colormap)

    return fig
end

function Mantis.Plot.plot_solution(
    field::Forms.AbstractFormField{3},
    plot_points_per_element=5;
    draw_patch_wireframe=true,
    colorrange = (-1.0, 1.0),
    colormap = :viridis
)
    fig = Figure()
    ax = Axis3(fig[1, 1]; viewmode=:fit)

    geometry = Forms.get_geometry(field)

    TPoint = Point{3, Float32}

    # First plot the geometry wireframe if desired.
    if draw_patch_wireframe
        _draw_patch_wireframe!(ax, geometry, plot_points_per_element, TPoint)
    end

    num_elements = Geometry.get_num_elements(geometry)
    xi = Points.CartesianPoints(ntuple(3) do i
        return LinRange(0.0, 1.0, plot_points_per_element)
    end)

    for element_id in 1:num_elements
        position_coordinates = Geometry.evaluate(Forms.get_geometry(field), element_id, xi)
        x = reshape(
            position_coordinates[:, 1],
            (plot_points_per_element, plot_points_per_element, plot_points_per_element),
        )
        y = reshape(
            position_coordinates[:, 2],
            (plot_points_per_element, plot_points_per_element, plot_points_per_element),
        )
        z = reshape(
            position_coordinates[:, 3],
            (plot_points_per_element, plot_points_per_element, plot_points_per_element),
        )
        points = TPoint.(x, y, z)

        function_values = Forms.evaluate(field, element_id, xi)[1]
        vals = reshape(
            function_values[1],
            (plot_points_per_element, plot_points_per_element, plot_points_per_element),
        )
        volume!(
            ax,
            x[1,1,1] .. x[end,end,end],
            y[1,1,1] .. y[end,end,end],
            z[1,1,1] .. z[end,end,end],
            vals;
            shading=true,
            transparency=true,
            colorrange=colorrange,
            colormap=colormap,
            ssao=true,
        )
    end

    Colorbar(fig[1, 2], limits=colorrange, colormap=colormap)

    return fig
end

# 1D
function Mantis.Plot.plot_topology(
    geometry::Geometry.AbstractGeometry{manifold_dim, 1};
    edge_color=:darkolivegreen3,
    vertex_color=:orange,
    draw_elements=false,
) where {manifold_dim}
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y")

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
function Mantis.Plot.plot_topology(
    geometry::Geometry.AbstractGeometry{manifold_dim, 2};
    edge_color=:darkolivegreen3,
    vertex_color=:orange,
    face_color=(:purple, 0.2),
    draw_elements=false,
    draw_faces=false,
    plot_points_per_element=5,
) where {manifold_dim}
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y")

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
function Mantis.Plot.plot_topology(
    geometry::Geometry.AbstractGeometry{manifold_dim, 3};
    edge_color=:darkolivegreen3,
    vertex_color=:orange,
    face_color=(:purple, 0.2),
    draw_elements=false,
    draw_faces=true,
    plot_points_per_element=5,
) where {manifold_dim}
    fig = Figure()
    ax = Axis3(fig[1, 1]; viewmode=:fit)

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

    TPoint = Point{image_dim, Float32}

    # First plot all element lines per patch, if desired.
    if draw_elements
        _draw_elements!(ax, geometry, TPoint)
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
                local_edge_id = Topology.get_local_id(topology, patch_id, edge_id, 1)
            end

            # Recompute the edge coordinates using the current patch_id to get the
            # orientation on the patch.
            starting_coordinate_local_raw, final_coordinate_local_raw = Geometry.get_edge_coordinates(
                TPoint, geometry, patch_id, abs(local_edge_id)
            )
            starting_coordinate_local = Mantis.Plot._pad_point(starting_coordinate_local_raw)
            final_coordinate_local = Mantis.Plot._pad_point(final_coordinate_local_raw)

            # Compute the edge as a curve
            elements_on_edge = Geometry.get_elements(
                geometry, patch_id, abs(local_edge_id), 1
            )
            xi_elements = Geometry.get_canonical_points(
                eltype(TPoint), geometry, abs(local_edge_id), 1, plot_points_per_element
            )
            for element_id in elements_on_edge
                curved_edge_coordinates = Geometry.evaluate(
                    geometry, element_id, xi_elements
                )
                curved_edge_points = [
                    Mantis.Plot._pad_point(TPoint(curved_edge_coordinates[i, :]...)) for
                    i in axes(curved_edge_coordinates, 1)
                ]
                lines!(curved_edge_points; color=edge_color)
            end

            num_elements_on_edge = length(elements_on_edge)
            if isodd(num_elements_on_edge)
                middle_element = elements_on_edge[div(num_elements_on_edge, 2) + 1]
                xi = Geometry.get_canonical_points(
                    eltype(TPoint), geometry, abs(local_edge_id), 1, 3
                )
            else
                middle_element = elements_on_edge[div(num_elements_on_edge, 2)]
                xi = Geometry.get_canonical_points(
                    eltype(TPoint), geometry, abs(local_edge_id), 1, 2
                )
            end
            curved_midpoint_coordinates = Geometry.evaluate(geometry, middle_element, xi)
            edge_midpoint = Mantis.Plot._pad_point(TPoint(curved_midpoint_coordinates[2, :]...))
            # dir = Geometry.jacobian(geometry, middle_element, xi)
            # t_tup = Geometry.get_tangent_vector(eltype(TPoint), geometry, local_edge_id, 1)
            # t = [(t_tup)...]
            # dir_midpoint = Mantis.Plot._pad_point(TPoint((dir[2] * t)...))
            dir_midpoint = final_coordinate_local - starting_coordinate_local

            # if isapprox(t_tup[1], zero(eltype(TPoint)))
            #     dir_midpoint_raw = Mantis.Plot._pad_point(TPoint((dir[2] * t)...))
            #     dir_midpoint = TPoint(0.0, dir_midpoint_raw[2], dir_midpoint_raw[3])
            # else
            #     dir_midpoint = Mantis.Plot._pad_point(TPoint((dir[2] * t)...))
            # end

            text!(
                edge_midpoint;
                text="($patch_id, $local_edge_id)",
                align=edge_alignment[abs(local_edge_id)],
                color=edge_color,
            )
            arrows2d!(
                edge_midpoint,
                dir_midpoint;
                align=:center,
                lengthscale=0.25,
                color=edge_color,
            )

            if patch_id == patch_ids[1]
                # Also add the global edge number
                text!(
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
            coordinate = Mantis.Plot._pad_point(coordinate_raw)
            scatter!(coordinate; marker=:circle, markersize=10, color=vertex_color)
            text!(
                coordinate;
                text="($patch_id, $local_vertex_id)",
                align=vertex_alignment[local_vertex_id],
                color=vertex_color,
            )
            if patch_id == topology[1, manifold_dim + 1][vertex_id][1]
                # Also add the global vertex number
                text!(
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
                    local_face_id = Topology.get_local_id(topology, patch_id, face_id, 2)
                end

                # Compute the surface
                elements_on_face = Geometry.get_elements(
                    geometry, patch_id, abs(local_face_id), 2
                )
                xi_elements = Geometry.get_canonical_points(
                    eltype(TPoint), geometry, abs(local_face_id), 2, plot_points_per_element
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
                        surface!(
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
                        surface!(
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


function Mantis.Plot.plot_basis(
    space::FunctionSpaces.AbstractFESpace{1, 1};
    ids=1:FunctionSpaces.get_num_basis(space),
    draw_patch_wireframe=true,
    draw_elements=true,
    plot_points_per_element=75,
    color_per_basis=true,
    colorrange=(0.0, 1.0),
    colormap=Makie.wong_colors(),
    show_legend=true,
    show_colormap=true,
)
    fig = Figure()
    ax = Axis(fig[1, 1])

    fig = _plot_basis!(
        fig,
        ax,
        space;
        ids=ids,
        draw_patch_wireframe=draw_patch_wireframe,
        draw_elements=draw_elements,
        plot_points_per_element=plot_points_per_element,
        color_per_basis=color_per_basis,
        colorrange=colorrange,
        colormap=colormap,
        show_legend=show_legend,
        show_colormap=show_colormap,
    )

    return fig
end

function Mantis.Plot.plot_basis(
    space::FunctionSpaces.AbstractFESpace{2, 1};
    ids=1:FunctionSpaces.get_num_basis(space),
    draw_patch_wireframe=true,
    draw_elements=false,
    plot_points_per_element=25,
    color_per_basis=false,
    colorrange=(0.0, 1.0),
    colormap=:viridis,
    show_legend=true,
    show_colormap=true,
)
    fig = Figure()
    ax = Axis3(fig[1, 1]; viewmode=:fit)

    fig = _plot_basis!(
        fig,
        ax,
        space;
        ids=ids,
        draw_patch_wireframe=draw_patch_wireframe,
        draw_elements=draw_elements,
        plot_points_per_element=plot_points_per_element,
        color_per_basis=color_per_basis,
        colorrange=colorrange,
        colormap=colormap,
        show_legend=show_legend,
        show_colormap=show_colormap,
    )

    return fig
end

function _plot_basis!(
    fig,
    ax,
    space;
    ids,
    draw_patch_wireframe,
    draw_elements,
    plot_points_per_element,
    color_per_basis,
    colorrange,
    colormap,
    show_legend,
    show_colormap,
)
    if typeof(ids) <: Integer
        basis_ids = ids:ids
    else
        basis_ids = ids
    end

    manifold_dim = FunctionSpaces.get_manifold_dim(space)
    geometry = FunctionSpaces.get_geometry(space)
    image_dim = Geometry.get_image_dim(geometry)

    TPoint = Point{image_dim, Float32}
    xi = Points.CartesianPoints(
        ntuple(manifold_dim) do i
            return LinRange(zero(eltype(TPoint)), one(eltype(TPoint)), plot_points_per_element)
        end,
    )
    point_shape = ntuple(image_dim) do i
        return plot_points_per_element
    end

    # First plot the geometry wireframe and/or the elements, if desired.
    if draw_patch_wireframe
        _draw_patch_wireframe!(
            ax, FunctionSpaces.get_geometry(space), plot_points_per_element, TPoint
        )
    end
    if draw_elements
        _draw_elements!(ax, geometry, TPoint)
    end

    # Then plot the basis functions on top.
    for basis_id in basis_ids
        # Get the elements on which the current basis function is supported.
        element_ids = FunctionSpaces.get_support(space, basis_id)

        for element_id in element_ids
            position_coordinates = Geometry.evaluate(geometry, element_id, xi)

            x = reshape(
                position_coordinates[:, 1],
                point_shape,
            )
            if manifold_dim == 2
                y = reshape(
                    position_coordinates[:, 2],
                    (plot_points_per_element, plot_points_per_element),
                )
            end

            function_values, bases = FunctionSpaces.evaluate(space, element_id, xi)
            basis_index = findfirst(isequal(basis_id), bases)
            z = reshape(
                function_values[1][1][1][:, basis_index],
                point_shape,
            )
            if manifold_dim == 1
                if color_per_basis
                    color = colormap[basis_id % length(colormap) + 1]
                    lines!(
                        ax,
                        x,
                        z;
                        color=color,
                        label=string(basis_id),
                    )
                else
                    lines!(
                        ax,
                        x,
                        z;
                        color=z,
                        colorrange=colorrange,
                        colormap=colormap,
                    )
                end

            else
                if color_per_basis
                    color = colormap[basis_id % length(colormap) + 1]
                    surface!(
                        ax,
                        x,
                        y,
                        z;
                        shading=true,
                        transparency=false,
                        color=fill(
                            color, plot_points_per_element, plot_points_per_element
                        ),
                        label=string(basis_id),
                    )
                else
                    surface!(
                        ax,
                        x,
                        y,
                        z;
                        shading=true,
                        transparency=false,
                        colorrange=colorrange,
                        colormap=colormap,
                    )
                end
            end
        end
    end

    if show_legend && color_per_basis
        Legend(fig[1, 2], ax, unique=true)
    end
    if show_colormap && !color_per_basis
        Colorbar(fig[1, 2], limits=colorrange, colormap=colormap)
    end

    return fig
end

function _draw_elements!(ax, geometry, TPoint)
    manifold_dim = Geometry.get_manifold_dim(geometry)
    image_dim = Geometry.get_image_dim(geometry)

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
            Mantis.Plot._pad_point(TPoint(element_vertices[i, :]...)) for
            i in axes(element_vertices, 1)
        ]
        # Element lines in the x direction.
        linesegments!(element_vertices_points; color=:black)
        if image_dim == 1
            # Also add a vertical bar for the edges, otherwise they are invisble.
            scatter!(
                ax, element_vertices_points; marker=:vline, markersize=15, color=:black
            )
        end

        # Element lines in the other directions.
        for dim in 1:(manifold_dim - 1)
            linesegments!(
                ax, element_vertices_points[permutations[dim]]; color=:black
            )
        end
    end
end

function _draw_patch_wireframe!(ax, geometry, plot_points_per_element, TPoint)
    topology = Geometry.get_topology(geometry)
    image_dim = Geometry.get_image_dim(geometry)
    for edge_id in 1:size(topology, 2)
        if image_dim == 1
            # Edges and patches are equal, so are their ids.
            patch_ids = edge_id:edge_id
        else
            patch_ids = topology[2, Geometry.get_manifold_dim(geometry) + 1][edge_id]
        end
        for patch_id in patch_ids
            # Go through all patches so that we know the global and local ids.
            if image_dim == 1
                local_edge_id = 1
            else
                local_edge_id = Topology.get_local_id(topology, patch_id, edge_id, 1)
            end

            # Compute the edge as a curve
            elements_on_edge = Geometry.get_elements(
                geometry, patch_id, abs(local_edge_id), 1
            )
            xi_elements = Geometry.get_canonical_points(
                eltype(TPoint), geometry, abs(local_edge_id), 1, plot_points_per_element
            )
            for element_id in elements_on_edge
                curved_edge_coordinates = Geometry.evaluate(
                    geometry, element_id, xi_elements
                )
                curved_edge_points = [
                    Mantis.Plot._pad_point(TPoint(curved_edge_coordinates[i, :]...)) for
                    i in axes(curved_edge_coordinates, 1)
                ]
                lines!(ax, curved_edge_points; color=:black)
            end
        end
    end
end

# Pad for 1D
function Mantis.Plot._pad_point(point::Point{1, T}, pad_value=zero(T)) where {T}
    return Point{2, T}(point[1], pad_value)
end

end

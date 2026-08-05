module MakieExt

using Mantis
using Makie
import LaTeXStrings

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
    xi = Points.TensorProductPoints((LinRange(0.0, 1.0, num_plot_points_per_element),))

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

function Mantis.Plot.plot_basis(
    space::FunctionSpaces.AbstractFESpace{1, 1};
    ids=1:FunctionSpaces.get_num_basis(space),
    draw_elements=true,
    plot_points_per_element=75,
    color_per_basis=true,
    colorrange=(0.0, 1.0),
    colormap=Makie.wong_colors(),
    show_legend=true,
    show_colormap=true,
    label_prefix="",
    axis_kwargs...,
)
    fig = Figure()
    ax = Axis(fig[1, 1]; axis_kwargs...)

    fig = _plot_basis!(
        fig,
        ax,
        space;
        ids=ids,
        draw_elements=draw_elements,
        plot_points_per_element=plot_points_per_element,
        color_per_basis=color_per_basis,
        colorrange=colorrange,
        colormap=colormap,
        show_legend=show_legend,
        show_colormap=show_colormap,
        label_prefix=label_prefix,
    )

    return fig
end

function Mantis.Plot.plot_basis(
    space::FunctionSpaces.AbstractFESpace{2, 1};
    ids=1:FunctionSpaces.get_num_basis(space),
    draw_elements=false,
    plot_points_per_element=25,
    color_per_basis=false,
    colorrange=(0.0, 1.0),
    colormap=:viridis,
    show_legend=true,
    show_colormap=true,
    label_prefix="",
    axis_kwargs...,
)
    fig = Figure()
    ax = Axis3(fig[1, 1]; viewmode=:fit, axis_kwargs...)

    fig = _plot_basis!(
        fig,
        ax,
        space;
        ids=ids,
        draw_elements=draw_elements,
        plot_points_per_element=plot_points_per_element,
        color_per_basis=color_per_basis,
        colorrange=colorrange,
        colormap=colormap,
        show_legend=show_legend,
        show_colormap=show_colormap,
        label_prefix=label_prefix,
    )

    return fig
end

function sanitise_label(label_prefix::AbstractString, main_label::AbstractString)
    label = label_prefix * main_label

    return label
end
function sanitise_label(label_prefix::LaTeXStrings.LaTeXString, main_label::AbstractString)
    # Remove trailing $ and add it back at the end.
    label = LaTeXStrings.LaTeXString(label_prefix.s[1:(end - 1)] * main_label * "\$")

    return label
end

function _plot_basis!(
    fig,
    ax,
    space;
    ids,
    draw_elements,
    plot_points_per_element,
    color_per_basis,
    colorrange,
    colormap,
    show_legend,
    show_colormap,
    label_prefix,
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
    xi = Points.TensorProductPoints(
        ntuple(manifold_dim) do i
            return LinRange(
                zero(eltype(TPoint)), one(eltype(TPoint)), plot_points_per_element
            )
        end,
    )
    point_shape = ntuple(image_dim) do i
        return plot_points_per_element
    end

    # First plot the geometry wireframe and/or the elements, if desired.
    if draw_elements
        _draw_elements!(ax, geometry, TPoint)
    end

    # Then plot the basis functions on top.
    for basis_id in basis_ids

        # Get the elements on which the current basis function is supported.
        element_ids = FunctionSpaces.get_support(space, basis_id)

        for element_id in element_ids
            position_coordinates = Geometry.evaluate(geometry, element_id, xi)

            x = reshape(position_coordinates[:, 1], point_shape)
            if manifold_dim == 2
                y = reshape(
                    position_coordinates[:, 2],
                    (plot_points_per_element, plot_points_per_element),
                )
            end

            function_values, bases = FunctionSpaces.evaluate(space, element_id, xi)
            basis_index = findfirst(isequal(basis_id), bases)
            z = reshape(function_values[1][1][1][:, basis_index], point_shape)
            if manifold_dim == 1
                if color_per_basis
                    color = colormap[basis_id % length(colormap) + 1]
                    label = sanitise_label(label_prefix, string(basis_id))
                    lines!(ax, x, z; color=color, label=label)
                else
                    lines!(ax, x, z; color=z, colorrange=colorrange, colormap=colormap)
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
                        color=fill(color, plot_points_per_element, plot_points_per_element),
                        label=label_prefix*string(basis_id),
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
        Legend(fig[1, 2], ax; unique=true)
    end
    if show_colormap && !color_per_basis
        Colorbar(fig[1, 2]; limits=colorrange, colormap=colormap)
    end

    return fig
end

function _draw_elements!(ax, geometry, TPoint)
    manifold_dim = Geometry.get_manifold_dim(geometry)
    image_dim = Geometry.get_image_dim(geometry)

    xi_element = Points.TensorProductPoints(
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
            linesegments!(ax, element_vertices_points[permutations[dim]]; color=:black)
        end
    end
end

# Pad for 1D
function Mantis.Plot._pad_point(point::Point{1, T}, pad_value=zero(T)) where {T}
    return Point{2, T}(point[1], pad_value)
end

end

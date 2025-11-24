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
function export_geometry_to_vtk(geo::Geometry.AbstractGeometry, filename::String; n_subcells::Int = 1, degree::Int = 4, output_directory_tree::Vector{String} = ["examples", "data", "output"])

    output_file = export_path(output_directory_tree, filename)
    plot(geo; vtk_filename = output_file, n_subcells = n_subcells, degree = degree, ascii = false, compress = false)

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
function export_form_fields_to_vtk(form_sols, var_names, filename; n_subcells::Int = 1, degree::Int = 4, output_directory_tree::Vector{String} = ["examples", "data", "output"])

    for (form_sol, var_name) in zip(form_sols, var_names)
        println("Writing form '$var_name' to file ...")
        output_file = export_path(output_directory_tree, "$filename-$var_name")
        plot(form_sol; vtk_filename = output_file, n_subcells = n_subcells, degree = degree, ascii = false, compress = false)
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
function export_form_fields_to_vtk(form_sols, filename; n_subcells::Int = 1, degree::Int = 4, output_directory_tree::Vector{String} = ["examples", "data", "output"])

    for form in form_sols
        label = form.label
        println("Writing form '$label' to file ...")
        output_file = export_path(output_directory_tree, "$filename-$label")
        plot(form; vtk_filename = output_file, n_subcells = n_subcells, degree = degree, ascii = false, compress = false)
    end

    return nothing
end

"""
    visualize_tensor_product_controlnet(control_points::Array{Float64}, manifold_dim::Int, range_dim::Int, periodic::Vector{Bool}, filename::String; output_directory_tree::Vector{String} = ["examples", "data", "output"])

Given a tensor-product control net, create a `manifold_dim`-linear geometry and use it to visualize the control net connectivity.

# Arguments
- `control_points::Array{Float64, projective_dim}`: The control points of the tensor product control net arranged in a `projective_dim=manifold_dim+1`-dimensional array.
- `manifold_dim::Int`: The dimension of the manifold.
- `range_dim::Int`: The dimension of the range.
- `periodic::Vector{Bool}`: A vector of booleans indicating whether the manifold is periodic.
- `filename::String`: The name of the output file.
- `output_directory_tree::Vector{String}`: A vector of strings representing the directory tree.
"""
function visualize_tensor_product_controlnet(control_points::Array{Float64,projective_dim}, manifold_dim::Int, range_dim::Int, periodic::Vector{Bool}, filename::String; output_directory_tree::Vector{String} = ["examples", "data", "output"]) where {projective_dim}
    if projective_dim != manifold_dim+1
        throw(ArgumentError("The control points must be structured into a `manifold_dim+1`-dimensional array."))
    end
    if length(periodic) != manifold_dim
        throw(ArgumentError("The periodicity vector must have length equal to the manifold dimension."))
    end

    # create bilinear geometry
    B = [FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry(LinRange(0.0, 1.0, size(control_points,i)+periodic[i])), 1, 0) for i in 1:manifold_dim]

    # impose periodicity if required
    for i = 1:manifold_dim
        if periodic[i]
            B[i] = FunctionSpaces.GTBSplineSpace((B[i],), [0])
        end
    end

    # create n-linear tensor product space
    TP = FunctionSpaces.TensorProductSpace(B...)

    # create geometry
    geo = Geometry.FEGeometry(TP, reshape(control_points, :, range_dim))

    # export to vtk
    visualize_geometry(geo, filename; n_subcells=1, degree=1, output_directory_tree=output_directory_tree)

    return nothing
end

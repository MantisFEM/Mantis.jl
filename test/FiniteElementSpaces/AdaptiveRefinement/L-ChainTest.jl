import Mantis

#using Test
using Plots
using ColorSchemes
using Random
using LinearAlgebra

function plot_grid(ne1, ne2, marked_elements, marked_bsplines, refinement_domain)
    # Create a grid of zeros (white background)
    grid = fill(0, ne1, ne2)
    pal = Plots.palette(:tab10)

    # Mark specific elements
    els_per_dim = Mantis.FunctionSpaces._get_num_elements_per_space(TB)
    for idx in refinement_domain
        row, col = Mantis.FunctionSpaces.linear_to_ordered_index(idx, els_per_dim)
        grid[row, col] = 1
    end
    for idx in marked_bsplines
        row, col = Mantis.FunctionSpaces.linear_to_ordered_index(idx, els_per_dim)
        grid[row, col] = 2
    end
    for idx in marked_elements
        row, col = Mantis.FunctionSpaces.linear_to_ordered_index(idx, els_per_dim)
        grid[row, col] = -1  
    end

    # Plot the grid with background
    custom_gradient = cgrad([:orange, :white, :red, pal[1]])
    plt = heatmap(grid', clim=(-1,2), st=:heatmap, color=custom_gradient, legend=false, aspect_ratio=:equal, framestyle=:box, fmt=:pdf)

    # Add grid lines for better visualization
    for i in 1:ne1+1
        vline!([i-0.5], color=:black, linewidth=0.5)
    end
    for i in 1:ne2+1
        hline!([i-0.5], color=:black, linewidth=0.5)
    end
    xlims!(0.5,ne1+0.5)
    ylims!(0.5,ne2+0.5)

    # Show the plot
    display(plt)
    #savefig("many-marked.pdf")
end

function plot_geometry(hspace)
    degrees = Mantis.FunctionSpaces.get_polynomial_degree_per_dim(hspace.spaces[1])
    nxi_per_dim = maximum(degrees) + 1
    nxi = nxi_per_dim^2
    xi_per_dim = collect(range(0,1, nxi_per_dim))
    xi = Matrix{Float64}(undef, nxi,2)

    xi_eval = (xi_per_dim, xi_per_dim)

    for (idx,x) ∈ enumerate(Iterators.product(xi_per_dim, xi_per_dim))
        xi[idx,:] = [x[1] x[2]]
    end

    xs = Matrix{Float64}(undef, Mantis.FunctionSpaces.get_num_elements(hspace)*nxi,2)
    nx = size(xs)[1]

    A = zeros(nx, Mantis.FunctionSpaces.get_num_basis(hspace))

    for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)
        level = Mantis.FunctionSpaces.get_active_level(hspace.active_elements, el)
        element_id = Mantis.FunctionSpaces.get_active_id(hspace.active_elements, el)

        max_ind_els = Mantis.FunctionSpaces._get_num_elements_per_space(hspace.spaces[level])
        ordered_index = Mantis.FunctionSpaces.linear_to_ordered_index(element_id, max_ind_els)

        borders_x = Mantis.Mesh.get_element(hspace.spaces[level].function_space_1.knot_vector.patch_1d, ordered_index[1])
        borders_y = Mantis.Mesh.get_element(hspace.spaces[level].function_space_2.knot_vector.patch_1d, ordered_index[2])

        x = [(borders_x[1] .+ xi[:,1] .* (borders_x[2] - borders_x[1])) (borders_y[1] .+ xi[:,2] .* (borders_y[2] - borders_y[1]))]

        idx = (el-1)*nxi+1:el*nxi
        xs[idx,:] = x

        local eval = Mantis.FunctionSpaces.evaluate(hspace, el, xi_eval, 0)

        A[idx, eval[2]] = eval[1][1][1]
    end

    coeffs = A \ xs
    println(size(A))
    println(size(coeffs))
    display(coeffs)

    hierarchical_geo = Mantis.Geometry.FEMGeometry(hspace, coeffs)

    # Generate the Plot
    Mantis_folder =  dirname(dirname(pathof(Mantis)))
    data_folder = joinpath(Mantis_folder, "test", "data")
    output_data_folder = joinpath(data_folder, "output", "Geometry")

    output_filename = "L_chain_geo_test.vtu"
    output_file = joinpath(output_data_folder, output_filename)
    Mantis.Plot.plot(hierarchical_geo; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)
end

function plot_field(hspace)
    degrees = Mantis.FunctionSpaces.get_polynomial_degree_per_dim(hspace.spaces[1])
    nxi_per_dim = maximum(degrees) + 1
    nxi = nxi_per_dim^2
    xi_per_dim = collect(range(0,1, nxi_per_dim))
    xi = Matrix{Float64}(undef, nxi,2)

    xi_eval = (xi_per_dim, xi_per_dim)

    for (idx,x) ∈ enumerate(Iterators.product(xi_per_dim, xi_per_dim))
        xi[idx,:] = [x[1] x[2]]
    end

    xs = Matrix{Float64}(undef, Mantis.FunctionSpaces.get_num_elements(hspace)*nxi,2)
    nx = size(xs)[1]

    A = zeros(nx, Mantis.FunctionSpaces.get_num_basis(hspace))

    for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)
        level = Mantis.FunctionSpaces.get_active_level(hspace.active_elements, el)
        element_id = Mantis.FunctionSpaces.get_active_id(hspace.active_elements, el)

        max_ind_els = Mantis.FunctionSpaces._get_num_elements_per_space(hspace.spaces[level])
        ordered_index = Mantis.FunctionSpaces.linear_to_ordered_index(element_id, max_ind_els)

        borders_x = Mantis.Mesh.get_element(hspace.spaces[level].function_space_1.knot_vector.patch_1d, ordered_index[1])
        borders_y = Mantis.Mesh.get_element(hspace.spaces[level].function_space_2.knot_vector.patch_1d, ordered_index[2])

        x = [(borders_x[1] .+ xi[:,1] .* (borders_x[2] - borders_x[1])) (borders_y[1] .+ xi[:,2] .* (borders_y[2] - borders_y[1]))]

        idx = (el-1)*nxi+1:el*nxi
        xs[idx,:] = x

        local eval = Mantis.FunctionSpaces.evaluate(hspace, el, xi_eval, 0)

        A[idx, eval[2]] = eval[1][1][1]
    end

    coeffs = A \ xs

    hierarchical_geo = Mantis.Geometry.FEMGeometry(hspace, coeffs)
    field_coeffs = Matrix{Float64}(LinearAlgebra.I,Mantis.FunctionSpaces.get_num_basis(hspace), Mantis.FunctionSpaces.get_num_basis(hspace))
    field = Mantis.Fields.FEMField(hspace, field_coeffs)

    # Generate the Plot
    Mantis_folder =  dirname(dirname(pathof(Mantis)))
    data_folder = joinpath(Mantis_folder, "test", "data")
    output_data_folder = joinpath(data_folder, "output", "Field")

    output_filename = "L_chain_geo_test.vtu"
    output_file = joinpath(output_data_folder, output_filename)
    Mantis.Plot.plot(hierarchical_geo, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)
end

# Test parameters
ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0, 1, ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0, 1, ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

deg1 = 2
deg2 = 2

B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])
TB = Mantis.FunctionSpaces.TensorProductSpace(B1, B2)

nsub1 = 2
nsub2 = 2

n_steps = 2
n_els_per_step = 4
new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(TB, (nsub1, nsub2))
spaces = [TB, new_space]
operators = [new_operator]
hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[], Int[]])
Random.seed!(28) 

for step ∈ 1:n_steps
    L = Mantis.FunctionSpaces.get_num_levels(hspace)

    new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(spaces[L], (nsub1, nsub2))

    marked_elements_per_level = [Int[] for _ ∈ 1:L-2]
    push!(marked_elements_per_level, rand(Mantis.FunctionSpaces.get_level_active(hspace.active_elements, L-1)[2], n_els_per_step))
    push!(marked_elements_per_level, Int[])

    refinement_domains = Mantis.FunctionSpaces.get_refinement_domain(hspace, Int[], new_operator)

    if length(refinement_domains) > L
        push!(operators, new_operator)
        push!(spaces, new_space)
    end
    hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, refinement_domains, false)
end

plot_geometry(hspace)
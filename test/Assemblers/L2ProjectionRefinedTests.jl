import Mantis

using Test
using LinearAlgebra, Plots, SparseArrays

function get_thb_geometry(hspace::Mantis.FunctionSpaces.HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:Mantis.FunctionSpaces.AbstractFiniteElementSpace{n}, T<:Mantis.FunctionSpaces.AbstractTwoScaleOperator}
    L = Mantis.FunctionSpaces.get_num_levels(hspace)
    
    coefficients = Matrix{Float64}(undef, (Mantis.FunctionSpaces.get_num_basis(hspace), 2))

    id_sum = 1
    for level ∈ 1:1:L
        max_ind_basis = Mantis.FunctionSpaces._get_num_basis_per_space(hspace.spaces[level])
        x_greville_points = Mantis.FunctionSpaces.get_greville_points(hspace.spaces[level].function_space_1.knot_vector)
        y_greville_points = Mantis.FunctionSpaces.get_greville_points(hspace.spaces[level].function_space_2.knot_vector)
        grevile_mesh(x_id,y_id) = x_greville_points[x_id]*y_greville_points[y_id]
        
        _, level_active_basis = Mantis.FunctionSpaces.get_level_active(hspace.active_basis, level)

        for (y_count, y_id) ∈ enumerate(y_greville_points)
            for (x_count, x_id) ∈ enumerate(x_greville_points)
                if Mantis.FunctionSpaces.ordered_to_linear_index((x_count, y_count), max_ind_basis) ∈ level_active_basis
                    coefficients[id_sum, :] .= [x_id, y_id]
                    id_sum += 1
                end
            end
        end
    end

    return Mantis.Geometry.FEMGeometry(hspace, coefficients)
end

function get_hb_geometry(hspace::Mantis.FunctionSpaces.HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:Mantis.FunctionSpaces.AbstractFiniteElementSpace{n}, T<:Mantis.FunctionSpaces.AbstractTwoScaleOperator}
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

    return Mantis.Geometry.FEMGeometry(hspace, coeffs)
end

function fe_run(source_function, trial_space, test_space, geom, q_nodes, 
    q_weights, exact_sol, p, k, case, bc_dirichlet = Dict{Int, Float64}(), output_to_file=false, test=true, 
    verbose=false)
    if verbose
        println("Starting setup of problem and assembler for case "*case*" ...")
    end
    # Setup the element assembler.
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(source_function,
                                                trial_space,
                                                test_space,
                                                geom,
                                                q_nodes,
                                                q_weights)

    # Setup the global assembler.
    global_assembler = Mantis.Assemblers.Assembler(bc_dirichlet)

    if verbose
        println("Assembling ...")
    end
    A, b = global_assembler(Mantis.Assemblers.l2_weak_form, weak_form_inputs)

    if test
        if verbose
            println("Running tests ...")
        end
        @test isapprox(A, A', rtol=1e-12)  # Full system matrices need not be symmetric due to the boundary conditions.
        @test isempty(nullspace(Matrix(A)))  # Only works on dense matrices!
        @test LinearAlgebra.cond(Matrix(A)) < 1e10
    end

    # Solve & add bcs.
    if verbose
        println("Solving ",size(A,1)," x ",size(A,2)," sized system ...")
    end
    sol = A \ b
    
    sol_rsh = reshape(sol, :, 1)


    # This is for the plotting. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.
    if output_to_file
        if verbose
        println("Writing to file ...")
        end

        out_deg = maximum([1, maximum(p)])

        nels = Mantis.Geometry.get_num_elements(geom)
        output_filename = "L2-Projection-p$p-k$k-nels$nels-case-"*case*".vtu"
        output_filename_error = "L2-Projection-Error-p$p-k$k-nels$nels-case-"*case*".vtu"
        field = Mantis.Fields.FEMField(trial_space, sol_rsh)

        output_file = joinpath(output_data_folder, output_filename)
        output_file_error = joinpath(output_data_folder, output_filename_error)
        Mantis.Plot.plot(geom, field; vtk_filename = output_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
        Mantis.Plot.plot(geom, field, exact_sol; vtk_filename = output_file_error, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
    end

    # Compute error
    if verbose
        println("Computing L^2 error w.r.t. exact solution ...")
    end
    err_assembler = Mantis.Assemblers.AssemblerErrorPerElement(q_nodes, q_weights)
    err_per_element = err_assembler(trial_space, sol_rsh, geom, exact_sol)
    if verbose
        println("The L^2 error is: ", sqrt(sum(err_per_element)))
    end

    # Extra check to test if the metric computation was correct.
    # err2 = err_assembler(trial_space, reshape(ones(length(sol)-1), :, 1), geom, (x,y) -> 0.0)
    # println(err2)

    return err_per_element, maximum(err_per_element), A
end

case = "gianelli2012-approximation-test"
source_function(x, y) = 2/(3*exp(sqrt((10*x-3)^2 + (10*y-3)^2))) + 2/(3*exp(sqrt((10*x)^2 + (10*y)^2))) + 2/(3*exp(sqrt((10*x+3)^2 + (10*y+3)^2)))

Mantis_folder = dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "L2Projection") # Create this folder first if you haven't done so yet.
output_to_file = true
test = false
verbose = false
verbose_step = false
verbose_convergence = false

reference_errors = [4.493e-1, 3.877e-1, 2.223e-1, 1.153e-1, 3.047e-2, 2.987e-3]
reference_dofs = [36, 64, 112, 190, 456, 600]

# THB Refinement
nsteps = 6
dorfler_parameter = 0.3

nels = (4, 4)
p = (2, 2)
k = (1, 1)

patches = map(n -> Mantis.Mesh.Patch1D(collect(range(-1, 1, n+1))), nels)
bsplines = [Mantis.FunctionSpaces.BSplineSpace(patches[i], p[i], [-1; fill(k[i], nels[i]-1); -1]) for i ∈ 1:2]
tensor_bspline = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)

nsubdiv = (2, 2)
spaces = [tensor_bspline]
operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]])
hspace_geo = get_thb_geometry(hspace)
bc = Dict{Int, Float64}()
q_nodes, q_weights = Mantis.Quadrature.tensor_product_rule(p .+ 2, Mantis.Quadrature.gauss_legendre)

err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, case, bc, output_to_file, test, verbose)

if verbose_step
    println("Step 0:")
    println("Polynomial degrees: $p with regularities: $k.")
    println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
    println("DoF: $(Mantis.FunctionSpaces.get_num_basis(hspace)).")
    println("Maximum error: $max_error. \n")
end

thb_errors = Vector{Float64}(undef, nsteps+1)
thb_dofs = Vector{Int}(undef, nsteps+1)
thb_nnz = Vector{Int}(undef, nsteps+1)

thb_dofs[1] = Mantis.FunctionSpaces.get_num_basis(hspace)
thb_errors[1] = max_error
SparseArrays.dropzeros!(A)
thb_nnz[1] = SparseArrays.nnz(A)

for step ∈ 1:nsteps
    # Solve current hierarchical space solution

    L = Mantis.FunctionSpaces.get_num_levels(hspace)
    new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], nsubdiv)

    dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
    marked_domains = Mantis.FunctionSpaces.get_marked_domains(hspace, dorfler_marking, new_operator, false)

    if length(marked_domains) > L
        push!(spaces, new_space)
        push!(operators, new_operator)
    end
    
    hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_domains, true)
    hspace_geo = get_thb_geometry(hspace)

    err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, case, bc, output_to_file, test, verbose)

    if verbose_step
        println("Step $step") 
        println("Maximum error: $(max_error).") 
        println("Number of marked_elements: $(length(dorfler_marking)).")

        println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
        println("DoF: $(Mantis.FunctionSpaces.get_num_basis(hspace)). \n")
    end
    thb_errors[step+1] = max_error
    thb_dofs[step+1] = Mantis.FunctionSpaces.get_num_basis(hspace)
    SparseArrays.dropzeros!(A)
    thb_nnz[step+1] = SparseArrays.nnz(A)
end

output_to_file = false
# HB Refinement

spaces = [tensor_bspline]
operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]])
hspace_geo = get_hb_geometry(hspace)

err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, case, bc, output_to_file, test, verbose)

hb_errors = Vector{Float64}(undef, nsteps+1)
hb_dofs = Vector{Int}(undef, nsteps+1)
hb_nnz = Vector{Int}(undef, nsteps+1)

hb_dofs[1] = Mantis.FunctionSpaces.get_num_basis(hspace)
hb_errors[1] = max_error
SparseArrays.dropzeros!(A)
hb_nnz[1] = SparseArrays.nnz(A)

for step ∈ 1:nsteps
    # Solve current hierarchical space solution

    L = Mantis.FunctionSpaces.get_num_levels(hspace)
    new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], nsubdiv)

    dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
    marked_domains = Mantis.FunctionSpaces.get_marked_domains(hspace, dorfler_marking, new_operator, false)

    if length(marked_domains) > L
        push!(spaces, new_space)
        push!(operators, new_operator)
    end
    
    hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_domains, false)
    hspace_geo = get_hb_geometry(hspace)

    err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, case, bc, output_to_file, test, verbose)

    if verbose_step
        println("Step $step") 
        println("Maximum error: $(max_error).") 
        println("Number of marked_elements: $(length(dorfler_marking)).")

        println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
        println("DoF: $(Mantis.FunctionSpaces.get_num_basis(hspace)). \n")
    end
    hb_errors[step+1] = max_error
    hb_dofs[step+1] = Mantis.FunctionSpaces.get_num_basis(hspace)
    SparseArrays.dropzeros!(A)
    hb_nnz[step+1] = SparseArrays.nnz(A)
end

output_to_file = false
# THB L-chain

spaces = [tensor_bspline]
operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]])
hspace_geo = get_hb_geometry(hspace)

err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, case, bc, output_to_file, test, verbose)

lchain_errors = Vector{Float64}(undef, nsteps)
lchain_dofs = Vector{Int}(undef, nsteps)
lchain_nnz = Vector{Int}(undef, nsteps)

lchain_dofs[1] = Mantis.FunctionSpaces.get_num_basis(hspace)
lchain_errors[1] = max_error
SparseArrays.dropzeros!(A)
lchain_nnz[1] = SparseArrays.nnz(A)

for step ∈ 1:nsteps-1
    # Solve current hierarchical space solution

    L = Mantis.FunctionSpaces.get_num_levels(hspace)
    new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], nsubdiv)

    dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
    marked_domains = Mantis.FunctionSpaces.get_marked_domains(hspace, dorfler_marking, new_operator, true)

    if length(marked_domains) > L
        push!(spaces, new_space)
        push!(operators, new_operator)
    end
    
    hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_domains, true)
    hspace_geo = get_hb_geometry(hspace)

    err_per_element, max_error, A = fe_run(source_function, hspace, hspace, hspace_geo, q_nodes, q_weights, source_function, p, k, case, bc, output_to_file, test, verbose)

    if verbose_step
        println("Step $step") 
        println("Maximum error: $(max_error).") 
        println("Number of marked_elements: $(length(dorfler_marking)).")

        println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
        println("DoF: $(Mantis.FunctionSpaces.get_num_basis(hspace)). \n")
    end
    lchain_errors[step+1] = max_error
    lchain_dofs[step+1] = Mantis.FunctionSpaces.get_num_basis(hspace)
    SparseArrays.dropzeros!(A)
    lchain_nnz[step+1] = SparseArrays.nnz(A)
end


output_to_file = false
# h-Refinement

nsubdivs = 2

h_errors = Vector{Float64}(undef, nsubdivs+1)
h_dofs = Vector{Int}(undef, nsubdivs+1)
for subdiv_factor ∈ 0:nsubdivs
    nels = nels .* 2^subdiv_factor
    
    patches = map(n -> Mantis.Mesh.Patch1D(collect(range(-1, 1, n+1))), nels)
    bsplines = [Mantis.FunctionSpaces.BSplineSpace(patches[i], p[i], [-1; fill(k[i], nels[i]-1); -1]) for i ∈ 1:2]
    
    trial_space = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)
    test_space = Mantis.FunctionSpaces.TensorProductSpace(bsplines...)
    bc = Dict{Int, Float64}()
    
    geom_cartesian = Mantis.Geometry.CartesianGeometry(Tuple(patches[i].breakpoints for i ∈ 1:2))
    
    # Setup the quadrature rule.
    q_nodes, q_weights = Mantis.Quadrature.tensor_product_rule(p .+ 2, Mantis.Quadrature.gauss_legendre)
    h_errors[subdiv_factor+1] = fe_run(source_function, trial_space, test_space, geom_cartesian, q_nodes, q_weights, source_function, p, k, case, bc, output_to_file, test, verbose)[2]
    h_dofs[subdiv_factor+1] = Mantis.FunctionSpaces.get_num_basis(trial_space)
end

println("HB number of non-zero entries")
println(hb_nnz)
println("THB number of non-zero entries")
println(thb_nnz)
println("THB L-chain number of non-zero entries")
println(lchain_nnz)
println("number of non-zero entries ratio (THB/HB)")
println( map( x -> round(x, digits=2), thb_nnz ./ hb_nnz))
#println("number of non-zero entries ratio (lchain/HB)")
#println( map( x -> round(x, digits=2), lchain_nnz ./ hb_nnz))

plt = plot(reference_dofs, reference_errors, label="reference", xlabel="DoF", ylabel="Error", yscale=:log10, xscale=:log10)
plot!(hb_dofs, hb_errors, label="HB")
plot!(thb_dofs, thb_errors, label="THB")
plot!(lchain_dofs, lchain_errors, label="THB L-chain")
plot!(h_dofs, h_errors, label="h")

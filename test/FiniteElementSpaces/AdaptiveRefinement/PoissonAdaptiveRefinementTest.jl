import Mantis

using Test
using Random
using LinearAlgebra

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

function fe_run(source_function, trial_space, test_space, geom, q_nodes, 
    q_weights, exact_sol, p, k, case, bc_dirichlet = Dict{Int, Float64}(), output_to_file=false, test=true, verbose=false)
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
    A, b = global_assembler(Mantis.Assemblers.poisson_weak_form_1, weak_form_inputs)

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

        Mantis_folder =  dirname(dirname(pathof(Mantis)))
        data_folder = joinpath(Mantis_folder, "test", "data")
        output_data_folder = joinpath(data_folder, "output", "Poisson")

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
        println("The L^2 error is: ", sqrt(sum((err_per_element))))
    end

    # Extra check to test if the metric computation was correct.
    # err2 = err_assembler(trial_space, reshape(ones(length(sol)-1), :, 1), geom, (x,y) -> 0.0)
    # println(err2)

    return err_per_element
end

function test()
    # Dimension
    n_2d = 2
    # Test parameters
    ne1 = 10
    ne2 = 10
    breakpoints1 = collect(range(0, 1, ne1+1))
    patch1 = Mantis.Mesh.Patch1D(breakpoints1)
    breakpoints2 = collect(range(0, 1, ne2+1))
    patch2 = Mantis.Mesh.Patch1D(breakpoints2)

    deg1 = 2
    deg2 = 2
    reg1 = 1
    reg2 = 1

    B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(reg1, ne1-1); -1])
    B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(reg2, ne2-1); -1])
    TB = Mantis.FunctionSpaces.TensorProductSpace(B1, B2)

    q_nodes, q_weights = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1), Mantis.Quadrature.gauss_legendre)

    # Domain. The length of the domain is chosen so that the normal 
    # derivatives of the exact solution are zero at the boundary. This is 
    # the only Neumann b.c. that we can specify at the moment.

    function forcing_sine(x::Float64, y::Float64)
        return 8.0 * pi^2 * sinpi(2.0 * x) * sinpi(2.0 * y)
    end

    function exact_sol_sine(x::Float64, y::Float64)
        return sinpi(2.0 * x) * sinpi(2.0 * y)
    end

    nsub1 = 2
    nsub2 = 2

    case = "sine2d-THB-adaptive-refinement-no-lchain"
    output_to_file = false
    test = false
    verbose = false
    n_steps = 3
    dorfler_parameter = 0.3
    spaces = [TB]
    operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]
    hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, [Int[]])
    hspace_geo = get_thb_geometry(hspace)
    bc_dirichlet_2d = Dict{Int, Float64}(i => 0.0 for i in Mantis.FunctionSpaces.get_boundary_dof_indices(hspace))
    err_per_element = fe_run(forcing_sine, hspace, hspace, hspace_geo, 
    q_nodes, q_weights, exact_sol_sine, (deg1, deg2), 
    (reg1, reg2), case, bc_dirichlet_2d, output_to_file, test, verbose)
    
    if verbose
        println("Initial data:")
        println("Maximum error: $(sqrt(maximum(err_per_element))).") 
        println("Polynomial degrees: $((deg1, deg2)) with regularities: $((reg1, reg2)).")
        println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
        println("DoF: $(Mantis.FunctionSpaces.get_num_basis(hspace)). \n")
    end

    for step ∈ 1:n_steps
        # Solve current hierarchical space solution

        L = Mantis.FunctionSpaces.get_num_levels(hspace)
        new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(hspace.spaces[L], (nsub1, nsub2))
        dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
        marked_domains = Mantis.FunctionSpaces.get_marked_domains(hspace, dorfler_marking, new_operator, false)

        if length(marked_domains) > L
            push!(spaces, new_space)
            push!(operators, new_operator)
        end
        
        hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_domains, true)
        hspace_geo = get_thb_geometry(hspace)
        bc_dirichlet_2d = Dict{Int, Float64}(i => 0.0 for i in Mantis.FunctionSpaces.get_boundary_dof_indices(hspace))
        err_per_element = fe_run(forcing_sine, hspace, hspace, hspace_geo, 
        q_nodes, q_weights, exact_sol_sine, (deg1, deg2), 
        (reg1, reg2), case, bc_dirichlet_2d, output_to_file, test, verbose)

        if verbose
            println("Step $step") 
            println("Maximum error: $(sqrt(maximum(err_per_element))).") 
            println("Number of marked_elements: $(length(dorfler_marking)).")

            println("Number of elements: $(Mantis.FunctionSpaces.get_num_elements(hspace)).")
            println("DoF: $(Mantis.FunctionSpaces.get_num_basis(hspace)). \n")
        end
    end
end

test()
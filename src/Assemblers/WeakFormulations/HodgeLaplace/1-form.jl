############################################################################################
#                                  1-form Hodge Laplacian                                  #
############################################################################################

"""
    one_form_hodge_laplacian(
        inputs::AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
    )

Function for assembling the weak form of the 1-form Hodge Laplacian problem.

# Arguments
- `inputs::AbstractInputs`: The inputs for the weak form assembly, including test, trial and
    forcing terms.
- `dΩ::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to use for the integral
    evaluation.

# Returns
- `lhs_expressions<:NTuple{num_lhs_rows, NTuple{num_lhs_cols, AbstractRealValuedOperator}}`:
    The left-hand side of the weak form, which is a tuple of tuples contain all the blocks
    of the left-hand side matrix.
- `rhs_expressions<:NTuple{num_rhs_rows, NTuple{num_rhs_cols, AbstractRealValuedOperator}}`:
    The right-hand side of the weak form, which is a tuple of tuples contain all the blocks
    of the right-hand side matrix.
"""
function one_form_hodge_laplacian(
    inputs::AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    τ⁰, v¹ = Assemblers.get_test_forms(inputs)
    σ⁰, u¹ = Assemblers.get_trial_forms(inputs)
    f¹ = Assemblers.get_forcing(inputs)
    A_11 = ∫(τ⁰ ∧ ★(σ⁰), dΩ)
    A_12 = -∫(d(τ⁰) ∧ ★(u¹), dΩ)
    A_21 = ∫(v¹ ∧ ★(d(σ⁰)), dΩ)
    A_22 = ∫(d(v¹) ∧ ★(d(u¹)), dΩ)
    lhs_expressions = ((A_11, A_12), (A_21, A_22))
    b_21 = ∫(v¹ ∧ ★(f¹), dΩ)
    rhs_expressions = ((0,), (b_21,))

    return lhs_expressions, rhs_expressions
end

"""
    solve_one_form_hodge_laplacian(X⁰, X¹, f¹, dΩ)

Returns the solution of the weak form of the 1-form Hodge Laplacian.

# Arguments
- `X⁰`: The 0-form space to use as trial and test space.
- `X¹`: The 1-form space to use as trial and test space.
- `f¹`: The forcing term to use for the right-hand side of the weak formulation.
- `dΩ`: The quadrature rule to use for the assembly.

# Returns
- `δu¹ₕ::Forms.FormField`: The 0-form solution of the weak-formulation.
- `u¹ₕ::Forms.FormField`: The 1-form solution of the weak-formulation.
"""
function solve_one_form_hodge_laplacian(X⁰, X¹, f¹, dΩ, bc_type="")
    weak_form_inputs = Assemblers.WeakFormInputs((X⁰, X¹), (f¹,))
    lhs_expressions, rhs_expressions = one_form_hodge_laplacian(weak_form_inputs, dΩ)
    weak_form = WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    if isempty(bc_type)
        A, b = Assemblers.assemble(weak_form)
    elseif bc_type == "dirichlet"
        offset = Forms.get_num_basis(X⁰)
        trace_ids = Forms.trace_basis_idxs(X¹)
        bc = Dict{Int, Float64}(i + offset => 0.0 for i in trace_ids)
        A, b = Assemblers.assemble(weak_form, bc)
    end

    sol = vec(A \ b)
    δu¹ₕ, u¹ₕ = Forms.build_form_fields((X⁰, X¹), sol)

    return δu¹ₕ, u¹ₕ
end

"""
    solve_one_form_hodge_laplacian(
        complex::C,
        forcing_function::Function,
        dΩₐ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        num_steps::Int,
        dorfler_parameter::Float64,
        dΩₑ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        Lchains::Bool;
        verbose::Bool=false
    ) where {manifold_dim, num_forms, C <: NTuple{num_forms, Forms.AbstractFormSpace}}

Returns the solution of the weak form of the 1-form Hodge Laplacian from an adaptive loop.

# Arguments
- `complex::C`: The initial de Rham complex to use for the problem.
- `forcing_function::Function`: The function to use for the forcing term.
- `dΩₐ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}`: The quadrature rule to use
    for the assembly.
- `num_steps::Int`: The number of steps to use for the adaptive loop.
- `dorfler_parameter::Float64`: The parameter to use for the Dörfler marking.
- `dΩₑ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}`: The quadrature rule to use
    for the error estimation.
- `Lchains::Bool`: Whether to use L-chains for the refinement.
- `verbose::Bool=false`: Whether to print the progress of the adaptive loop.
"""
function solve_one_form_hodge_laplacian(
    complex::C,
    problem_data::Function,
    dΩₐ::Quadrature.StandardQuadrature{manifold_dim},
    num_steps::Int,
    dorfler_parameter::Float64,
    dΩₑ::Quadrature.StandardQuadrature{manifold_dim},
    Lchains::Bool,
    bc_type="";
    VERBOSE::Bool=false,
    EXPORT::Bool=false,
    Plot::Module,
) where {manifold_dim, num_forms, C <: NTuple{num_forms, Forms.AbstractFormSpace}}
    if VERBOSE
        println("Solving the problem on initial step...")
    end

    # Exact solution on initial step
    δu¹, u¹, f¹ = problem_data(1, Forms.get_geometry(complex[1]))
    δu¹ₕ, u¹ₕ = solve_one_form_hodge_laplacian(complex[1], complex[2], f¹, dΩₐ, bc_type)
    err_per_element = Analysis.compute_error_per_element(u¹ₕ, u¹, dΩₑ)
    if VERBOSE
        error_δu = Analysis.L2_norm(δu¹ - δu¹ₕ, dΩₑ)
        error_u = Analysis.L2_norm(u¹ - u¹ₕ, dΩₑ)
        @show error_δu, error_u
    end

    if EXPORT
        Plot.export_form_fields_to_vtk(
            (δu¹, δu¹ₕ, u¹, u¹ₕ), "Example-L-chains-$(Lchains)-step-0"
        )
    end

    for step in 1:num_steps
        if VERBOSE
            println("Solving the problem on step $step...")
        end

        H⁰ = FunctionSpaces.get_component_spaces(Forms.get_fe_space(complex[1]))[1]
        dorfler_marking = FunctionSpaces.get_dorfler_marking(
            err_per_element, dorfler_parameter
        )
        marked_elements_per_level = FunctionSpaces.get_padding_per_level(
            H⁰, dorfler_marking
        )
        if Lchains
            domains = FunctionSpaces.update_domains_with_lchains!(
                H⁰, marked_elements_per_level
            )
            complex = Forms.update_hierarchical_de_rham_complex(complex, domains)
        else
            complex = Forms.update_hierarchical_de_rham_complex(
                complex, marked_elements_per_level
            )
        end

        geo = Forms.get_geometry(complex[1])
        dΩₐ = Quadrature.StandardQuadrature(
            Quadrature.get_canonical_quadrature_rule(dΩₐ), Geometry.get_num_elements(geo)
        )
        dΩₑ = Quadrature.StandardQuadrature(
            Quadrature.get_canonical_quadrature_rule(dΩₑ), Geometry.get_num_elements(geo)
        )
        # Update exact solution
        δu¹, u¹, f¹ = problem_data(1, geo)
        # Solve problem on current step
        δu¹ₕ, u¹ₕ = solve_one_form_hodge_laplacian(complex[1], complex[2], f¹, dΩₐ, bc_type)
        err_per_element = Analysis.compute_error_per_element(u¹ₕ, u¹, dΩₑ)
        if VERBOSE
            error_δu = Analysis.L2_norm(δu¹ - δu¹ₕ, dΩₑ)
            error_u = Analysis.L2_norm(u¹ - u¹ₕ, dΩₑ)
            @show error_δu, error_u
        end

        if EXPORT
            Plot.export_form_fields_to_vtk(
                (δu¹, δu¹ₕ, u¹, u¹ₕ), "Example-L-chains-$(Lchains)-step-$(step)"
            )
        end
    end

    return δu¹ₕ, u¹ₕ
end

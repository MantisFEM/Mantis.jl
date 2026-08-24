############################################################################################
#                                    Maxwell Eigenvalue                                    #
############################################################################################

"""
    maxwell_eigenvalue(inputs::WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule)

Set up the equations for Maxwell's eigenvalue problem.

Weak form: find ``u^1 \\in H(\\text{curl})\\Lambda^1(\\Omega)`` such that
```math
    \\int d v^1 \\wedge \\star d u^1 = \\int v^1 \\wedge \\star u^1 \\quad \\forall v^1 \\in H(\\text{curl})\\Lambda^1(\\Omega)
```

# Arguments
- `inputs::AbstractInputs`: The inputs for the weak form assembly. See
    [`WeakFormInputs`](@ref) for the details.
- `dΩ::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to use for the integral
    evaluation.

# Returns
- `lhs_expression<:NTuple{1, NTuple{1, AbstractRealValuedOperator}}`: The left-hand side of
    the weak form.
- `rhs_expression<:NTuple{1, NTuple{1, AbstractRealValuedOperator}}`: The right-hand side
    of the weak form.
"""
function maxwell_eigenvalue(
    inputs::WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    v¹ = get_test_form(inputs)
    u¹ = get_trial_form(inputs)
    A = ∫(d(v¹) ∧ ★(d(u¹)), dΩ)
    lhs_expression = ((A,),)
    B = ∫(v¹ ∧ ★(u¹), dΩ)
    rhs_expression = ((B,),)

    return lhs_expression, rhs_expression
end

"""
    analytical_maxwell_eigenfunction(
        m::Int, n::Int, scale_factors::NTuple{2, Float64}, x::Matrix{Float64}
    )

Evaluates the analytical Maxwell eigenfunction for the eigenmode `(m, n)` at points `x`.

# Arguments
- `m::Int`: Eigenmode component x.
- `n::Int`: Eigenmode component y.
- `scale_factors::NTuple{2, Int}`: Scaling factors based on the size of the domain.
- `x::Matrix{Float64}`: Evaluation points.

# Returns
- `NTuple{2, Vector{Float64}}`: The evaluated Maxwell eigenfunction at points `x`.
"""
function analytical_maxwell_eigenfunction(
    m::Int, n::Int, scale_factors::NTuple{2, Float64}, x::Matrix{Float64}
)
    num_points = size(x, 1)
    x_component = Vector{Float64}(undef, num_points)
    y_component = Vector{Float64}(undef, num_points)
    for point in axes(x, 1)
        x_component[point] =
            cos(m * scale_factors[1] * x[point, 1]) *
            sin(n * scale_factors[2] * x[point, 2])
        y_component[point] =
            sin(m * scale_factors[1] * x[point, 1]) *
            cos(n * scale_factors[2] * x[point, 2])
    end

    return [x_component, y_component]
end

"""
    get_maxwell_eig(num_eig::Int, geom::G) where {G <: Geometry.AbstractGeometry{2}}

Returns the first `num_eig` eigenvalues and 1-form eigenfunctions on the geometry `geom`.

# Arguments
- `num_eig::Int`: Number of eigenvalues and eigenfunctions to compute.
- `geom::Geometry.AbstractGeometry{2}`: The two-dimensional geometry.

# Returns
- `Vector{Float64}`: The first `num_eig` analytical eigenvalues.
- `Vector{Forms.AnalyticalFormField{2, 1, G}`: The first `num_eig` analytical
    eigenfunctions.
"""
function get_analytical_maxwell_eig(
    num_eig::Int, geom::G, scale_factors::NTuple{2, Float64}
) where {G <: Geometry.AbstractGeometry{2}}
    eig_vals = Vector{Float64}(undef, (num_eig + 1)^2)
    eig_funcs = Vector{Forms.AnalyticalFormField{2, 1, typeof(geom)}}(
        undef, (num_eig + 1)^2
    )
    eig_count = 1
    for m in 0:num_eig
        for n in 0:num_eig
            curr_val = (scale_factors[1] * m)^2 + (scale_factors[2] * n)^2
            eig_vals[eig_count] = curr_val
            eig_func_expr = x -> analytical_maxwell_eigenfunction(m, n, scale_factors, x)
            eig_funcs[eig_count] = Forms.AnalyticalFormField(1, eig_func_expr, geom, "u")
            eig_count += 1
        end
    end

    sort_inds = sortperm(eig_vals)
    eig_vals = (eig_vals[sort_inds])[2:(num_eig + 1)]
    eig_funcs = (eig_funcs[sort_inds])[2:(num_eig + 1)]

    return eig_vals, eig_funcs
end

"""
    solve_maxwell_eig(
        X⁰::Forms.AbstractFormSpace{2, 0},
        X¹::Forms.AbstractFormSpace{2, 1},
        dΩ::Quadrature.AbstractGlobalQuadratureRule{2},
        num_eig::Int;
        verbose::Bool=false,
    )

Returns the first `num_eig` eigenvalues and 1-form eigenfunctions of the Maxwell eigenvalue
problem.

Weak form: find ``u^1 \\in X^1`` such that
```math
    \\int d v^1 \\wedge \\star d u^1 = \\int v^1 \\wedge \\star u^1 \\quad \\forall v^1 \\in X^1
```

# Arguments
- `X⁰::Forms.AbstractFormSpace{2, 0}`: The 0-form space to compute the nullspace offset
    with.
- `X¹::Forms.AbstractFormSpace{2, 1}`: The 1-form space to use as trial and test space.
- `dΩ::Quadrature.AbstractGlobalQuadratureRule{2}`: The quadrature rule to use for the
    assembly.
- `num_eig::Int`: The number of eigenvalues and eigenfunctions to compute.
- `verbose::Bool=false`: Whether to print the nullspace offset.

# Returns
- `ω²ₕ::Vector{Float64}`: The first `num_eig` eigenvalues.
- `u¹ₕ::Vector{Forms.FormField{2, 1}}`: The first `num_eig` eigenfunctions.
"""
function solve_maxwell_eig(
    X⁰::Forms.AbstractFormSpace{2, 0},
    X¹::Forms.AbstractFormSpace{2, 1},
    dΩ::Quadrature.AbstractGlobalQuadratureRule{2},
    num_eig::Int;
    verbose::Bool=false,
)
    weak_form_inputs = WeakFormInputs(X¹, X¹)
    lhs_expressions, rhs_expressions = maxwell_eigenvalue(weak_form_inputs, dΩ)
    weak_form = WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    bc = Forms.set_dirichlet_boundary_conditions(X¹, 0.0)
    A, B = assemble(weak_form, bc; lhs_type=Matrix{Float64}, rhs_type=Matrix{Float64})
    non_boundary_rows_cols = setdiff(1:Forms.get_num_basis(X¹), keys(bc))
    ωₕ², eig_vecs = LinearAlgebra.eigen(A, B)
    ωₕ² = real.(ωₕ²)
    sort_ids = sortperm(ωₕ²)
    ωₕ² = ωₕ²[sort_ids]
    eig_vecs = eig_vecs[:, sort_ids]
    nullspace_offset = Forms.get_num_basis(X⁰) - length(bc)
    if verbose
        println("""
            The nullspace offset is:
            \t$(nullspace_offset) = dim(X⁰) - (dim(X¹) - dim(x¹ ∩ H₀(curl; Ω)) .
            """)
    end

    ωₕ² = (ωₕ²[(nullspace_offset + 1):end])[1:num_eig]
    u¹ₕ = Vector{Forms.FormField{2, 1}}(undef, num_eig)
    original_label = Forms.get_label(X¹)
    pad = ndigits(num_eig)
    for eig_id in 1:num_eig
        subscript_str = join(Char(0x2080 + d) for d in reverse(digits(eig_id; pad=pad)))
        u¹ₕ[eig_id] = Forms.FormField(
            X¹, zeros(Forms.get_num_basis(X¹)), original_label * subscript_str
        )
        u¹ₕ[eig_id].coefficients[non_boundary_rows_cols] .= real.(
            eig_vecs[:, nullspace_offset + eig_id]
        )
    end

    return ωₕ², u¹ₕ
end

"""
    solve_maxwell_eig(
        complex::C,
        dΩₐ::Quadrature.StandardQuadrature{manifold_dim},
        num_steps::Int,
        dorfler_parameter::Float64,
        dΩₑ::Quadrature.StandardQuadrature{manifold_dim},
        Lchains::Bool,
        eigenfunction::Int,
        num_eig::Int,
        scale_factors::NTuple{manifold_dim, Float64};
        verbose::Bool=false,
    ) where {manifold_dim, num_forms, C <: NTuple{num_forms, Forms.AbstractFormSpace}}

Returns the first `num_eig` eigenvalues and 1-form eigenfunctions of the Maxwell eigenvalue
problem for an adaptive loop.

# Arguments
- `complex::C`: The initial de Rham complex to use for the problem.
- `dΩₐ::Quadrature.StandardQuadrature{manifold_dim}`: The quadrature rule to use for the
    assembly.
- `num_steps::Int`: The number of adaptive steps to perform.
- `dorfler_parameter::Float64`: The Dörfler marking parameter.
- `dΩₑ::Quadrature.StandardQuadrature{manifold_dim}`: The quadrature rule to use for the
    error estimation.
- `Lchains::Bool`: Whether to use L-chains for the refinement.
- `eigenfunction::Int`: The index of the eigenfunction to use for the error estimation.
- `num_eig::Int`: The number of eigenvalues and eigenfunctions to compute.
- `scale_factors::NTuple{manifold_dim, Float64}`: The scaling factors for the geometry.
- `verbose::Bool=false`: Whether to print the progress of the adaptive loop.

# Returns
- `ω²ₕ::Vector{Float64}`: The first `num_eig` eigenvalues.
- `u¹ₕ::Vector{Forms.FormField{manifold_dim, 1, G}}`: The first `num_eig` eigenfunctions.
"""
function solve_maxwell_eig(
    complex::C,
    dΩₐ::Quadrature.StandardQuadrature{manifold_dim},
    num_steps::Int,
    dorfler_parameter::Float64,
    dΩₑ::Quadrature.StandardQuadrature{manifold_dim},
    Lchains::Bool,
    eigenfunction::Int,
    num_eig::Int,
    scale_factors::NTuple{manifold_dim, Float64};
    verbose::Bool=false,
) where {manifold_dim, num_forms, C <: NTuple{num_forms, Forms.AbstractFormSpace}}
    if verbose
        println("Solving the problem on initial step...")
    end

    # Exact solution on initial step
    ω², u¹ = get_analytical_maxwell_eig(
        num_eig, Forms.get_geometry(complex[1]), scale_factors
    )
    ω²ₕ, u¹ₕ = solve_maxwell_eig(complex[1], complex[2], dΩₐ, num_eig)
    err_per_element = Analysis.compute_error_per_element(
        u¹ₕ[eigenfunction], u¹[eigenfunction], dΩₑ
    )
    for step in 1:num_steps
        if verbose
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
        ω², u¹ = get_analytical_maxwell_eig(num_eig, geo, scale_factors)
        ω²ₕ, u¹ₕ = solve_maxwell_eig(complex[1], complex[2], dΩₐ, num_eig; verbose)
        err_per_element = Analysis._compute_square_error_per_element(
            u¹ₕ[eigenfunction], u¹[eigenfunction], dΩₑ
        )
    end

    return ω²ₕ, u¹ₕ
end

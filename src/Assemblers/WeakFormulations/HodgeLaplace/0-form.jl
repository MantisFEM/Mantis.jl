############################################################################################
#                                  0-form Hodge Laplacian                                  #
############################################################################################

"""
    zero_form_hodge_laplacian(
        inputs::AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
    )

Set up the equations for the ``0``-form Hodge-Laplacian. Does not specify any boundary
conditions.

Weak form: for given ``f^0 \\in L^2\\Lambda^0(\\Omega)``, find
``u^0 \\in H^1\\Lambda^0(\\Omega)`` such that
```math
    \\int d v^0 \\wedge \\star d u^0 = \\int v^0 \\wedge \\star f^0 \\quad \\forall v^0 \\in H^1_0\\Lambda^0(\\Omega)
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
function zero_form_hodge_laplacian(
    inputs::AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    v⁰ = Assemblers.get_test_form(inputs)
    u⁰ = Assemblers.get_trial_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)
    A = ∫(d(v⁰) ∧ ★(d(u⁰)), dΩ)
    lhs_expression = ((A,),)
    b = ∫(v⁰ ∧ ★(f⁰), dΩ)
    rhs_expression = ((b,),)

    return lhs_expression, rhs_expression
end

"""
    solve_zero_form_hodge_laplacian(X⁰, fₑ, dΩ)

Returns the solution of the weak form of the 0-form Hodge Laplacian using homogeneous
Dirichlet boundary conditions.

Weak form: for given ``f_e^0 \\in L^2\\Lambda^k(\\Omega)``, find
``u_h^0 \\in X^0_0`` such that
```math
    \\int d v^0 \\wedge \\star d u_h^0 = \\int v^0 \\wedge \\star f_e^0 \\quad \\forall v^0 \\in X^0
```

# Arguments
- `X⁰`: The 0-form space to use as trial and test space.
- `fₑ`: The forcing term to use for the right-hand side of the weak formulation.
- `dΩ`: The quadrature rule to use for the assembly.

# Returns
- `uₕ::Forms.FormField`: The solution of the weak-formulation.
"""
function solve_zero_form_hodge_laplacian(X⁰, fₑ, dΩ)
    weak_form_inputs = WeakFormInputs(X⁰, fₑ)
    lhs_expressions, rhs_expressions = zero_form_hodge_laplacian(weak_form_inputs, dΩ)
    weak_form = WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    # homogeneous boundary conditions
    bc = Forms.set_dirichlet_boundary_conditions(X⁰, 0.0)
    # assemble all matrices
    A, b = assemble(weak_form, bc)
    # solve for coefficients of solution
    sol = vec(A \ b)
    # create the form field from the solution coefficients
    uₕ = Forms.build_form_field(X⁰, sol)

    return uₕ
end

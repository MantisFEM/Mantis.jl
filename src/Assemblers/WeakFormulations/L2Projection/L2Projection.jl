############################################################################################
#                                     Global L2 projection                                 #
############################################################################################

"""
    L2_projection(inputs::AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule)

Set up the L2 projection equations of projecting a function onto a discrete form space.

Weak form: for given ``f^k \\in L^2\\Lambda^k(\\Omega)``, find
``u^k \\in L^2\\Lambda^k(\\Omega)`` such that
```math
    \\int v^k \\wedge \\star u^k = \\int v^k \\wedge \\star f^k \\quad \\forall v^k \\in L^2\\Lambda^k(\\Omega)
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
function L2_projection(inputs::AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule)
    vᵏ = get_test_form(inputs)
    uᵏ = get_trial_form(inputs)
    fᵏ = get_forcing(inputs)
    A = ∫(vᵏ ∧ ★(uᵏ), dΩ)
    lhs_expression = ((A,),)
    b = ∫(vᵏ ∧ ★(fᵏ), dΩ)
    rhs_expression = ((b,),)

    return lhs_expression, rhs_expression
end

"""
    solve_L2_projection(Xᵏ, fₑ, dΩ)

Returns the solution of the weak form of the L2 projection.

Weak form: for given ``f_e^k \\in L^2\\Lambda^k(\\Omega)``, find
``f_h^k \\in X^k`` such that
```math
    \\int v^k \\wedge \\star f_h^k = \\int v^k \\wedge \\star f_e^k \\quad \\forall v^k \\in X^k
```

# Arguments
- `Xᵏ`: The k-form space to use as trial and test space.
- `fₑ`: The forcing term to use for the right-hand side of the weak formulation.
- `dΩ`: The quadrature rule to use for the assembly.

# Returns
- `fₕ::FormField`: The projection of `fₑ` onto `Xᵏ`.
"""
function solve_L2_projection(Xᵏ, fₑ, dΩ)
    weak_form_inputs = WeakFormInputs(Xᵏ, fₑ)
    lhs_expressions, rhs_expressions = L2_projection(weak_form_inputs, dΩ)
    weak_form = WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    A, b = assemble(weak_form)
    sol = vec(A \ b)
    fₕ = Forms.build_form_field(Xᵏ, sol)

    return fₕ
end

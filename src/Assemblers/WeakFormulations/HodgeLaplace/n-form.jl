############################################################################################
#                              n-form Hodge Laplacian                                      #
############################################################################################

"""
    n_form_hodge_laplacian(
        inputs::WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
    )

Function for assembling the weak form of the n-form Hodge Laplacian problem.

# Arguments
- `inputs::WeakFormInputs`: The inputs for the weak form assembly, including test, trial and
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
function n_form_hodge_laplacian(
    inputs::WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ϵ¹, ε² = get_test_forms(inputs)
    u¹, ϕ² = get_trial_forms(inputs)
    f² = get_forcing(inputs)
    A_11 = ∫(ϵ¹ ∧ ★(u¹), dΩ)
    A_12 = -∫(d(ϵ¹) ∧ ★(ϕ²), dΩ)
    A_21 = ∫(ε² ∧ ★(d(u¹)), dΩ)
    lhs_expressions = ((A_11, A_12), (A_21, 0))
    b_21 = ∫(ε² ∧ ★(f²), dΩ)
    rhs_expressions = ((0,), (b_21,))

    return lhs_expressions, rhs_expressions
end

"""
    solve_volume_form_hodge_laplacian(Xⁿ⁻¹, Xⁿ, fₑ, dΩ)

Returns the solution of the weak form of the n-form Hodge Laplacian.

# Arguments
- `Xⁿ⁻¹`: The (n-1)-form space to use as trial and test space.
- `Xⁿ`: The n-form space to use as trial and test space.
- `fₑ`: The forcing term to use for the right-hand side of the weak formulation.
- `dΩ`: The quadrature rule to use for the assembly.

# Returns
- `u¹ₕ`: The (n-1)-form solution of the weak-formulation.
- `ϕ²ₕ`: The n-form solution of the weak-formulation.
"""
function solve_volume_form_hodge_laplacian(Xⁿ⁻¹, Xⁿ, fₑ, dΩ)
    weak_form_inputs = WeakFormInputs((Xⁿ⁻¹, Xⁿ), (fₑ,))
    lhs_expressions, rhs_expressions = n_form_hodge_laplacian(weak_form_inputs, dΩ)
    weak_form = WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    A, b = assemble(weak_form)
    sol = vec(A \ b)
    u¹ₕ, ϕ²ₕ = Forms.build_form_fields((Xⁿ⁻¹, Xⁿ), sol)

    return u¹ₕ, ϕ²ₕ
end

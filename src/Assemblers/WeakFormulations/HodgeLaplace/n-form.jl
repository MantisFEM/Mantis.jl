############################################################################################
#                              n-form Hodge Laplacian                                      #
############################################################################################

"""
    n_form_hodge_laplacian(
        inputs::WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
    )

Set up the equations for the ``n``-form Hodge-Laplacian. Does not specify any boundary
conditions.

Weak form: for given ``f^n \\in L^2\\Lambda^n(\\Omega)``, find
``u^{n-1}, \\phi^n \\in H(\\text{div})\\Lambda^{n-1}(\\Omega) \\times L^2\\Lambda^n(\\Omega)`` such that
```math
\\begin{align}
    \\int \\epsilon^{n-1} \\wedge \\star u^{n-1} - \\int d \\epsilon^{n-1} \\wedge \\star \\phi^n &= 0 \\quad &&\\forall \\epsilon^{n-1} \\in H(\\text{div})\\Lambda^{n-1}(\\Omega) \\\\
    \\int \\epsilon^n \\wedge \\star d u^{n-1} &= \\int \\epsilon^n \\wedge \\star f^n \\quad &&\\forall \\epsilon^n \\in L^2\\Lambda^n(\\Omega)
\\end{align}
```

# Arguments
- `inputs::AbstractInputs`: The inputs for the weak form assembly. See
    [`WeakFormInputs`](@ref) for the details.
- `dΩ::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to use for the integral
    evaluation.

# Returns
- `lhs_expression<:NTuple{2, NTuple{2, Union{Int, AbstractRealValuedOperator}}}`: The
    left-hand side of the weak form.
- `rhs_expression<:NTuple{2, NTuple{1, Union{Int, AbstractRealValuedOperator}}}`: The
    right-hand side of the weak form.
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

Weak form: for given ``f_e^n \\in L^2\\Lambda^n(\\Omega)``, find
``u^{n-1}_h, \\phi^n_h \\in X^{n-1} \\times X^n`` such that
```math
\\begin{align}
    \\int \\epsilon^{n-1}_h \\wedge \\star u^{n-1}_h - \\int d \\epsilon^{n-1}_h \\wedge \\star \\phi^n_h &= 0 \\quad &&\\forall \\epsilon^{n-1}_h \\in X^{n-1} \\\\
    \\int \\epsilon^n_h \\wedge \\star d u^{n-1}_h &= \\int \\epsilon^n_h \\wedge \\star f_e^n \\quad &&\\forall \\epsilon^n_h \\in X^{n}
\\end{align}
```

# Arguments
- `Xⁿ⁻¹`: The (n-1)-form space to use as trial and test space.
- `Xⁿ`: The n-form space to use as trial and test space.
- `fₑ`: The forcing term to use for the right-hand side of the weak formulation.
- `dΩ`: The quadrature rule to use for the assembly.

# Returns
- `uⁿ⁻¹ₕ`: The (n-1)-form solution of the weak-formulation.
- `ϕⁿₕ`: The n-form solution of the weak-formulation.
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

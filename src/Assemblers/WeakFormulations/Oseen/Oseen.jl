############################################################################################
#                                    Oseen                                                 #
############################################################################################

"""
    Oseen(inputs::WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule)

Function for assembling the weak form of the Oseen flow problem.

# Arguments
- `inputs::WeakFormInputs`: The inputs for the weak form assembly, including test and trial
    spaces.
- `dΩ::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to use for the integral
    evaluation.

# Returns
- `lhs_expressions<:NTuple{num_lhs_rows, NTuple{num_lhs_cols, AbstractRealValuedOperator}}`:
    The left-hand side of the weak form, which is a tuple of tuples containing all the blocks
    of the left-hand side matrix.
- `rhs_expressions<:NTuple{num_rhs_rows, NTuple{num_rhs_cols, AbstractRealValuedOperator}}`:
    The right-hand side of the weak form, which is a tuple of tuples containing all the blocks
    of the right-hand side vector.
"""
function Oseen(
    inputs::WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ϵⁿ⁻², ϵⁿ⁻¹, ϵⁿ, θⁿ = get_test_forms(inputs)
    ωⁿ⁻², uⁿ⁻¹, pⁿ, ψⁿ = get_trial_forms(inputs)
    fⁿ⁻¹, pₑ, β¹, κ = get_forcings(inputs)

    A_11 = ∫(ϵⁿ⁻² ∧ ★(ωⁿ⁻²), dΩ)
    A_12 = -∫(d(ϵⁿ⁻²) ∧ ★(uⁿ⁻¹), dΩ)
    A_21 = -(∫(ϵⁿ⁻¹ ∧ ★(β¹ ∧ ωⁿ⁻²), dΩ) + ∫(ϵⁿ⁻¹ ∧ ★(κ ∧ d(ωⁿ⁻²)), dΩ))
    A_23 = ∫(d(ϵⁿ⁻¹) ∧ ★(pⁿ), dΩ)
    A_32 = ∫(ϵⁿ ∧ ★(d(uⁿ⁻¹)), dΩ)
    A_34 = ∫(ϵⁿ ∧ ★(ψⁿ), dΩ)
    A_43 = ∫(θⁿ ∧ ★(pⁿ), dΩ)
    lhs_expressions = ((A_11, A_12, 0, 0), (A_21, 0, A_23, 0), (0, A_32, 0, A_34), (0, 0, A_43, 0))

    b_2 = ∫(ϵⁿ⁻¹ ∧ ★(fⁿ⁻¹), dΩ)
    b_4 = ∫(θⁿ ∧ ★(pₑ), dΩ)
    rhs_expressions = ((0,), (b_2,), (0,), (b_4,))

    return lhs_expressions, rhs_expressions
end

"""
    solve_Oseen(Xⁿ⁻², Xⁿ⁻¹, Xⁿ, Cⁿ, fₑ, pₑ, β, κ, dΩ)

Returns the solution to the Oseen flow problem assuming homogeneous boundary conditions for the normal component of velocity.

# Arguments
- `Xⁿ⁻²::Forms.AbstractFormSpace`: The (n-2)-form space for vorticity.
- `Xⁿ⁻¹::Forms.AbstractFormSpace`: The (n-1)-form space for velocity.
- `Xⁿ::Forms.AbstractFormSpace`: The n-form space for pressure.
- `Cⁿ::Forms.AbstractFormSpace`: The n-form space for the Lagrange multiplier enforcing the pressure average.
- `fₑ::Forms.AbstractFormField`: The forcing term.
- `pₑ::Forms.AbstractFormField`: The exact pressure (only used for computing the average pressure).
- `β::Forms.AbstractFormField`: The advection velocity 1-form (inner oriented).
- `κ::Forms.AbstractFormField`: The viscosity/diffusion 0-form.
- `dΩ::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to use for the assembly.

# Returns
- `ωₕ::Forms.FormField`: The solution for the vorticity.
- `uₕ::Forms.FormField`: The solution for the velocity.
- `pₕ::Forms.FormField`: The solution for the pressure.
- `λₕ::Forms.FormField`: The solution for the Lagrange multiplier.
"""
function solve_Oseen(
    Xⁿ⁻²::Forms.AbstractFormSpace{manifold_dim, rank1},
    Xⁿ⁻¹::Forms.AbstractFormSpace{manifold_dim, rank2},
    Xⁿ::Forms.AbstractFormSpace{manifold_dim, manifold_dim},
    Cⁿ::Forms.ConstantFormSpace{manifold_dim, manifold_dim},
    fₑ::Forms.AbstractFormField{manifold_dim, rank2},
    pₑ::Forms.AbstractFormField{manifold_dim, manifold_dim},
    β::Forms.AbstractFormField{manifold_dim, 1},
    κ::Forms.AbstractFormField{manifold_dim, 0},
    dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
)
    if rank1 != manifold_dim - 2 || rank2 != manifold_dim - 1
        throw(
            ArgumentError(
                "If manifold_dim = $manifold_dim, then rank(X^n-2) must be equal to $(manifold_dim - 2) and rank(X^n-1) must be equal to $(manifold_dim - 1)."
            )
        )
    end

    weak_form_inputs = WeakFormInputs((Xⁿ⁻², Xⁿ⁻¹, Xⁿ, Cⁿ), (fₑ, pₑ, β, κ))
    lhs_expressions, rhs_expressions = Oseen(weak_form_inputs, dΩ)
    weak_form = WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    # display(" assembling...")
    # time1 = time()
    A, b = assemble(weak_form)
    #time2 = time()
    # display(" ...done: $(time2 - time1) seconds")

    # total number of dofs
    n₀, n₁, n₂, n₃ = Forms.get_num_basis.((Xⁿ⁻², Xⁿ⁻¹, Xⁿ, Cⁿ))
    n_dofs = n₀ + n₁ + n₂ + n₃

    # get boundary dof indices for 1-forms (= velocities)
    n₁₁ = FunctionSpaces.get_num_basis(Xⁿ⁻¹.fem_space.component_spaces[1])
    ∂X₁₁ = FunctionSpaces.get_dof_partition(Xⁿ⁻¹.fem_space.component_spaces[1])
    BC₁₁ = vcat([∂X₁₁[1][i] for i in [2,8]]...)
    ∂X₁₂ = FunctionSpaces.get_dof_partition(Xⁿ⁻¹.fem_space.component_spaces[2])
    BC₁₂ = vcat([∂X₁₂[1][i] for i in [4,6]]...)
    # rows and column indices to remove/keep
    remv_inds = n₀ .+ vcat(BC₁₁, BC₁₂ .+ n₁₁)
    keep_inds = setdiff(1:n_dofs, remv_inds)

    # reduced problem
    # display(" reducing problem...")
    # time1 = time()
    A = A[keep_inds, keep_inds]
    b = b[keep_inds]
    # time2 = time()
    # display(" ...done: $(time2 - time1) seconds")
    # display(" done.")

    # solve problem
    sol = zeros(n_dofs)
    # display(" solving...")
    # time1 = time()
    sol[keep_inds] = SparseArrays.qr(A) \ b
    # time2 = time()
    # display(" ...done: $(time2 - time1) seconds")
    # display(" done.")
    ωₕ, uₕ, pₕ, λₕ = Forms.build_form_fields((Xⁿ⁻², Xⁿ⁻¹, Xⁿ, Cⁿ), sol; labels=("ωh","uh","ph", "λh"))

    return ωₕ, uₕ, pₕ, λₕ
end
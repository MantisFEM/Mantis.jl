############################################################################################
#                                     General Scaling                                      #
############################################################################################

# TODO: Rename to `scaling_matrix_approximate`
function scaling_matrix_general(
    parent::FunctionSpaces.AbstractFESpace{manifold_dim, num_components},
    child::FunctionSpaces.AbstractFESpace{manifold_dim, num_components},
    Q::Quadrature.AbstractGlobalQuadratureRule{manifold_dim};
    tol=1e-12,
) where {manifold_dim, num_components}
    form_rank = num_components == manifold_dim ? 1 : 0
    PF = Forms.FormSpace(form_rank, parent, "P")
    CF = Forms.FormSpace(form_rank, child, "C")
    wfi = WeakFormInputs((CF, PF), (CF, PF))
    # Define the weak form
    M_int = ∫(CF ∧ ★(CF), Q)
    P_int = ∫(CF ∧ ★(PF), Q)
    wf = WeakForm(((M_int, P_int), (0, 0)), ((0,), (0,)), wfi)
    F, _ = assemble(wf)

    np = Forms.get_num_basis(PF)
    nc = Forms.get_num_basis(CF)
    M = view(F, 1:nc, 1:nc)
    B = Matrix(view(F, 1:nc, (nc + 1):(nc + np)))
    # WARNING: This assumes the matrix is symmetric positive-definite. Will be true when the
    # basis for `child` is linearly independent.
    Mf = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(M))
    scaling_matrix = Mf \ B
    @inbounds for i in eachindex(scaling_matrix)
        abs(scaling_matrix[i]) < tol && (scaling_matrix[i] = 0.0)
    end

    return scaling_matrix
end

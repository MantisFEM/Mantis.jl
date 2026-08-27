module IntegralInferenceTests

const verbose = false

include("FormsInferenceTestSetup.jl")

for form in forms
    manifold_dim = Forms.get_manifold_dim(form)
    form_rank = Forms.get_form_rank(form)
    if form_rank != manifold_dim
        # We can only construct integrals of top forms.
        continue
    end

    if verbose
        println("New test ---")
        @show nameof(typeof(form))
        @show Forms.get_manifold_dim(form)
        @show Forms.get_form_rank(form)
        @show Forms.get_expression_rank(form)
    end

    element_quad_rule = Quadrature.tensor_product_rule(
        ntuple(i -> 3, manifold_dim), Quadrature.gauss_legendre
    )
    quad_rule = Quadrature.StandardQuadrature(
        element_quad_rule, Forms.get_num_elements(form)
    )

    # Construction
    @test_opt Forms.Integral(form, quad_rule)
    int = Forms.Integral(form, quad_rule)

    # Methods defined for all real valued operators (they can be specialised or throw an
    # error).
    @test_opt Forms.get_manifold_dim(int)
    @test_opt Forms.get_expression_rank(int)
    @test_opt Forms.get_form(int)
    @test_opt Forms.get_form_space_tree(int)
    @test_opt Forms.get_geometry(int)

    @test_opt Forms.get_quadrature_rule(int)
    @test_opt Forms.get_num_elements(int)
    @test_opt Forms.get_num_evaluation_elements(int)
    @test_opt Forms.get_estimated_nnz_per_elem(int)

    # Evaluate.
    if Forms.get_manifold_dim(int) == 1
        @test_opt Forms.evaluate(int, element_id_1D)
    elseif Forms.get_manifold_dim(int) == 2
        @test_opt Forms.evaluate(int, element_id_2D)
    elseif Forms.get_manifold_dim(int) == 3
        @test_opt Forms.evaluate(int, element_id_3D)
    elseif Forms.get_manifold_dim(int) == 4
        @test_opt Forms.evaluate(int, element_id_4D)
    elseif Forms.get_manifold_dim(int) == 5
        @test_opt Forms.evaluate(int, element_id_5D)
    else
        @warn "IntegralInference: The evaluate for this int was not tested: $(int)"
    end
end

end

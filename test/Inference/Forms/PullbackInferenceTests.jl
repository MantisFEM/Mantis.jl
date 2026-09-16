module pullbackInferenceTests

const verbose = false

include("FormsInferenceTestSetup.jl")

for form in forms
    expression_rank = Forms.get_expression_rank(form)

    if verbose
        println("New test ---")
        @show nameof(typeof(form))
        @show Forms.get_manifold_dim(form)
        @show Forms.get_form_rank(form)
        @show Forms.get_expression_rank(form)
    end

    # Construction
    @test_opt Forms.FormPullback(form, Forms.Canonical)
    formpullback = Forms.FormPullback(form, Forms.Canonical)
    @test_opt Forms.ComponentWisePullback(form, Forms.Canonical)
    componentpullback = Forms.ComponentWisePullback(form, Forms.Canonical)

    for pullback in (formpullback, componentpullback)
        # Methods defined for all forms (they can be specialised or throw an error).
        @test_opt Forms.get_manifold_dim(pullback)
        @test_opt Forms.get_form_rank(pullback)
        @test_opt Forms.get_expression_rank(pullback)
        @test_opt Forms.get_label(pullback)
        @test_opt Forms.get_form(pullback)
        @test_opt Forms.get_form_space_tree(pullback)
        @test_opt Forms.get_source_location(pullback)
        @test_opt Forms.get_geometry(pullback)
        @test_opt Forms.get_num_elements(pullback)
        @test_opt Forms.get_estimated_nnz_per_elem(pullback)
        @test_opt Forms.get_fe_space(pullback)

        # Methods defined for all AbstractForms with expression rank 1 (AbstractFormSpaces).
        if Forms.get_expression_rank(pullback) == 1
            @test_opt Forms.get_num_basis(pullback)
            @test_opt Forms.get_num_basis(pullback, element_id)
            @test_opt Forms.get_max_local_dim(pullback)
            @test_opt Forms.get_num_basis_per_expression(pullback, element_id)
        end

        # Evaluate.
        if Forms.get_manifold_dim(pullback) == 1
            @test_opt Forms.evaluate(pullback, element_id_1D, xi_1D)
        elseif Forms.get_manifold_dim(pullback) == 2
            @test_opt Forms.evaluate(pullback, element_id_2D, xi_2D)
        elseif Forms.get_manifold_dim(pullback) == 3
            @test_opt Forms.evaluate(pullback, element_id_3D, xi_3D)
        elseif Forms.get_manifold_dim(pullback) == 4
            @test_opt Forms.evaluate(pullback, element_id_4D, xi_4D)
        elseif Forms.get_manifold_dim(pullback) == 5
            @test_opt Forms.evaluate(pullback, element_id_5D, xi_5D)
        else
            @warn "PullbackInference: The evaluate for this pullback was not tested: $(pullback)"
        end
    end
end

end

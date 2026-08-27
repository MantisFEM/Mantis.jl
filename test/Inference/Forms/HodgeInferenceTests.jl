module HodgeInferenceTests

const verbose = false

include("FormsInferenceTestSetup.jl")

for form in forms
    expression_rank = Forms.get_expression_rank(form)
    if expression_rank >= 2
        # We can't construct the hodge of a form with an expression rank of 2 or higher.
        continue
    end

    if verbose
        println("New test ---")
        @show nameof(typeof(form))
        @show Forms.get_manifold_dim(form)
        @show Forms.get_form_rank(form)
        @show Forms.get_expression_rank(form)
    end

    # Construction
    @test_opt Forms.Hodge(form)
    hodge = Forms.Hodge(form)

    # Methods defined for all forms (they can be specialised or throw an error).
    @test_opt Forms.get_manifold_dim(hodge)
    @test_opt Forms.get_form_rank(hodge)
    @test_opt Forms.get_expression_rank(hodge)
    @test_opt Forms.get_label(hodge)
    @test_opt Forms.get_form(hodge)
    @test_opt Forms.get_form_space_tree(hodge)
    @test_opt Forms.get_geometry(hodge)
    @test_opt Forms.get_num_elements(hodge)
    @test_opt Forms.get_estimated_nnz_per_elem(hodge)
    @test_opt Forms.get_fe_space(hodge)

    # Methods defined for all AbstractForms with expression rank 1 (AbstractFormSpaces).
    if Forms.get_expression_rank(hodge) == 1
        @test_opt Forms.get_num_basis(hodge)
        @test_opt Forms.get_num_basis(hodge, element_id)
        @test_opt Forms.get_max_local_dim(hodge)
        @test_opt Forms.get_num_basis_per_basis(hodge, element_id)
    end

    # Evaluate.
    if Forms.get_manifold_dim(hodge) == 1
        @test_opt Forms.evaluate(hodge, element_id_1D, xi_1D)
    elseif Forms.get_manifold_dim(hodge) == 2
        @test_opt Forms.evaluate(hodge, element_id_2D, xi_2D)
    elseif Forms.get_manifold_dim(hodge) == 3
        @test_opt Forms.evaluate(hodge, element_id_3D, xi_3D)
    elseif Forms.get_manifold_dim(hodge) == 4
        @test_opt Forms.evaluate(hodge, element_id_4D, xi_4D)
    elseif Forms.get_manifold_dim(hodge) == 5
        @test_opt Forms.evaluate(hodge, element_id_5D, xi_5D)
    else
        @warn "HodgeInference: The evaluate for this hodge was not tested: $(hodge)"
    end
end

end

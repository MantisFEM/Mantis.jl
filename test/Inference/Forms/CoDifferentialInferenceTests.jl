module CoDifferentialInferenceTests

const verbose = false

include("FormsInferenceTestSetup.jl")

for form in forms
    manifold_dim = Forms.get_manifold_dim(form)
    form_rank = Forms.get_form_rank(form)
    if form_rank == 0
        # We can't construct the exterior derivative of a top form.
        continue
    end
    # The following two conditions are current limitations only.
    if manifold_dim > 2
        continue
    elseif form_rank > 1
        continue
    end

    if verbose
        @show nameof(typeof(form))
        @show Forms.get_manifold_dim(form)
        @show Forms.get_form_rank(form)
        @show Forms.get_expression_rank(form)
    end

    # Construction
    @test_opt Forms.CoDifferential(form)
    coder = Forms.CoDifferential(form)

    # Methods defined for all forms (they can be specialised or throw an error).
    @test_opt Forms.get_manifold_dim(coder)
    @test_opt Forms.get_form_rank(coder)
    @test_opt Forms.get_expression_rank(coder)
    @test_opt Forms.get_label(coder)
    @test_opt Forms.get_form(coder)
    @test_opt Forms.get_form_space_tree(coder)
    @test_opt Forms.get_geometry(coder)
    @test_opt Forms.get_num_elements(coder)
    @test_opt Forms.get_estimated_nnz_per_elem(coder)
    @test_opt Forms.get_fe_space(coder)

    # Methods defined for all AbstractForms with expression rank 1 (AbstractFormSpaces).
    if Forms.get_expression_rank(coder) == 1
        @test_opt Forms.get_num_basis(coder)
        @test_opt Forms.get_num_basis(coder, element_id)
        @test_opt Forms.get_max_local_dim(coder)
        @test_opt Forms.get_num_basis_per_basis(coder, element_id)
    end

    # Evaluate.
    if Forms.get_manifold_dim(coder) == 1
        @test_opt Forms.evaluate(coder, element_id_1D, xi_1D)
    elseif Forms.get_manifold_dim(coder) == 2
        @test_opt Forms.evaluate(coder, element_id_2D, xi_2D)
    elseif Forms.get_manifold_dim(coder) == 3
        @test_opt Forms.evaluate(coder, element_id_3D, xi_3D)
    elseif Forms.get_manifold_dim(coder) == 4
        @test_opt Forms.evaluate(coder, element_id_4D, xi_4D)
    elseif Forms.get_manifold_dim(coder) == 5
        @test_opt Forms.evaluate(coder, element_id_5D, xi_5D)
    else
        @warn "CodifferentialInference: The evaluate for this coder was not tested: $(coder)"
    end
end

end

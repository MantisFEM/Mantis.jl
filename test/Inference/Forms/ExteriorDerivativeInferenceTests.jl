module ExteriorDerivativeInferenceTests

const verbose = false

include("FormsInferenceTestSetup.jl")

for form in forms
    manifold_dim = Forms.get_manifold_dim(form)
    form_rank = Forms.get_form_rank(form)
    if form_rank == manifold_dim
        # We can't construct the exterior derivative of a top form.
        continue
    end

    if verbose
        @show nameof(typeof(form))
        @show Forms.get_manifold_dim(form)
        @show Forms.get_form_rank(form)
        @show Forms.get_expression_rank(form)
    end

    # Construction
    @test_opt Forms.ExteriorDerivative(form)
    extder = Forms.ExteriorDerivative(form)

    # Methods defined for all forms (they can be specialised or throw an error).
    @test_opt Forms.get_manifold_dim(extder)
    @test_opt Forms.get_form_rank(extder)
    @test_opt Forms.get_expression_rank(extder)
    @test_opt Forms.get_label(extder)
    @test_opt Forms.get_form(extder)
    @test_opt Forms.get_form_space_tree(extder)
    @test_opt Forms.get_geometry(extder)
    @test_opt Forms.get_num_elements(extder)
    @test_opt Forms.get_estimated_nnz_per_elem(extder)
    @test_opt Forms.get_fe_space(extder)

    # Methods defined for all AbstractForms with expression rank 1 (AbstractFormSpaces).
    if Forms.get_expression_rank(extder) == 1
        @test_opt Forms.get_num_basis(extder)
        @test_opt Forms.get_num_basis(extder, element_id)
        @test_opt Forms.get_max_local_dim(extder)
        @test_opt Forms.get_num_basis_per_expression(extder, element_id)
    end

    # Evaluate.
    if Forms.get_manifold_dim(extder) == 1
        @test_opt Forms.evaluate(extder, element_id_1D, xi_1D)
    elseif Forms.get_manifold_dim(extder) == 2
        @test_opt Forms.evaluate(extder, element_id_2D, xi_2D)
    elseif Forms.get_manifold_dim(extder) == 3
        @test_opt Forms.evaluate(extder, element_id_3D, xi_3D)
    elseif Forms.get_manifold_dim(extder) == 4
        @test_opt Forms.evaluate(extder, element_id_4D, xi_4D)
    elseif Forms.get_manifold_dim(extder) == 5
        @test_opt Forms.evaluate(extder, element_id_5D, xi_5D)
    else
        @warn "ExteriorDerivativeInference: The evaluate for this extder was not tested: $(extder)"
    end
end

end

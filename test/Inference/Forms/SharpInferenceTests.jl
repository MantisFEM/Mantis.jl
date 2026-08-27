module SharpInferenceTests

const verbose = false

include("FormsInferenceTestSetup.jl")

for form in forms
    manifold_dim = Forms.get_manifold_dim(form)
    form_rank = Forms.get_form_rank(form)
    expression_rank = Forms.get_expression_rank(form)
    if !(manifold_dim >= 2 && form_rank == 1 && expression_rank == 0)
        # We can only construct the Sharp for 1 forms of fields in 2D or up.
        continue
    end

    if verbose
        println("New test ---")
        @show nameof(typeof(form))
        @show Forms.get_manifold_dim(form)
        @show Forms.get_form_rank(form)
        @show Forms.get_expression_rank(form)
    end

    # The Sharp is not a form itself, so it has far fewer methods than the other operators.

    # Construction
    @test_opt Forms.Sharp(form)
    sharp = Forms.Sharp(form)

    # Methods defined for all forms (they can be specialised or throw an error).
    @test_opt Forms.get_form(sharp)
    @test_opt Forms.get_form_space_tree(sharp)

    # Evaluate.
    if Forms.get_manifold_dim(form) == 1
        @test_opt Forms.evaluate(sharp, element_id_1D, xi_1D)
    elseif Forms.get_manifold_dim(form) == 2
        @test_opt Forms.evaluate(sharp, element_id_2D, xi_2D)
    elseif Forms.get_manifold_dim(form) == 3
        @test_opt Forms.evaluate(sharp, element_id_3D, xi_3D)
    elseif Forms.get_manifold_dim(form) == 4
        @test_opt Forms.evaluate(sharp, element_id_4D, xi_4D)
    elseif Forms.get_manifold_dim(form) == 5
        @test_opt Forms.evaluate(sharp, element_id_5D, xi_5D)
    else
        @warn "SharpInference: The evaluate for this sharp was not tested: $(sharp)"
    end
end

end

module FormsHelpersTests

using Mantis
using Mantis.Hierarchical
using Test

############################################################################################
#                                       Test Helpers                                       #
############################################################################################

get_spaces(complex) = map(Forms.get_fe_space, complex)
get_num_basis(complex) = map(FunctionSpaces.get_num_basis, get_spaces(complex))
get_geometry(complex) = FunctionSpaces.get_parametric_geometry(first(get_spaces(complex)))

function shares_geometry(complex)
    geometry = get_geometry(complex)

    return all(get_spaces(complex)) do space
        return FunctionSpaces.get_geometry(space) === geometry &&
               FunctionSpaces.get_parametric_geometry(space) === geometry
    end
end

# In 1D, the 0-forms and 1-forms of an exact complex differ by the constant functions.
is_exact_1d(complex) = get_num_basis(complex)[1] - get_num_basis(complex)[2] == 1

############################################################################################
#                                          Tests                                           #
############################################################################################

@testset "Hierarchical de Rham Complex" verbose = true begin
    # 4 elements, quadratic B-splines with C¹ regularity, bisected when refined.
    args = ((0.0,), (1.0,), (4,), (2,), (1,), (2,))

    for (S, B) in (
        (FunctionSpaces.SelectionStandard, FunctionSpaces.THB),
        (FunctionSpaces.SelectionStandard, FunctionSpaces.HB),
        (FunctionSpaces.SelectionSimple, FunctionSpaces.THB),
    )
        @testset "$(nameof(S)), $(nameof(B))" verbose = true begin
            complex = Forms.create_hierarchical_de_rham_complex(args..., S, B)
            spaces = get_spaces(complex)
            @test map(Forms.get_form_rank, complex) == (0, 1)
            @test all(space -> space isa FunctionSpaces.HierarchicalSpace, spaces)
            @test all(space -> FunctionSpaces.get_selection_type(space) == S, spaces)
            @test all(space -> FunctionSpaces.get_basis_type(space) == B, spaces)

            #=
                The finest level is empty until refined.

                x inactive
                - active

                Level 1: |-------|-------|-------|-------|
                Level 2: |xxx|xxx|xxx|xxx|xxx|xxx|xxx|xxx|
            =#
            @test shares_geometry(complex)
            @test sort.(get_level_ids(Geometry.get_hierarchy(get_geometry(complex)))) ==
                [[1, 2, 3, 4], Int[]]
            @test get_num_basis(complex) == (6, 5)
            @test is_exact_1d(complex)

            #=
                Refining elements 1 and 2 of level 1 (hierarchical elements 1 and 2).

                Level 1: |xxxxxxx|xxxxxxx|-------|-------|
                Level 2: |---|---|---|---|xxx|xxx|xxx|xxx|
            =#
            complex = Forms.update_hierarchical_de_rham_complex(complex, [1, 2])
            @test shares_geometry(complex)
            @test sort.(get_level_ids(Geometry.get_hierarchy(get_geometry(complex)))) ==
                [[3, 4], [1, 2, 3, 4]]
            @test is_exact_1d(complex)

            #=
                Refining elements 1 and 2 of level 2 (hierarchical elements 3 and 4) creates
                a new level.

                Level 1: |xxxxxxx|xxxxxxx|-------|-------|
                Level 2: |xxx|xxx|---|---|xxx|xxx|xxx|xxx|
                Level 3: |-|-|-|-|x|x|x|x|x|x|x|x|x|x|x|x|
            =#
            complex = Forms.update_hierarchical_de_rham_complex(complex, [3, 4])
            @test shares_geometry(complex)
            @test all(space -> FunctionSpaces.get_num_levels(space) == 3, get_spaces(complex))
            @test sort.(get_level_ids(Geometry.get_hierarchy(get_geometry(complex)))) ==
                [[3, 4], [3, 4], [1, 2, 3, 4]]
            @test is_exact_1d(complex)
        end
    end

    @testset "Invalid Arguments" verbose = true begin
        # TODO: Waiting for multi-component extraction overhaul.
        # The 1-forms in 2D have 2 components.
        @test_throws ArgumentError Forms.create_hierarchical_de_rham_complex(
            (0.0, 0.0),
            (1.0, 1.0),
            (4, 4),
            (2, 2),
            (1, 1),
            (2, 2),
            FunctionSpaces.SelectionStandard,
            FunctionSpaces.THB,
        )
        # The spaces of a tensor-product complex are not hierarchical.
        complex = Forms.create_tensor_product_bspline_de_rham_complex(
            (0.0,), (1.0,), (4,), (2,), (1,)
        )
        @test_throws ArgumentError Forms.update_hierarchical_de_rham_complex(complex, [1])
    end
end

end

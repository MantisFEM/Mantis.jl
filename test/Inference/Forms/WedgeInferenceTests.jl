module WedgeInferenceTests

include("FormsInferenceTestSetup.jl")

# Wedge construction
# wedge_formrank1formrank2_expr.rank1expr.rank2
# 1D
@test_opt Forms.Wedge(zero_form_space_B1_1D, zero_form_space_B1_1D)
@test_opt Forms.Wedge(zero_form_space_B1_1D, zero_form_space_L1_1D)
@test_opt Forms.Wedge(zero_form_space_L1_1D, zero_form_space_B1_1D)
@test_opt Forms.Wedge(zero_form_space_B1_1D, zero_form_field_B1_1D)
@test_opt Forms.Wedge(zero_form_field_B1_1D, zero_form_space_L1_1D)
@test_opt Forms.Wedge(zero_form_field_B1_1D, zero_form_space_B1_1D)
@test_opt Forms.Wedge(zero_form_field_B1_1D, zero_form_field_B1_1D)
@test_opt Forms.Wedge(one_form_space_B2_1D, zero_form_space_B1_1D)
@test_opt Forms.Wedge(one_form_space_B2_1D, zero_form_space_L1_1D)
@test_opt Forms.Wedge(zero_form_space_L1_1D, one_form_space_B2_1D)
@test_opt Forms.Wedge(one_form_space_B2_1D, zero_form_field_B1_1D)
@test_opt Forms.Wedge(one_form_field_B2_1D, zero_form_space_L1_1D)
@test_opt Forms.Wedge(zero_form_field_B1_1D, one_form_space_B2_1D)
@test_opt Forms.Wedge(one_form_field_B2_1D, zero_form_field_B1_1D)
@test_opt Forms.Wedge(zero_form_field_B1_1D, one_form_field_B2_1D)

wedge_00_11_B1B1_1D = Forms.Wedge(zero_form_space_B1_1D, zero_form_space_B1_1D)
wedge_00_11_B1L1_1D = Forms.Wedge(zero_form_space_B1_1D, zero_form_space_L1_1D)
wedge_00_11_L1B1_1D = Forms.Wedge(zero_form_space_L1_1D, zero_form_space_B1_1D)
wedge_00_10_B1B1_1D = Forms.Wedge(zero_form_space_B1_1D, zero_form_field_B1_1D)
wedge_00_01_B1L1_1D = Forms.Wedge(zero_form_field_B1_1D, zero_form_space_L1_1D)
wedge_00_01_B1B1_1D = Forms.Wedge(zero_form_field_B1_1D, zero_form_space_B1_1D)
wedge_00_00_B1B1_1D = Forms.Wedge(zero_form_field_B1_1D, zero_form_field_B1_1D)
wedge_10_11_B2B1_1D = Forms.Wedge(one_form_space_B2_1D, zero_form_space_B1_1D)
wedge_10_11_B2L1_1D = Forms.Wedge(one_form_space_B2_1D, zero_form_space_L1_1D)
wedge_01_11_L1B2_1D = Forms.Wedge(zero_form_space_L1_1D, one_form_space_B2_1D)
wedge_10_10_B2B2_1D = Forms.Wedge(one_form_space_B2_1D, zero_form_field_B1_1D)
wedge_10_01_B2E1_1D = Forms.Wedge(one_form_field_B2_1D, zero_form_space_L1_1D)
wedge_01_01_B2B2_1D = Forms.Wedge(zero_form_field_B1_1D, one_form_space_B2_1D)
wedge_10_00_B2B2_1D = Forms.Wedge(one_form_field_B2_1D, zero_form_field_B1_1D)
wedge_01_00_B1B2_1D = Forms.Wedge(zero_form_field_B1_1D, one_form_field_B2_1D)

# 2D
@test_opt Forms.Wedge(zero_form_space_B1_2D, zero_form_space_B1_2D)
@test_opt Forms.Wedge(zero_form_space_B1_2D, zero_form_space_L1_2D)
@test_opt Forms.Wedge(zero_form_space_L1_2D, zero_form_space_B1_2D)
@test_opt Forms.Wedge(zero_form_space_B1_2D, zero_form_field_B1_2D)
@test_opt Forms.Wedge(zero_form_field_B1_2D, zero_form_space_L1_2D)
@test_opt Forms.Wedge(zero_form_field_B1_2D, zero_form_space_B1_2D)
@test_opt Forms.Wedge(zero_form_field_B1_2D, zero_form_field_B1_2D)
@test_opt Forms.Wedge(one_form_space_BB_2D, one_form_space_BB_2D)
@test_opt Forms.Wedge(one_form_space_BB_2D, one_form_space_LE_2D)
@test_opt Forms.Wedge(one_form_space_LE_2D, one_form_space_BB_2D)
@test_opt Forms.Wedge(one_form_space_BB_2D, one_form_field_BB_2D)
@test_opt Forms.Wedge(one_form_field_BB_2D, one_form_space_LE_2D)
@test_opt Forms.Wedge(one_form_field_BB_2D, one_form_space_BB_2D)
@test_opt Forms.Wedge(one_form_field_BB_2D, one_form_field_BB_2D)

wedge_00_11_B1B1_2D = Forms.Wedge(zero_form_space_B1_2D, zero_form_space_B1_2D)
wedge_00_11_B1L1_2D = Forms.Wedge(zero_form_space_B1_2D, zero_form_space_L1_2D)
wedge_00_11_L1B1_2D = Forms.Wedge(zero_form_space_L1_2D, zero_form_space_B1_2D)
wedge_00_10_B1B1_2D = Forms.Wedge(zero_form_space_B1_2D, zero_form_field_B1_2D)
wedge_00_01_B1L1_2D = Forms.Wedge(zero_form_field_B1_2D, zero_form_space_L1_2D)
wedge_00_01_B1B1_2D = Forms.Wedge(zero_form_field_B1_2D, zero_form_space_B1_2D)
wedge_00_00_B1B1_2D = Forms.Wedge(zero_form_field_B1_2D, zero_form_space_B1_2D)
wedge_11_11_BBBB_2D = Forms.Wedge(one_form_space_BB_2D, one_form_space_BB_2D)
wedge_11_11_BBLE_2D = Forms.Wedge(one_form_space_BB_2D, one_form_space_LE_2D)
wedge_11_11_LEBB_2D = Forms.Wedge(one_form_space_LE_2D, one_form_space_BB_2D)
wedge_11_10_BBBB_2D = Forms.Wedge(one_form_space_BB_2D, one_form_field_BB_2D)
wedge_11_01_BBLE_2D = Forms.Wedge(one_form_field_BB_2D, one_form_space_LE_2D)
wedge_11_01_BBBB_2D = Forms.Wedge(one_form_field_BB_2D, one_form_space_BB_2D)
wedge_11_00_BBBB_2D = Forms.Wedge(one_form_field_BB_2D, one_form_field_BB_2D)

# 3D
@test_opt Forms.Wedge(zero_form_space_B1_3D, zero_form_space_B1_3D)
@test_opt Forms.Wedge(zero_form_space_B1_3D, zero_form_space_L1_3D)
@test_opt Forms.Wedge(zero_form_space_L1_3D, zero_form_space_B1_3D)
@test_opt Forms.Wedge(zero_form_space_B1_3D, zero_form_field_B1_3D)
@test_opt Forms.Wedge(zero_form_field_B1_3D, zero_form_space_L1_3D)
@test_opt Forms.Wedge(zero_form_field_B1_3D, zero_form_space_B1_3D)
@test_opt Forms.Wedge(zero_form_field_B1_3D, zero_form_field_B1_3D)
@test_opt Forms.Wedge(one_form_space_BBB_3D, one_form_space_BBB_3D)
@test_opt Forms.Wedge(one_form_space_BBB_3D, one_form_space_LEL_3D)
@test_opt Forms.Wedge(one_form_space_LEL_3D, one_form_space_BBB_3D)
@test_opt Forms.Wedge(one_form_space_BBB_3D, one_form_field_BBB_3D)
@test_opt Forms.Wedge(one_form_field_BBB_3D, one_form_space_LEL_3D)
@test_opt Forms.Wedge(one_form_field_BBB_3D, one_form_space_BBB_3D)
@test_opt Forms.Wedge(one_form_field_BBB_3D, one_form_field_BBB_3D)
@test_opt Forms.Wedge(two_form_space_BBB_3D_2, one_form_space_BBB_3D)
@test_opt Forms.Wedge(two_form_space_BBB_3D_2, one_form_space_LEL_3D)
@test_opt Forms.Wedge(one_form_space_LEL_3D, two_form_space_BBB_3D_2)
@test_opt Forms.Wedge(two_form_space_BBB_3D_2, one_form_field_BBB_3D)
@test_opt Forms.Wedge(one_form_field_BBB_3D, two_form_space_LEE_3D)
@test_opt Forms.Wedge(one_form_field_BBB_3D, two_form_space_BBB_3D_2)
@test_opt Forms.Wedge(two_form_space_BBB_3D_2, one_form_field_BBB_3D)

wedge_00_11_B1B1_3D = Forms.Wedge(zero_form_space_B1_3D, zero_form_space_B1_3D)
wedge_00_11_B1L1_3D = Forms.Wedge(zero_form_space_B1_3D, zero_form_space_L1_3D)
wedge_00_11_L1B1_3D = Forms.Wedge(zero_form_space_L1_3D, zero_form_space_B1_3D)
wedge_00_10_B1B1_3D = Forms.Wedge(zero_form_space_B1_3D, zero_form_field_B1_3D)
wedge_00_01_B1L1_3D = Forms.Wedge(zero_form_field_B1_3D, zero_form_space_L1_3D)
wedge_00_01_B1B1_3D = Forms.Wedge(zero_form_field_B1_3D, zero_form_space_B1_3D)
wedge_00_00_B1B1_3D = Forms.Wedge(zero_form_field_B1_3D, zero_form_space_B1_3D)
wedge_11_11_BBBBBB_3D = Forms.Wedge(one_form_space_BBB_3D, one_form_space_BBB_3D)
wedge_11_11_BBBLEL_3D = Forms.Wedge(one_form_space_BBB_3D, one_form_space_LEL_3D)
wedge_11_11_LELBBB_3D = Forms.Wedge(one_form_space_LEL_3D, one_form_space_BBB_3D)
wedge_11_10_BBBBBB_3D = Forms.Wedge(one_form_space_BBB_3D, one_form_field_BBB_3D)
wedge_11_01_BBBLEL_3D = Forms.Wedge(one_form_field_BBB_3D, one_form_space_LEL_3D)
wedge_11_01_BBBBBB_3D = Forms.Wedge(one_form_field_BBB_3D, one_form_space_BBB_3D)
wedge_11_00_BBBBBB_3D = Forms.Wedge(one_form_field_BBB_3D, one_form_field_BBB_3D)
wedge_21_11_BB_3D = Forms.Wedge(two_form_space_BBB_3D_2, one_form_space_BBB_3D)
wedge_12_11_BL_3D = Forms.Wedge(one_form_space_BBB_3D, two_form_space_LEE_3D)
wedge_21_11_LB_3D = Forms.Wedge(two_form_space_LEE_3D, one_form_space_BBB_3D)
wedge_21_10_BB_3D = Forms.Wedge(two_form_space_BBB_3D_2, one_form_field_BBB_3D)
wedge_12_01_BL_3D = Forms.Wedge(one_form_field_BBB_3D, two_form_space_LEE_3D)
wedge_12_01_BB_3D = Forms.Wedge(one_form_field_BBB_3D, two_form_space_BBB_3D_2)
wedge_21_00_BB_3D = Forms.Wedge(two_form_field_BBB_3D_2, one_form_field_BBB_3D)

# 4D
@test_opt Forms.Wedge(zero_form_space_BBBB_4D, zero_form_space_BBBB_4D)
@test_opt Forms.Wedge(Forms.d(zero_form_space_BBBB_4D), Forms.d(zero_form_space_BBBB_4D))
@test_opt Forms.Wedge(zero_form_space_BBBB_4D, one_form_space_BBBB_4D)
@test_opt Forms.Wedge(one_form_space_BBBB_4D, zero_form_space_BBBB_4D)
@test_opt Forms.Wedge(one_form_space_BBBB_4D, one_form_space_BBBB_4D)

wedge_00_11_BB_4D = Forms.Wedge(zero_form_space_BBBB_4D, zero_form_space_BBBB_4D)
wedge_01_11_BB_4D = Forms.Wedge(zero_form_space_BBBB_4D, one_form_space_BBBB_4D)
wedge_10_11_BB_4D = Forms.Wedge(one_form_space_BBBB_4D, zero_form_space_BBBB_4D)
wedge_11_11_BB_4D = Forms.Wedge(one_form_space_BBBB_4D, one_form_space_BBBB_4D)
wedge_11_11_dd_4D = Forms.Wedge(
    Forms.d(zero_form_space_BBBB_4D), Forms.d(zero_form_space_BBBB_4D)
)

# 5D
wedge_00_11_BB_5D = Forms.Wedge(zero_form_space_BBBBB_5D, zero_form_space_BBBBB_5D)

# All together now.
wedges = (
    wedge_00_11_B1B1_1D,
    wedge_00_11_B1L1_1D,
    wedge_00_11_L1B1_1D,
    wedge_00_10_B1B1_1D,
    wedge_00_01_B1L1_1D,
    wedge_00_01_B1B1_1D,
    wedge_00_00_B1B1_1D,
    wedge_10_11_B2B1_1D,
    wedge_10_11_B2L1_1D,
    wedge_10_10_B2B2_1D,
    wedge_10_01_B2E1_1D,
    wedge_10_00_B2B2_1D,
    wedge_01_11_L1B2_1D,
    wedge_01_01_B2B2_1D,
    wedge_01_00_B1B2_1D,
    wedge_00_11_B1B1_2D,
    wedge_00_11_B1L1_2D,
    wedge_00_11_L1B1_2D,
    wedge_00_10_B1B1_2D,
    wedge_00_01_B1L1_2D,
    wedge_00_01_B1B1_2D,
    wedge_00_00_B1B1_2D,
    wedge_11_11_BBBB_2D,
    wedge_11_11_BBLE_2D,
    wedge_11_11_LEBB_2D,
    wedge_11_10_BBBB_2D,
    wedge_11_01_BBLE_2D,
    wedge_11_01_BBBB_2D,
    wedge_11_00_BBBB_2D,
    wedge_00_11_B1B1_3D,
    wedge_00_11_B1L1_3D,
    wedge_00_11_L1B1_3D,
    wedge_00_10_B1B1_3D,
    wedge_00_01_B1L1_3D,
    wedge_00_01_B1B1_3D,
    wedge_00_00_B1B1_3D,
    wedge_11_11_BBBBBB_3D,
    wedge_11_11_BBBLEL_3D,
    wedge_11_11_LELBBB_3D,
    wedge_11_10_BBBBBB_3D,
    wedge_11_01_BBBLEL_3D,
    wedge_11_01_BBBBBB_3D,
    wedge_11_00_BBBBBB_3D,
    wedge_21_11_BB_3D,
    wedge_12_11_BL_3D,
    wedge_21_11_LB_3D,
    wedge_21_10_BB_3D,
    wedge_12_01_BL_3D,
    wedge_12_01_BB_3D,
    wedge_21_00_BB_3D,
    wedge_00_11_BB_4D,
    wedge_01_11_BB_4D,
    wedge_10_11_BB_4D,
    wedge_11_11_BB_4D,
    wedge_11_11_dd_4D,
    wedge_00_11_BB_5D,
)

for wedge in wedges

    # Methods defined for all forms (they can be specialised or throw an error).
    @test_opt Forms.get_manifold_dim(wedge)
    @test_opt Forms.get_form_rank(wedge)
    @test_opt Forms.get_expression_rank(wedge)
    @test_opt Forms.get_label(wedge)
    @test_opt Forms.get_form(wedge)
    @test_opt Forms.get_form_space_tree(wedge)
    @test_opt Forms.get_geometry(wedge)
    @test_opt Forms.get_num_elements(wedge)
    @test_opt Forms.get_estimated_nnz_per_elem(wedge)
    @test_opt Forms.get_fe_space(wedge)

    # Methods defined for all AbstractForms with expression rank 1 (AbstractFormSpaces).
    if Forms.get_expression_rank(wedge) == 1
        @test_opt Forms.get_num_basis(wedge)
        @test_opt Forms.get_num_basis(wedge, element_id)
        @test_opt Forms.get_max_local_dim(wedge)
        @test_opt Forms.get_num_basis_per_expression(wedge, element_id)
    end

    # Methods that only exist for the wedge.
    @test_opt Forms.get_forms(wedge)

    # Evaluate.
    if Forms.get_manifold_dim(wedge) == 1
        @test_opt Forms.evaluate(wedge, element_id_1D, xi_1D)
    elseif Forms.get_manifold_dim(wedge) == 2
        @test_opt Forms.evaluate(wedge, element_id_2D, xi_2D)
    elseif Forms.get_manifold_dim(wedge) == 3
        @test_opt Forms.evaluate(wedge, element_id_3D, xi_3D)
    elseif Forms.get_manifold_dim(wedge) == 4
        @test_opt Forms.evaluate(wedge, element_id_4D, xi_4D)
    elseif Forms.get_manifold_dim(wedge) == 5
        @test_opt Forms.evaluate(wedge, element_id_5D, xi_5D)
    else
        @warn "WedgeInference: The evaluate for this wedge was not tested: $(wedge)"
    end
end

end

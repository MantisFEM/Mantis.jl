
using Mantis

using Test
using JET

# Note that JET only uses the types of the inputs, so which numbers we pick here is
# irrelevant.
const xi_1D = Points.TensorProductPoints(([0.0, 1.0],))
const xi_2D = Points.TensorProductPoints(([0.0, 1.0], [0.0, 1.0]))
const xi_3D = Points.TensorProductPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
const xi_4D = Points.TensorProductPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
const xi_5D = Points.TensorProductPoints((
    [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
))

element_id = 2
element_id_1D = 2
element_id_2D = 5
element_id_3D = 15
element_id_4D = 8
element_id_5D = 9

const nodes = Points.get_factor_points(Quadrature.get_nodes(Quadrature.gauss_lobatto(3)))[1]

# Setup ------------------------------------------------------------------------------------
# 1D
B1_1D = FunctionSpaces.create_bspline_space(
    (0.0,), (1.0,), (3,), (FunctionSpaces.Bernstein(2),), (1,)
)
B2_1D = FunctionSpaces.create_bspline_space(
    (0.0,), (1.0,), (3,), (FunctionSpaces.Bernstein(1),), (0,)
)
L1_1D = FunctionSpaces.create_bspline_space(
    (0.0,), (1.0,), (3,), (FunctionSpaces.Lagrange(nodes),), (0,)
)
E1_1D = FunctionSpaces.create_bspline_space(
    (0.0,), (1.0,), (3,), (FunctionSpaces.Edge(nodes),), (-1,)
)
# FormSpaces
zero_form_space_B1_1D = Forms.FormSpace(0, B1_1D, "0S-B1-1D")
zero_form_space_L1_1D = Forms.FormSpace(0, L1_1D, "0S-L1-1D")
one_form_space_B2_1D = Forms.FormSpace(1, B2_1D, "1S-B2-1D")
one_form_space_E1_1D = Forms.FormSpace(1, E1_1D, "1S-E1-1D")
# FormFields
coeffs_1D_B1 = ones(Forms.get_num_basis(zero_form_space_B1_1D))
coeffs_1D_B2 = ones(Forms.get_num_basis(one_form_space_B2_1D))
coeffs_1D_L1 = ones(Forms.get_num_basis(zero_form_space_L1_1D))
coeffs_1D_E1 = ones(Forms.get_num_basis(one_form_space_E1_1D))
zero_form_field_B1_1D = Forms.FormField(zero_form_space_B1_1D, coeffs_1D_B1, "0F-B1-1D")
zero_form_field_L1_1D = Forms.FormField(zero_form_space_L1_1D, coeffs_1D_L1, "0F-L1-1D")
one_form_field_B2_1D = Forms.FormField(one_form_space_B2_1D, coeffs_1D_B2, "1F-B2-1D")
one_form_field_E1_1D = Forms.FormField(one_form_space_E1_1D, coeffs_1D_E1, "1F-B2-1D")

# 2D
B1_2D = FunctionSpaces.create_bspline_space(
    (0.0, 0.0),
    (1.0, 1.0),
    (3, 4),
    (FunctionSpaces.Bernstein(2), FunctionSpaces.Bernstein(2)),
    (1, 1),
)
L1_2D = FunctionSpaces.create_bspline_space(
    (0.0, 0.0),
    (1.0, 1.0),
    (3, 4),
    (FunctionSpaces.Lagrange(nodes), FunctionSpaces.Lagrange(nodes)),
    (0, 0),
)
B12_2D = FunctionSpaces.TensorProductSpace((B1_1D, B2_1D))
B21_2D = FunctionSpaces.TensorProductSpace((B2_1D, B1_1D))
LE_2D = FunctionSpaces.TensorProductSpace((L1_1D, E1_1D))
EL_2D = FunctionSpaces.TensorProductSpace((E1_1D, L1_1D))
DS_BB_2D = FunctionSpaces.DirectSumSpace((B12_2D, B21_2D))
DS_LE_2D = FunctionSpaces.DirectSumSpace((LE_2D, EL_2D))
# FormSpaces
zero_form_space_B1_2D = Forms.FormSpace(0, B1_2D, "0S-B1-2D")
zero_form_space_L1_2D = Forms.FormSpace(0, L1_2D, "0S-L1-2D")
one_form_space_BB_2D = Forms.FormSpace(1, DS_BB_2D, "1S-BB-2D")
one_form_space_LE_2D = Forms.FormSpace(1, DS_LE_2D, "1S-LE-2D")
# FormFields
coeffs_2D = ones(Forms.get_num_basis(zero_form_space_B1_2D))
coeffs_2D_BB = ones(Forms.get_num_basis(one_form_space_BB_2D))
zero_form_field_B1_2D = Forms.FormField(zero_form_space_B1_2D, coeffs_2D, "0F-B1-2D")
one_form_field_BB_2D = Forms.FormField(one_form_space_BB_2D, coeffs_2D_BB, "1F-BB-2D")

# 3D
B1_3D = FunctionSpaces.create_bspline_space(
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (3, 4, 5),
    (FunctionSpaces.Bernstein(2), FunctionSpaces.Bernstein(2), FunctionSpaces.Bernstein(3)),
    (1, 1, 1),
)
L1_3D = FunctionSpaces.create_bspline_space(
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (3, 4, 5),
    (
        FunctionSpaces.Lagrange(nodes),
        FunctionSpaces.Lagrange(nodes),
        FunctionSpaces.Lagrange(nodes),
    ),
    (0, 0, 0),
)
zero_form_space_B1_3D = Forms.FormSpace(0, B1_3D, "zero-form-B1-3D")
zero_form_space_L1_3D = Forms.FormSpace(0, L1_3D, "zero-form-L1-3D")
coeffs_3D = ones(Forms.get_num_basis(zero_form_space_B1_3D))
zero_form_field_B1_3D = Forms.FormField(
    zero_form_space_B1_3D, coeffs_3D, "zero-form-field-B1-3D"
)
B112_3D = FunctionSpaces.TensorProductSpace((B1_1D, B1_1D, B2_1D))
B121_3D = FunctionSpaces.TensorProductSpace((B1_1D, B2_1D, B1_1D))
B211_3D = FunctionSpaces.TensorProductSpace((B2_1D, B1_1D, B1_1D))
LLE_3D = FunctionSpaces.TensorProductSpace((L1_1D, L1_1D, E1_1D))
LEL_3D = FunctionSpaces.TensorProductSpace((L1_1D, E1_1D, L1_1D))
ELL_3D = FunctionSpaces.TensorProductSpace((E1_1D, L1_1D, L1_1D))

DS_BBB_3D = FunctionSpaces.DirectSumSpace((B112_3D, B121_3D, B211_3D))
DS_LEL_3D = FunctionSpaces.DirectSumSpace((LLE_3D, LEL_3D, ELL_3D))

one_form_space_BBB_3D = Forms.FormSpace(1, DS_BBB_3D, "one-form-BBB")
one_form_space_LEL_3D = Forms.FormSpace(1, DS_LEL_3D, "one-form-LEL")
coeffs_3D_BBB = ones(Forms.get_num_basis(one_form_space_BBB_3D))
one_form_field_BBB_3D = Forms.FormField(
    one_form_space_BBB_3D, coeffs_3D_BBB, "one-form-field-BBB"
)

B122_3D = FunctionSpaces.TensorProductSpace((B1_1D, B2_1D, B2_1D))
B221_3D = FunctionSpaces.TensorProductSpace((B2_1D, B2_1D, B1_1D))
B212_3D = FunctionSpaces.TensorProductSpace((B2_1D, B1_1D, B2_1D))
LEE_3D = FunctionSpaces.TensorProductSpace((L1_1D, E1_1D, E1_1D))
EEL_3D = FunctionSpaces.TensorProductSpace((E1_1D, E1_1D, L1_1D))
ELE_3D = FunctionSpaces.TensorProductSpace((E1_1D, L1_1D, E1_1D))

DS_BBB_3D_2 = FunctionSpaces.DirectSumSpace((B122_3D, B212_3D, B221_3D))
DS_LEE_3D = FunctionSpaces.DirectSumSpace((LEE_3D, ELE_3D, EEL_3D))

two_form_space_BBB_3D_2 = Forms.FormSpace(2, DS_BBB_3D_2, "two-form-BBB-2")
two_form_space_LEE_3D = Forms.FormSpace(2, DS_LEE_3D, "two-form-LEE")
coeffs_3D_BBB_2 = ones(Forms.get_num_basis(two_form_space_BBB_3D_2))
two_form_field_BBB_3D_2 = Forms.FormField(
    two_form_space_BBB_3D_2, coeffs_3D_BBB_2, "two-form-field-BBB-2"
)

# 4D
B1111_4D = FunctionSpaces.TensorProductSpace((B1_1D, B1_1D, B1_1D, B1_1D))
B1112_4D = FunctionSpaces.TensorProductSpace((B1_1D, B1_1D, B1_1D, B2_1D))
B1121_4D = FunctionSpaces.TensorProductSpace((B1_1D, B1_1D, B2_1D, B1_1D))
B1211_4D = FunctionSpaces.TensorProductSpace((B1_1D, B2_1D, B1_1D, B1_1D))
B2111_4D = FunctionSpaces.TensorProductSpace((B2_1D, B1_1D, B1_1D, B1_1D))

DS_BBBB_4D = FunctionSpaces.DirectSumSpace((B1112_4D, B1121_4D, B1211_4D, B2111_4D))

zero_form_space_BBBB_4D = Forms.FormSpace(0, B1111_4D, "zero-form-BBBB")
one_form_space_BBBB_4D = Forms.FormSpace(1, DS_BBBB_4D, "one-form-BBBB")

# 5D
B11111_5D = FunctionSpaces.TensorProductSpace((B1_1D, B1_1D, B1_1D, B1_1D, B1_1D))
zero_form_space_BBBBB_5D = Forms.FormSpace(0, B11111_5D, "zero-form-BBBBB")

forms_base = (
    zero_form_space_B1_1D,
    zero_form_space_L1_1D,
    one_form_space_B2_1D,
    one_form_space_E1_1D,
    zero_form_field_B1_1D,
    zero_form_field_L1_1D,
    one_form_field_B2_1D,
    one_form_field_E1_1D,
    zero_form_space_B1_2D,
    zero_form_space_L1_2D,
    one_form_space_BB_2D,
    one_form_space_LE_2D,
    zero_form_field_B1_2D,
    one_form_field_BB_2D,
    zero_form_space_B1_3D,
    zero_form_space_L1_3D,
    zero_form_field_B1_3D,
    one_form_space_BBB_3D,
    one_form_space_LEL_3D,
    one_form_field_BBB_3D,
    two_form_space_BBB_3D_2,
    two_form_space_LEE_3D,
    two_form_field_BBB_3D_2,
    zero_form_space_BBBB_4D,
    one_form_space_BBBB_4D,
    zero_form_space_BBBBB_5D,
)

forms_base_uni_min = (
    -zero_form_space_B1_1D,
    -zero_form_space_L1_1D,
    -one_form_space_B2_1D,
    -one_form_space_E1_1D,
    -zero_form_field_B1_1D,
    -zero_form_field_L1_1D,
    -one_form_field_B2_1D,
    -one_form_field_E1_1D,
    -zero_form_space_B1_2D,
    -zero_form_space_L1_2D,
    -one_form_space_BB_2D,
    -one_form_space_LE_2D,
    -zero_form_field_B1_2D,
    -one_form_field_BB_2D,
    -zero_form_space_B1_3D,
    -zero_form_space_L1_3D,
    -zero_form_field_B1_3D,
    -one_form_space_BBB_3D,
    -one_form_space_LEL_3D,
    -one_form_field_BBB_3D,
    -two_form_space_BBB_3D_2,
    -two_form_space_LEE_3D,
    -two_form_field_BBB_3D_2,
    -zero_form_space_BBBB_4D,
    -one_form_space_BBBB_4D,
    -zero_form_space_BBBBB_5D,
)

forms_base_uni_prod = (
    3.0 * zero_form_space_B1_1D,
    3.0 * zero_form_space_L1_1D,
    3.0 * one_form_space_B2_1D,
    3.0 * one_form_space_E1_1D,
    3.0 * zero_form_field_B1_1D,
    3.0 * zero_form_field_L1_1D,
    3.0 * one_form_field_B2_1D,
    3.0 * one_form_field_E1_1D,
    3.0 * zero_form_space_B1_2D,
    3.0 * zero_form_space_L1_2D,
    3.0 * one_form_space_BB_2D,
    3.0 * one_form_space_LE_2D,
    3.0 * zero_form_field_B1_2D,
    3.0 * one_form_field_BB_2D,
    3.0 * zero_form_space_B1_3D,
    3.0 * zero_form_space_L1_3D,
    3.0 * zero_form_field_B1_3D,
    3.0 * one_form_space_BBB_3D,
    3.0 * one_form_space_LEL_3D,
    3.0 * one_form_field_BBB_3D,
    3.0 * two_form_space_BBB_3D_2,
    3.0 * two_form_space_LEE_3D,
    3.0 * two_form_field_BBB_3D_2,
    3.0 * zero_form_space_BBBB_4D,
    3.0 * one_form_space_BBBB_4D,
    3.0 * zero_form_space_BBBBB_5D,
)

forms_base_bin_min = (f - f for f in forms_base)
forms_base_bin_plus = (f + f for f in forms_base)
forms_base_bin_prod = (f * f for f in forms_base)

forms = (
    forms_base...,
    forms_base_uni_min...,
    forms_base_uni_prod...,
    forms_base_bin_min...,
    forms_base_bin_plus...,
    forms_base_bin_prod...,
)

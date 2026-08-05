module HodgeTests

using Mantis

using Test

using LinearAlgebra
using SparseArrays

############################################################################################
#                                         2D Tests                                         #
############################################################################################

# Domain
Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

# Setup the form spaces
# First the FEM spaces
breakpoints1 = [Lleft, 0.5, Lright]
cart1 = Geometry.CartesianGeometry(breakpoints1)
breakpoints2 = [Lbottom, 0.5, 0.6, Ltop]
cart2 = Geometry.CartesianGeometry(breakpoints2)

# Crazy mesh
c = 0.2
crazy_mapping = Geometry.create_curvilinear_mapping(
    (Lleft, Lbottom), (Lright, Ltop), c
)

# first B-spline patch
deg1 = 2
deg2 = 2
B1 = FunctionSpaces.BSplineSpace(cart1, deg1, [-1, deg1 - 1, -1])
# second B-spline patch
B2 = FunctionSpaces.BSplineSpace(cart2, deg2, [-1, min(deg2 - 1, 1), deg2 - 1, -1])
# tensor-product B-spline patch
tp_space_tp_geo_2d = FunctionSpaces.TensorProductSpace((B1, B2))
tp_space_cart_geo_2d = FunctionSpaces.TensorProductSpace(
    (B1, B2), Geometry.CartesianGeometry
)
tp_space_mapp_geo_2d = FunctionSpaces.TensorProductSpace((B1, B2), crazy_mapping)

# Define the DirectSum spaces to be used to generate the formspaces
# On a Cartesian geometry
fe_space_1_cart = FunctionSpaces.DirectSumSpace((
    tp_space_cart_geo_2d, tp_space_cart_geo_2d
))
cart_fe_complex_2d = (tp_space_cart_geo_2d, fe_space_1_cart, tp_space_cart_geo_2d)
# On a tensor-product geometry
fe_space_1_tp = FunctionSpaces.DirectSumSpace((tp_space_tp_geo_2d, tp_space_tp_geo_2d))
tp_fe_complex_2d = (tp_space_tp_geo_2d, fe_space_1_tp, tp_space_tp_geo_2d)
# On a mapped geometry
fe_space_1_mapp = FunctionSpaces.DirectSumSpace((
    tp_space_mapp_geo_2d, tp_space_mapp_geo_2d
))
mapp_fe_complex_2d = (tp_space_mapp_geo_2d, fe_space_1_mapp, tp_space_mapp_geo_2d)

# Tensor product geometry
cart_geo_2d = Geometry.CartesianGeometry((breakpoints1, breakpoints2))
tp_geo_2d = FunctionSpaces.get_geometry(tp_space_tp_geo_2d)
mapp_geo_2d = Geometry.MappedGeometry(cart_geo_2d, crazy_mapping)

q_rule = Quadrature.tensor_product_rule((deg1 + 1, deg2 + 1), Quadrature.gauss_legendre)

# Test on multiple geometries. Type-wise and content/metric wise.
complexes_2d = [cart_fe_complex_2d, tp_fe_complex_2d, mapp_fe_complex_2d]
geometries_2d = [cart_geo_2d, tp_geo_2d, mapp_geo_2d]
@testset "2D" verbose = true begin
    for i in eachindex(complexes_2d)
        # Create form spaces
        zero_form_space = Forms.FormSpace(0, complexes_2d[i][1], "ν")
        one_form_space = Forms.FormSpace(1, complexes_2d[i][2], "η")
        top_form_space = Forms.FormSpace(2, complexes_2d[i][3], "σ")

        # Generate the form expressions
        α⁰ = Forms.FormField(zero_form_space)
        α⁰.coefficients .= 1.0
        ζ¹ = Forms.FormField(one_form_space)
        ζ¹.coefficients .= 1.0
        constdx = Forms.FormField(one_form_space)
        constdx.coefficients[begin:20] .= 1.0
        constdy = Forms.FormField(one_form_space)
        constdy.coefficients[21:end] .= 1.0
        dα⁰ = Forms.ExteriorDerivative(α⁰)
        γ² = Forms.FormField(top_form_space)
        γ².coefficients .= 1.0
        dζ¹ = Forms.ExteriorDerivative(ζ¹)

        ★α⁰ = Forms.Hodge(α⁰)
        ★ζ¹ = Forms.Hodge(ζ¹)
        ★★ζ¹ = Forms.Hodge(★ζ¹)
        ★γ² = Forms.Hodge(γ²)

        geom = geometries_2d[i]
        @testset "$(string(Base.typename(typeof(geom)).wrapper)[17:end])" begin
            for elem_id in 1:1:Geometry.get_num_elements(geom)
                # Note that we cannot do mixed inner products

                # Tests to see if the integrated metric terms are correctly recovered.
                inv_g, g, det_g = Geometry.inv_metric(
                    geom, elem_id, Quadrature.get_nodes(q_rule)
                )
                inv_g_times_det_g = inv_g .* det_g

                # 0-forms
                # Hodge of a unity 0-form is the volume form and has only 1 component.
                hodge_zero_form_eval, hodge_zero_form_indices = Forms.evaluate(
                    ★α⁰, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(isapprox(hodge_zero_form_eval[1], det_g; atol=1e-12))

                # 1-forms
                # Constant dx form
                hodge_dx_one_form_eval, hodge_dx_one_form_indices = Forms.evaluate(
                    Forms.Hodge(constdx), elem_id, Quadrature.get_nodes(q_rule)
                )
                dx_one_form_eval, dx_one_form_indices = Forms.evaluate(
                    constdx, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_dx_one_form_eval[1],
                        [
                            -inv_g_times_det_g[i][1, 2] * dx_one_form_eval[1][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        hodge_dx_one_form_eval[2],
                        [
                            inv_g_times_det_g[i][1, 1] * dx_one_form_eval[1][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                # Constant dy form
                hodge_dy_one_form_eval, hodge_dy_one_form_indices = Forms.evaluate(
                    Forms.Hodge(constdy), elem_id, Quadrature.get_nodes(q_rule)
                )
                dy_one_form_eval, dy_one_form_indices = Forms.evaluate(
                    constdy, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_dy_one_form_eval[1],
                        [
                            -inv_g_times_det_g[i][2, 2] * dy_one_form_eval[2][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        hodge_dy_one_form_eval[2],
                        [
                            inv_g_times_det_g[i][2, 1] * dy_one_form_eval[2][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                # Constant 1-form
                hodge_1_eval, hodge_1_indices = Forms.evaluate(
                    ★ζ¹, elem_id, Quadrature.get_nodes(q_rule)
                )
                zeta_one_form_eval, zeta_one_form_indices = Forms.evaluate(
                    ζ¹, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_1_eval[1],
                        [
                            -inv_g_times_det_g[i][1, 2] * zeta_one_form_eval[1][i] -
                            inv_g_times_det_g[i][2, 2] .* zeta_one_form_eval[2][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        hodge_1_eval[2],
                        [
                            inv_g_times_det_g[i][1, 1] * zeta_one_form_eval[1][i] +
                            inv_g_times_det_g[i][2, 1] .* zeta_one_form_eval[2][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                # Test if the Hodge-⋆ is the inverse of itself (in 2D with minus sign needed)
                hodge_hodge_1_eval, hodge_hodge_1_indices = Forms.evaluate(
                    ★★ζ¹, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(hodge_hodge_1_eval[1], -zeta_one_form_eval[1]; atol=1e-12)
                )
                @test all(
                    isapprox(hodge_hodge_1_eval[2], -zeta_one_form_eval[2]; atol=1e-12)
                )

                # n-forms
                # Hodge of a unity n-form is a form and has only 1 component.
                hodge_top_eval, hodge_top_indices = Forms.evaluate(
                    ★γ², elem_id, Quadrature.get_nodes(q_rule)
                )
                top_eval, top_indices = Forms.evaluate(
                    γ², elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(isapprox(hodge_top_eval[1], top_eval[1] ./ det_g; atol=1e-12))
            end
        end
    end
end

# -----------------------------------------------------------------------------

############################################################################################
#                                         3D Tests                                         #
############################################################################################

# Setup the geometry

# Domain
Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

# First the FEM spaces
breakpoints = [Lleft, 0.5, Lright]

# 1D "line" geometry
line_geo = Geometry.CartesianGeometry(breakpoints)

# B-spline space in 1D
deg = 2
B = FunctionSpaces.BSplineSpace(line_geo, deg, [-1, deg - 1, -1])

# Tensor-product B-spline patch (reference / tensor-product geometry)
TP_Space_3d_tp = FunctionSpaces.TensorProductSpace((B, B, B))
tp_geo_3d = FunctionSpaces.get_geometry(TP_Space_3d_tp)

# Cartesian 3D geometry
geo_3d_cart = Geometry.CartesianGeometry((breakpoints, breakpoints, breakpoints))

# Crazy tensor product geometry in 2D (auxiliary, same mapping as 2D tests)
tp_geo_2d = Geometry.TensorProductGeometry((line_geo, line_geo))
geo_2d_cart_aux = Geometry.CartesianGeometry((breakpoints, breakpoints))
crazy_geo_2d_cart = Geometry.MappedGeometry(geo_2d_cart_aux, crazy_mapping)

# Crazy mesh 3D (in x and y only, z is straight) as tensor-product of crazy 2D and line
crazy_geo_3d_cart = Geometry.TensorProductGeometry((crazy_geo_2d_cart, line_geo))

# -----------------------------------------------------------------------------
# Function spaces for the three geometries (Cartesian, tensor-product, mapped)
# -----------------------------------------------------------------------------

# For a Cartesian geometry: tensor-product space built directly on geo_3d_cart
TP_Space_3d_cart = FunctionSpaces.TensorProductSpace((B, B, B), Geometry.CartesianGeometry)

# For a "mapped" 3D geometry, we represent it as a tensor-product of the 2D mapped
# geometry (crazy_geo_2d_cart) and the line geometry.  We then define a corresponding
# tensor-product space for forms on that geometry.
mapp_fe_2d = FunctionSpaces.TensorProductSpace((B, B), crazy_mapping)
TP_Space_3d_mapp = FunctionSpaces.TensorProductSpace((mapp_fe_2d, B))

# Multivalued FEMSpaces (DirectSumSpace) for 3D forms on each geometry

# 0-forms: one scalar field
dsTP_0_cart = TP_Space_3d_cart
dsTP_0_tp = TP_Space_3d_tp
dsTP_0_mapp = TP_Space_3d_mapp

# 1-forms: three components
dsTP_1_cart = FunctionSpaces.DirectSumSpace((
    TP_Space_3d_cart, TP_Space_3d_cart, TP_Space_3d_cart
),)
dsTP_1_tp = FunctionSpaces.DirectSumSpace((TP_Space_3d_tp, TP_Space_3d_tp, TP_Space_3d_tp))
dsTP_1_mapp = FunctionSpaces.DirectSumSpace((
    TP_Space_3d_mapp, TP_Space_3d_mapp, TP_Space_3d_mapp
),)

# 2-forms: also three components
dsTP_2_cart = dsTP_1_cart
dsTP_2_tp = dsTP_1_tp
dsTP_2_mapp = dsTP_1_mapp

# 3-forms (top-forms): one scalar field
dsTP_top_cart = dsTP_0_cart
dsTP_top_tp = dsTP_0_tp
dsTP_top_mapp = dsTP_0_mapp

# Complexes: (space for 0‑forms, 1‑forms, 2‑forms, top‑forms), mirroring the 2D pattern
cart_fe_complex_3d = (dsTP_0_cart, dsTP_1_cart, dsTP_2_cart, dsTP_top_cart)
tp_fe_complex_3d = (dsTP_0_tp, dsTP_1_tp, dsTP_2_tp, dsTP_top_tp)
mapp_fe_complex_3d = (dsTP_0_mapp, dsTP_1_mapp, dsTP_2_mapp, dsTP_top_mapp)

# Quadrature rule
q_rule = Quadrature.tensor_product_rule((deg, deg, deg) .+ 1, Quadrature.gauss_legendre)

# Test on multiple geometries, in the same "complex + geometry" style as in 2D
complexes_3d = [cart_fe_complex_3d, tp_fe_complex_3d, mapp_fe_complex_3d]
geometries_3d = [geo_3d_cart, tp_geo_3d, crazy_geo_3d_cart]

@testset "3D" verbose = true begin
    for i in eachindex(complexes_3d)
        complex_3d = complexes_3d[i]
        geom = geometries_3d[i]
        if i < 3
            geom_name = string(Base.typename(typeof(geom)).wrapper)[17:end]
        else
            geom_name = "MappedGeometry ⊗ CartesianGeometry"
        end

        @testset "$(geom_name)" begin
            zero_form_space = Forms.FormSpace(0, complex_3d[1], "ν")
            one_form_space = Forms.FormSpace(1, complex_3d[2], "η")
            two_form_space = Forms.FormSpace(2, complex_3d[3], "μ")
            top_form_space = Forms.FormSpace(3, complex_3d[4], "σ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space)
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space)
            ζ¹.coefficients .= 1.0

            # 1-form: constant but nonzero only for first component
            constdx = Forms.FormField(one_form_space)
            constdx.coefficients[begin:64] .= 1.0

            # 1-form: constant but nonzero only for second component
            constdy = Forms.FormField(one_form_space)
            constdy.coefficients[65:128] .= 1.0

            # 1-form: constant but nonzero only for third component
            constdz = Forms.FormField(one_form_space)
            constdz.coefficients[128:end] .= 1.0

            # 2-form: constant
            ζ² = Forms.FormField(two_form_space)
            ζ².coefficients .= 1.0

            # 2-form: constant but nonzero only for first component
            input_dy_dz = Forms.FormField(two_form_space)
            input_dy_dz.coefficients[begin:64] .= 1.0

            # 2-form: constant but nonzero only for second component
            input_dz_dx = Forms.FormField(two_form_space)
            input_dz_dx.coefficients[65:128] .= 1.0

            # 2-form: constant but nonzero only for third component
            input_dx_dy = Forms.FormField(two_form_space)
            input_dx_dy.coefficients[128:end] .= 1.0

            # top-form: constant
            γ³ = Forms.FormField(top_form_space)
            γ³.coefficients .= 1.0

            # Hodge-⋆ of all forms
            ★α⁰ = Forms.Hodge(α⁰)
            ★ζ¹ = Forms.Hodge(ζ¹)
            ★★ζ¹ = Forms.Hodge(Forms.Hodge(ζ¹))
            ★ζ² = Forms.Hodge(ζ²)
            ★★ζ² = Forms.Hodge(Forms.Hodge(ζ²))
            ★γ³ = Forms.Hodge(γ³)

            for elem_id in 1:Geometry.get_num_elements(geom)
                # Note that we cannot do mixed inner products

                # Tests to see if the integrated metric terms are correctly recovered.
                inv_g, g, det_g = Geometry.inv_metric(
                    geom, elem_id, Quadrature.get_nodes(q_rule)
                )
                inv_g_times_det_g = inv_g .* det_g
                g_div_det_g = g ./ det_g

                # 0-forms
                # Hodge of a unity 0-form is the volume form and has only 1 component.
                hodge_0_form_eval, hodge_alpha_indices = Forms.evaluate(
                    ★α⁰, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(isapprox(hodge_0_form_eval[1], det_g; atol=1e-12))

                # 1-forms
                hodge_1_form_dx_eval, hodge_1_form_indices = Forms.evaluate(
                    Forms.Hodge(constdx), elem_id, Quadrature.get_nodes(q_rule)
                )
                form_dx_eval, form_dx_indices = Forms.evaluate(
                    constdx, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_1_form_dx_eval[1][:, 1],
                        [
                            inv_g_times_det_g[i][1, 1] * form_dx_eval[1][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                hodge_1_form_dy_eval, hodge_1_form_indices = Forms.evaluate(
                    Forms.Hodge(constdy), elem_id, Quadrature.get_nodes(q_rule)
                )
                form_dy_eval, form_dy_indices = Forms.evaluate(
                    constdy, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_1_form_dy_eval[2][:, 1],
                        [
                            inv_g_times_det_g[i][2, 2] * form_dy_eval[2][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                hodge_1_form_dz_eval, hodge_1_form_indices = Forms.evaluate(
                    Forms.Hodge(constdz), elem_id, Quadrature.get_nodes(q_rule)
                )
                form_dz_eval, form_dz_indices = Forms.evaluate(
                    constdz, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_1_form_dz_eval[3][:, 1],
                        [
                            inv_g_times_det_g[i][3, 3] * form_dz_eval[3][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                hodge_1_eval, hodge_1_indices = Forms.evaluate(
                    ★ζ¹, elem_id, Quadrature.get_nodes(q_rule)
                )
                form_zeta_eval, form_zeta_indices = Forms.evaluate(
                    ζ¹, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_1_eval[1],
                        [
                            inv_g_times_det_g[i][1, 1] * form_zeta_eval[1][i] +
                            inv_g_times_det_g[i][1, 2] * form_zeta_eval[2][i] +
                            inv_g_times_det_g[i][1, 3] * form_zeta_eval[3][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        hodge_1_eval[2],
                        [
                            inv_g_times_det_g[i][2, 1] * form_zeta_eval[1][i] +
                            inv_g_times_det_g[i][2, 2] * form_zeta_eval[2][i] +
                            inv_g_times_det_g[i][2, 3] * form_zeta_eval[3][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        hodge_1_eval[3],
                        [
                            inv_g_times_det_g[i][3, 1] * form_zeta_eval[1][i] +
                            inv_g_times_det_g[i][3, 2] * form_zeta_eval[2][i] +
                            inv_g_times_det_g[i][3, 3] * form_zeta_eval[3][i] for
                            i in eachindex(inv_g_times_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                # Finally, test if the Hodge-⋆ is the inverse of itself (in 3D without minus signs needed)
                hodge_hodge_1_eval, hodge_hodge_1_indices = Forms.evaluate(
                    ★★ζ¹, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(isapprox(hodge_hodge_1_eval[1], form_zeta_eval[1]; atol=1e-12))
                @test all(isapprox(hodge_hodge_1_eval[2], form_zeta_eval[2]; atol=1e-12))
                @test all(isapprox(hodge_hodge_1_eval[3], form_zeta_eval[3]; atol=1e-12))

                # 2-forms
                hodge_2_form_dy_dz_eval, hodge_2_form_dy_dz_indices = Forms.evaluate(
                    Forms.Hodge(input_dy_dz), elem_id, Quadrature.get_nodes(q_rule)
                )
                form_dy_dz_eval, form_dy_dz_indices = Forms.evaluate(
                    input_dy_dz, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_2_form_dy_dz_eval[1][:, 1],
                        [
                            g_div_det_g[i][1, 1] * form_dy_dz_eval[1][i] for
                            i in eachindex(g_div_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                hodge_2_form_dz_dx_eval, hodge_2_form_dz_dx_indices = Forms.evaluate(
                    Forms.Hodge(input_dz_dx), elem_id, Quadrature.get_nodes(q_rule)
                )
                form_dz_dx_eval, form_dz_dx_indices = Forms.evaluate(
                    input_dz_dx, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_2_form_dz_dx_eval[2][:, 1],
                        [
                            g_div_det_g[i][2, 2] * form_dz_dx_eval[2][i] for
                            i in eachindex(g_div_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                hodge_2_form_dx_dy_eval, hodge_2_form_dx_dy_indices = Forms.evaluate(
                    Forms.Hodge(input_dx_dy), elem_id, Quadrature.get_nodes(q_rule)
                )
                form_dx_dy_eval, form_dx_dy_indices = Forms.evaluate(
                    input_dx_dy, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_2_form_dx_dy_eval[3][:, 1],
                        [
                            g_div_det_g[i][3, 3] * form_dx_dy_eval[3][i] for
                            i in eachindex(g_div_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                hodge_2_eval, hodge_2_indices = Forms.evaluate(
                    ★ζ², elem_id, Quadrature.get_nodes(q_rule)
                )
                zeta_eval, zeta_indices = Forms.evaluate(
                    ζ², elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(
                        hodge_2_eval[1],
                        [
                            g_div_det_g[i][1, 1] * zeta_eval[1][i] +
                            g_div_det_g[i][1, 2] * zeta_eval[2][i] +
                            g_div_det_g[i][1, 3] * zeta_eval[3][i] for
                            i in eachindex(g_div_det_g)
                        ];
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        hodge_2_eval[2],
                        [
                            g_div_det_g[i][2, 1] * zeta_eval[1][i] +
                            g_div_det_g[i][2, 2] * zeta_eval[2][i] +
                            g_div_det_g[i][2, 3] * zeta_eval[3][i] for
                            i in eachindex(g_div_det_g)
                        ];
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        hodge_2_eval[3],
                        [
                            g_div_det_g[i][3, 1] * zeta_eval[1][i] +
                            g_div_det_g[i][3, 2] * zeta_eval[2][i] +
                            g_div_det_g[i][3, 3] * zeta_eval[3][i] for
                            i in eachindex(g_div_det_g)
                        ];
                        atol=1e-12,
                    ),
                )

                # Finally, test if the Hodge-⋆ is the inverse of itself (in 3D without minus signs needed)
                hodge_hodge_2_eval, hodge_hodge_2_indices = Forms.evaluate(
                    ★★ζ², elem_id, Quadrature.get_nodes(q_rule)
                )
                zeta_eval, zeta_indices = Forms.evaluate(
                    ζ², elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(isapprox(hodge_hodge_2_eval[1], zeta_eval[1]; atol=1e-12))
                @test all(isapprox(hodge_hodge_2_eval[2], zeta_eval[2]; atol=1e-12))
                @test all(isapprox(hodge_hodge_2_eval[3], zeta_eval[3]; atol=1e-12))

                # hodge_eval[1] .= @views (form_eval[1] .* (inv_g[:, 2, 2] .* inv_g[:, 3, 3] - inv_g[:, 2, 3] .* inv_g[:, 3, 2]) .+
                #                          form_eval[2] .* (inv_g[:, 2, 3] .* inv_g[:, 3, 1] - inv_g[:, 2, 1] .* inv_g[:, 3, 3]) .+
                #                          form_eval[3] .* (inv_g[:, 2, 1] .* inv_g[:, 3, 2] - inv_g[:, 2, 2] .* inv_g[:, 3, 1])) .* sqrt_g
                # # Second: (α₁²(g³²g¹³-g³³g¹²) + α₂²(g³³g¹¹-g³¹g¹³) + α₃²(g³¹g¹²-g³²g¹¹))dξ²
                # hodge_eval[2] .= @views (form_eval[1] .* (inv_g[:, 3, 2] .* inv_g[:, 1, 3] - inv_g[:, 3, 3] .* inv_g[:, 1, 2]) .+
                #                          form_eval[2] .* (inv_g[:, 3, 3] .* inv_g[:, 1, 1] - inv_g[:, 3, 1] .* inv_g[:, 1, 3]) .+
                #                          form_eval[3] .* (inv_g[:, 3, 1] .* inv_g[:, 1, 2] - inv_g[:, 3, 2] .* inv_g[:, 1, 1])) .* sqrt_g
                # # Third: (α₁²(g¹²g²³-g¹³g²²) + α₂²(g¹³g²¹-g¹¹g²³) + α₃²(g¹¹g²²-g¹²g²¹))dξ³
                # hodge_eval[3] .= @views (form_eval[1] .* (inv_g[:, 1, 2] .* inv_g[:, 2, 3] - inv_g[:, 1, 3] .* inv_g[:, 2, 2]) .+
                #                          form_eval[2] .* (inv_g[:, 1, 3] .* inv_g[:, 2, 1] - inv_g[:, 1, 1] .* inv_g[:, 2, 3]) .+
                #                          form_eval[3] .* (inv_g[:, 1, 1] .* inv_g[:, 2, 2] - inv_g[:, 1, 2] .* inv_g[:, 2, 1])) .* sqrt_g

                # hodge_1_form_dy_eval, hodge_1_form_indices = Forms.evaluate(Forms.Hodge(constdy), elem_id, Quadrature.get_nodes(q_rule))
                # @test all(isapprox(hodge_1_form_dy_eval[2][:,1], inv_g_times_det_g[:,2,2], atol=1e-12))

                # hodge_1_form_dz_eval, hodge_1_form_indices = Forms.evaluate(Forms.Hodge(constdz), elem_id, Quadrature.get_nodes(q_rule))
                # @test all(isapprox(hodge_1_form_dz_eval[3][:,1], inv_g_times_det_g[:,3,3], atol=1e-12))

                # hodge_1_eval, hodge_1_indices = Forms.evaluate(★ζ¹, elem_id, Quadrature.get_nodes(q_rule))
                # @test all(isapprox(hodge_1_eval[1], inv_g_times_det_g[:, 1, 1] .+ inv_g_times_det_g[:, 1, 2] .+ inv_g_times_det_g[:, 1, 3], atol=1e-12))
                # @test all(isapprox(hodge_1_eval[2], inv_g_times_det_g[:, 2, 1] .+ inv_g_times_det_g[:, 2, 2] .+ inv_g_times_det_g[:, 2, 3], atol=1e-12))
                # @test all(isapprox(hodge_1_eval[3], inv_g_times_det_g[:, 3, 1] .+ inv_g_times_det_g[:, 3, 2] .+ inv_g_times_det_g[:, 3, 3], atol=1e-12))

                # n-forms
                # Hodge of a unity n-form is a form and has only 1 component.
                hodge_top_form_eval, hodge_top_form_indices = Forms.evaluate(
                    ★γ³, elem_id, Quadrature.get_nodes(q_rule)
                )
                top_form_eval, top_form_indices = Forms.evaluate(
                    γ³, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox(hodge_top_form_eval[1], top_form_eval[1] ./ det_g; atol=1e-12)
                )
            end
        end
    end
end

# -----------------------------------------------------------------------------
end

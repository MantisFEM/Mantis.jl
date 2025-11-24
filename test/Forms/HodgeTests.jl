module HodgeTests

import Mantis

using Test

using LinearAlgebra
using SparseArrays

# 2D tests --------------------------------------------------------------------

# Domain
Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

# Setup the form spaces
# First the FEM spaces
breakpoints1 = [Lleft, 0.5, Lright]
patch1 = Mantis.Geometry.CartesianGeometry(breakpoints1)
breakpoints2 = [Lbottom, 0.5, 0.6, Ltop]
patch2 = Mantis.Geometry.CartesianGeometry(breakpoints2)

# first B-spline patch
deg1 = 2
deg2 = 2
B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1 - 1, -1])
# second B-spline patch
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2 - 1, 1), deg2 - 1, -1])
# tensor-product B-spline patch
TP_Space = Mantis.FunctionSpaces.TensorProductSpace((B1, B2))

# Define the DirectSum spaces to be used to generate the formspaces
dsTP_0_form_2d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space,))
dsTP_1_form_2d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space, TP_Space))
dsTP_top_form_2d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space,))

# Then the geometry
# Line 1
line_1_geo = Mantis.Geometry.CartesianGeometry((breakpoints1,))

# Line 2
line_2_geo = Mantis.Geometry.CartesianGeometry((breakpoints2,))

# Tensor product geometry
tensor_prod_geo = Mantis.Geometry.TensorProductGeometry((line_1_geo, line_2_geo))
geo_2d_cart = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2))

# Crazy mesh
crazy_c = 0.2
function mapping_h_test(x::AbstractVector)
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0
    return [
        x[1] + ((Lright - Lleft) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
        x[2] + ((Ltop - Lbottom) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
    ]
end
function dmapping_h_test(x::AbstractVector)
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0
    return [
        1.0+pi * crazy_c * cospi(x1_new) * sinpi(x2_new) ((Lright - Lleft)/(Ltop - Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)
        ((Ltop - Lbottom)/(Lright - Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0+pi * crazy_c * sinpi(x1_new) * cospi(x2_new)
    ]
end

dimension = (2, 2)
crazy_mapping = Mantis.Geometry.Mapping(dimension, mapping_h_test, dmapping_h_test)
geom_crazy = Mantis.Geometry.MappedGeometry(geo_2d_cart, crazy_mapping)

q_rule = Mantis.Quadrature.tensor_product_rule(
    (deg1 + 1, deg2 + 1), Mantis.Quadrature.gauss_legendre
)

# Test on multiple geometries. Type-wise and content/metric wise.
for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
    # Create form spaces
    zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_2d, "ν")
    one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_2d, "η")
    top_form_space = Mantis.Forms.FormSpace(2, geom, dsTP_top_form_2d, "σ")

    # Generate the form expressions
    α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
    α⁰.coefficients .= 1.0
    ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
    ζ¹.coefficients .= 1.0
    constdx = Mantis.Forms.FormField(one_form_space, "ζ")
    constdx.coefficients[begin:20] .= 1.0
    constdy = Mantis.Forms.FormField(one_form_space, "ζ")
    constdy.coefficients[21:end] .= 1.0
    dα⁰ = Mantis.Forms.ExteriorDerivative(α⁰)
    γ² = Mantis.Forms.FormField(top_form_space, "γ")
    γ².coefficients .= 1.0
    dζ¹ = Mantis.Forms.ExteriorDerivative(ζ¹)

    ★α⁰ = Mantis.Forms.Hodge(α⁰)
    ★ζ¹ = Mantis.Forms.Hodge(ζ¹)
    ★★ζ¹ = Mantis.Forms.Hodge(★ζ¹)
    ★γ² = Mantis.Forms.Hodge(γ²)

    for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
        # Note that we cannot do mixed inner products

        # Tests to see if the integrated metric terms are correctly recovered.
        inv_g, g, det_g = Mantis.Geometry.inv_metric(
            geom, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        inv_g_times_det_g = inv_g .* det_g

        # 0-forms
        # Hodge of a unity 0-form is the volume form and has only 1 component.
        hodge_zero_form_eval, hodge_zero_form_indices = Mantis.Forms.evaluate(
            ★α⁰, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(isapprox(hodge_zero_form_eval[1], det_g; atol=1e-12))

        # 1-forms
        # Constant dx form
        hodge_dx_one_form_eval, hodge_dx_one_form_indices = Mantis.Forms.evaluate(
            Mantis.Forms.Hodge(constdx), elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        dx_one_form_eval, dx_one_form_indices = Mantis.Forms.evaluate(
            constdx, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_dx_one_form_eval[1],
                [-inv_g_times_det_g[i][1,2] * dx_one_form_eval[1][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                hodge_dx_one_form_eval[2],
                [inv_g_times_det_g[i][1,1] * dx_one_form_eval[1][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )

        # Constant dy form
        hodge_dy_one_form_eval, hodge_dy_one_form_indices = Mantis.Forms.evaluate(
            Mantis.Forms.Hodge(constdy), elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        dy_one_form_eval, dy_one_form_indices = Mantis.Forms.evaluate(
            constdy, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_dy_one_form_eval[1],
                [-inv_g_times_det_g[i][2,2] * dy_one_form_eval[2][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                hodge_dy_one_form_eval[2],
                [inv_g_times_det_g[i][2,1] * dy_one_form_eval[2][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )

        # Constant 1-form
        hodge_1_eval, hodge_1_indices = Mantis.Forms.evaluate(
            ★ζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        zeta_one_form_eval, zeta_one_form_indices = Mantis.Forms.evaluate(
            ζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_1_eval[1],
                [-inv_g_times_det_g[i][1,2] * zeta_one_form_eval[1][i] - inv_g_times_det_g[i][2,2] .* zeta_one_form_eval[2][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                hodge_1_eval[2],
                [inv_g_times_det_g[i][1,1] * zeta_one_form_eval[1][i] + inv_g_times_det_g[i][2,1] .* zeta_one_form_eval[2][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )

        # Test if the Hodge-⋆ is the inverse of itself (in 2D with minus sign needed)
        hodge_hodge_1_eval, hodge_hodge_1_indices = Mantis.Forms.evaluate(
            ★★ζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(isapprox(hodge_hodge_1_eval[1], -zeta_one_form_eval[1]; atol=1e-12))
        @test all(isapprox(hodge_hodge_1_eval[2], -zeta_one_form_eval[2]; atol=1e-12))

        # n-forms
        # Hodge of a unity n-form is a form and has only 1 component.
        hodge_top_eval, hodge_top_indices = Mantis.Forms.evaluate(
            ★γ², elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        top_eval, top_indices = Mantis.Forms.evaluate(
            γ², elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(isapprox(hodge_top_eval[1], top_eval[1] ./ det_g; atol=1e-12))
    end
end

# -----------------------------------------------------------------------------

# 3D tests --------------------------------------------------------------------

# Setup the geometry

# Domain
Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

# Setup the form spaces

# First the FEM spaces
breakpoints = [Lleft, 0.5, Lright]
patch = Mantis.Geometry.CartesianGeometry(breakpoints)

# first B-spline patch
deg = 2
B = Mantis.FunctionSpaces.BSplineSpace(patch, deg, [-1, deg - 1, -1])

# tensor-product B-spline patch
TP_Space_2d = Mantis.FunctionSpaces.TensorProductSpace((B, B))
TP_Space_3d = Mantis.FunctionSpaces.TensorProductSpace((TP_Space_2d, B))

# Then the geometry
# Line
line_geo = Mantis.Geometry.CartesianGeometry((breakpoints,))
geo_3d_cart = Mantis.Geometry.CartesianGeometry((breakpoints, breakpoints, breakpoints))

# Crazy Tensor product geometry in 2D (auxiliary)
tensor_prod_geo_2d = Mantis.Geometry.TensorProductGeometry((line_geo, line_geo))
geo_2d_cart_aux = Mantis.Geometry.CartesianGeometry((breakpoints, breakpoints))
crazy_geo_2d_cart = Mantis.Geometry.MappedGeometry(geo_2d_cart_aux, crazy_mapping)

# Crazy mesh 3D (in x and y only, z is straight) (this is also a tensor product geometry)
crazy_geo_3d_cart = Mantis.Geometry.TensorProductGeometry((crazy_geo_2d_cart, line_geo))

# Setup the form spaces

# Generate the multivalued FEMSpaces (DirectSumSpace)
dsTP_0_form_3d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_3d,))  # direct sum space
dsTP_1_form_3d = Mantis.FunctionSpaces.DirectSumSpace((
    TP_Space_3d, TP_Space_3d, TP_Space_3d
))  # direct sum space
dsTP_2_form_3d = dsTP_1_form_3d
dsTP_top_form_3d = dsTP_0_form_3d

# Quadrature rule
q_rule = Mantis.Quadrature.tensor_product_rule(
    (deg, deg, deg) .+ 1, Mantis.Quadrature.gauss_legendre
)

# Test on multiple geometries. Type-wise and content/metric wise.
for geom in [geo_3d_cart, crazy_geo_3d_cart]
    # Create form spaces
    zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_3d, "ν")
    one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_3d, "η")
    two_form_space = Mantis.Forms.FormSpace(2, geom, dsTP_2_form_3d, "μ")
    top_form_space = Mantis.Forms.FormSpace(3, geom, dsTP_top_form_3d, "σ")

    # Generate the form expressions
    # 0-form: constant
    α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
    α⁰.coefficients .= 1.0

    # 1-form: constant
    ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
    ζ¹.coefficients .= 1.0

    # 1-form: constant but nonzero only for first component
    constdx = Mantis.Forms.FormField(one_form_space, "ζ")
    constdx.coefficients[begin:64] .= 1.0

    # 1-form: constant but nonzero only for second component
    constdy = Mantis.Forms.FormField(one_form_space, "ζ")
    constdy.coefficients[65:128] .= 1.0

    # 1-form: constant but nonzero only for third component
    constdz = Mantis.Forms.FormField(one_form_space, "ζ")
    constdz.coefficients[128:end] .= 1.0

    # 2-form: constant
    ζ² = Mantis.Forms.FormField(two_form_space, "ζ")
    ζ².coefficients .= 1.0

    # 2-form: constant but nonzero only for first component
    const_dy_dz = Mantis.Forms.FormField(two_form_space, "ζ")
    const_dy_dz.coefficients[begin:64] .= 1.0

    # 2-form: constant but nonzero only for second component
    const_dz_dx = Mantis.Forms.FormField(two_form_space, "ζ")
    const_dz_dx.coefficients[65:128] .= 1.0

    # 2-form: constant but nonzero only for third component
    const_dx_dy = Mantis.Forms.FormField(two_form_space, "ζ")
    const_dx_dy.coefficients[128:end] .= 1.0

    # top-form: constant
    γ³ = Mantis.Forms.FormField(top_form_space, "γ")
    γ³.coefficients .= 1.0

    # Hodge-⋆ of all forms
    ★α⁰ = Mantis.Forms.Hodge(α⁰)
    ★ζ¹ = Mantis.Forms.Hodge(ζ¹)
    ★★ζ¹ = Mantis.Forms.Hodge(Mantis.Forms.Hodge(ζ¹))
    ★ζ² = Mantis.Forms.Hodge(ζ²)
    ★★ζ² = Mantis.Forms.Hodge(Mantis.Forms.Hodge(ζ²))
    ★γ³ = Mantis.Forms.Hodge(γ³)

    for elem_id in 1:Mantis.Geometry.get_num_elements(geom)
        # Note that we cannot do mixed inner products

        # Tests to see if the integrated metric terms are correctly recovered.
        inv_g, g, det_g = Mantis.Geometry.inv_metric(
            geom, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        inv_g_times_det_g = inv_g .* det_g
        g_div_det_g = g ./ det_g

        # 0-forms
        # Hodge of a unity 0-form is the volume form and has only 1 component.
        hodge_0_form_eval, hodge_alpha_indices = Mantis.Forms.evaluate(
            ★α⁰, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(isapprox(hodge_0_form_eval[1], det_g; atol=1e-12))

        # 1-forms
        hodge_1_form_dx_eval, hodge_1_form_indices = Mantis.Forms.evaluate(
            Mantis.Forms.Hodge(constdx), elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        form_dx_eval, form_dx_indices = Mantis.Forms.evaluate(
            constdx, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_1_form_dx_eval[1][:, 1],
                [inv_g_times_det_g[i][1,1] * form_dx_eval[1][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )

        hodge_1_form_dy_eval, hodge_1_form_indices = Mantis.Forms.evaluate(
            Mantis.Forms.Hodge(constdy), elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        form_dy_eval, form_dy_indices = Mantis.Forms.evaluate(
            constdy, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_1_form_dy_eval[2][:, 1],
                [inv_g_times_det_g[i][2,2] * form_dy_eval[2][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )

        hodge_1_form_dz_eval, hodge_1_form_indices = Mantis.Forms.evaluate(
            Mantis.Forms.Hodge(constdz), elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        form_dz_eval, form_dz_indices = Mantis.Forms.evaluate(
            constdz, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_1_form_dz_eval[3][:, 1],
                [inv_g_times_det_g[i][3,3] * form_dz_eval[3][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )

        hodge_1_eval, hodge_1_indices = Mantis.Forms.evaluate(
            ★ζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        form_zeta_eval, form_zeta_indices = Mantis.Forms.evaluate(
            ζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_1_eval[1],
                [inv_g_times_det_g[i][1,1] * form_zeta_eval[1][i] +
                inv_g_times_det_g[i][1,2] * form_zeta_eval[2][i] +
                inv_g_times_det_g[i][1,3] * form_zeta_eval[3][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                hodge_1_eval[2],
                [inv_g_times_det_g[i][2,1] * form_zeta_eval[1][i] +
                inv_g_times_det_g[i][2,2] * form_zeta_eval[2][i] +
                inv_g_times_det_g[i][2,3] * form_zeta_eval[3][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                hodge_1_eval[3],
                [inv_g_times_det_g[i][3,1] * form_zeta_eval[1][i] +
                inv_g_times_det_g[i][3,2] * form_zeta_eval[2][i] +
                inv_g_times_det_g[i][3,3] * form_zeta_eval[3][i] for i in eachindex(inv_g_times_det_g)];
                atol=1e-12,
            ),
        )

        # Finally, test if the Hodge-⋆ is the inverse of itself (in 3D without minus signs needed)
        hodge_hodge_1_eval, hodge_hodge_1_indices = Mantis.Forms.evaluate(
            ★★ζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(isapprox(hodge_hodge_1_eval[1], form_zeta_eval[1]; atol=1e-12))
        @test all(isapprox(hodge_hodge_1_eval[2], form_zeta_eval[2]; atol=1e-12))
        @test all(isapprox(hodge_hodge_1_eval[3], form_zeta_eval[3]; atol=1e-12))

        # 2-forms
        hodge_2_form_dy_dz_eval, hodge_2_form_dy_dz_indices = Mantis.Forms.evaluate(
            Mantis.Forms.Hodge(const_dy_dz), elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        form_dy_dz_eval, form_dy_dz_indices = Mantis.Forms.evaluate(
            const_dy_dz, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_2_form_dy_dz_eval[1][:, 1],
                [g_div_det_g[i][1,1] * form_dy_dz_eval[1][i] for i in eachindex(g_div_det_g)];
                atol=1e-12,
            ),
        )

        hodge_2_form_dz_dx_eval, hodge_2_form_dz_dx_indices = Mantis.Forms.evaluate(
            Mantis.Forms.Hodge(const_dz_dx), elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        form_dz_dx_eval, form_dz_dx_indices = Mantis.Forms.evaluate(
            const_dz_dx, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_2_form_dz_dx_eval[2][:, 1],
                [g_div_det_g[i][2,2] * form_dz_dx_eval[2][i] for i in eachindex(g_div_det_g)];
                atol=1e-12,
            ),
        )

        hodge_2_form_dx_dy_eval, hodge_2_form_dx_dy_indices = Mantis.Forms.evaluate(
            Mantis.Forms.Hodge(const_dx_dy), elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        form_dx_dy_eval, form_dx_dy_indices = Mantis.Forms.evaluate(
            const_dx_dy, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_2_form_dx_dy_eval[3][:, 1],
                [g_div_det_g[i][3,3] * form_dx_dy_eval[3][i] for i in eachindex(g_div_det_g)];
                atol=1e-12,
            ),
        )

        hodge_2_eval, hodge_2_indices = Mantis.Forms.evaluate(
            ★ζ², elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        zeta_eval, zeta_indices = Mantis.Forms.evaluate(
            ζ², elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(
            isapprox(
                hodge_2_eval[1],
                [g_div_det_g[i][1,1] * zeta_eval[1][i] +
                g_div_det_g[i][1,2] * zeta_eval[2][i] +
                g_div_det_g[i][1,3] * zeta_eval[3][i] for i in eachindex(g_div_det_g)];
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                hodge_2_eval[2],
                [g_div_det_g[i][2,1] * zeta_eval[1][i] +
                g_div_det_g[i][2,2] * zeta_eval[2][i] +
                g_div_det_g[i][2,3] * zeta_eval[3][i] for i in eachindex(g_div_det_g)];
                atol=1e-12,
            ),
        )
        @test all(
            isapprox(
                hodge_2_eval[3],
                [g_div_det_g[i][3,1] * zeta_eval[1][i] +
                g_div_det_g[i][3,2] * zeta_eval[2][i] +
                g_div_det_g[i][3,3] * zeta_eval[3][i] for i in eachindex(g_div_det_g)];
                atol=1e-12,
            ),
        )

        # Finally, test if the Hodge-⋆ is the inverse of itself (in 3D without minus signs needed)
        hodge_hodge_2_eval, hodge_hodge_2_indices = Mantis.Forms.evaluate(
            ★★ζ², elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        zeta_eval, zeta_indices = Mantis.Forms.evaluate(
            ζ², elem_id, Mantis.Quadrature.get_nodes(q_rule)
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

        # hodge_1_form_dy_eval, hodge_1_form_indices = Mantis.Forms.evaluate(Mantis.Forms.Hodge(constdy), elem_id, Mantis.Quadrature.get_nodes(q_rule))
        # @test all(isapprox(hodge_1_form_dy_eval[2][:,1], inv_g_times_det_g[:,2,2], atol=1e-12))

        # hodge_1_form_dz_eval, hodge_1_form_indices = Mantis.Forms.evaluate(Mantis.Forms.Hodge(constdz), elem_id, Mantis.Quadrature.get_nodes(q_rule))
        # @test all(isapprox(hodge_1_form_dz_eval[3][:,1], inv_g_times_det_g[:,3,3], atol=1e-12))

        # hodge_1_eval, hodge_1_indices = Mantis.Forms.evaluate(★ζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule))
        # @test all(isapprox(hodge_1_eval[1], inv_g_times_det_g[:, 1, 1] .+ inv_g_times_det_g[:, 1, 2] .+ inv_g_times_det_g[:, 1, 3], atol=1e-12))
        # @test all(isapprox(hodge_1_eval[2], inv_g_times_det_g[:, 2, 1] .+ inv_g_times_det_g[:, 2, 2] .+ inv_g_times_det_g[:, 2, 3], atol=1e-12))
        # @test all(isapprox(hodge_1_eval[3], inv_g_times_det_g[:, 3, 1] .+ inv_g_times_det_g[:, 3, 2] .+ inv_g_times_det_g[:, 3, 3], atol=1e-12))

        # n-forms
        # Hodge of a unity n-form is a form and has only 1 component.
        hodge_top_form_eval, hodge_top_form_indices = Mantis.Forms.evaluate(
            ★γ³, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        top_form_eval, top_form_indices = Mantis.Forms.evaluate(
            γ³, elem_id, Mantis.Quadrature.get_nodes(q_rule)
        )
        @test all(isapprox(hodge_top_form_eval[1], top_form_eval[1] ./ det_g; atol=1e-12))
    end
end

# -----------------------------------------------------------------------------
end

module CodifferentialTests

using Mantis
using Test

import LaTeXStrings

# Constructor, property, and getters and setters tests -------------------------------------
function basic_tests(form, answers)
    @test Forms.get_manifold_dim(form) == answers[1]
    @test Forms.get_form_rank(form) == answers[2]
    @test Forms.get_expression_rank(form) == answers[3]
    @test Forms.get_label(form) == answers[4]
    @test Forms.get_form(form) === answers[5]
    @test Forms.get_form_space_tree(form) === answers[6]

    @test Forms.get_geometry(form) == answers[7]
    @test Forms.get_num_elements(form) == answers[8]

    @test Forms.get_estimated_nnz_per_elem(form) == answers[9]
    @test Forms.get_max_local_dim(form) == answers[10]
    @test Forms.get_fe_space(form) === answers[11]
    @test Forms.get_num_basis(form) == answers[12]
    @test Forms.get_num_basis(form, 1) == answers[13]

    return nothing
end

############################################################################################
#                                         Cartesian geometry                               #
############################################################################################

# 1D ---------------------------------------------------------------------------------------
breakpoints_1D = LinRange(0.0, 1.0, 4)
geometry_1D = Geometry.CartesianGeometry(breakpoints_1D)
canonical_qrule_1D = Quadrature.tensor_product_rule((8,), Quadrature.gauss_legendre)
dΩ_1D = Mantis.Quadrature.StandardQuadrature(
    canonical_qrule_1D, Geometry.get_num_elements(geometry_1D)
)

B54_1D = FunctionSpaces.BSplineSpace(geometry_1D, 5, 4)
B65_1D = FunctionSpaces.BSplineSpace(geometry_1D, 6, 5)

zero_form_space_1D = Forms.FormSpace(0, B54_1D, "B54-0-1D")
one_form_space_1D = Forms.FormSpace(1, B54_1D, "B54-1-1D")

@test_throws ArgumentError Forms.CoDifferential(zero_form_space_1D)

delta_one_form_space_1D = Forms.CoDifferential(one_form_space_1D)
@test isa(delta_one_form_space_1D, Forms.CoDifferential{1, 0, 1})

basic_tests(
    delta_one_form_space_1D,
    (
        1,
        0,
        1,
        "\$\\delta\$(B54-1-1D)",
        one_form_space_1D,
        (one_form_space_1D,),
        geometry_1D,
        3,
        6,
        6,
        B54_1D,
        8,
        6,
    ),
)

# Exact solution to the Biharmonic problem
function exact_sol_func_1D(xx::Matrix{Float64})
    xs = xx[:, 1]
    return [[x^4 for x in xs]]
end
function exact_dsol_func_1D(xx::Matrix{Float64})
    xs = xx[:, 1]
    return [[4*x^3 for x in xs]]
end
function exact_ddsol_func_1D(xx::Matrix{Float64})
    xs = xx[:, 1]
    return [[-12*x^2 for x in xs]]
end
function exact_dddsol_func_1D(xx::Matrix{Float64}) # grad laplacian
    xs = xx[:, 1]
    return [[-24*x for x in xs]]
end
function forcing_function_1D(xx::Matrix{Float64})
    xs = xx[:, 1]
    return [[24 for x in xs]]
end

sol_1D = Forms.AnalyticalFormField(0, exact_sol_func_1D, geometry_1D, "sol")
dsol_1D = Forms.AnalyticalFormField(1, exact_dsol_func_1D, geometry_1D, "dsol")
δdsol_1D = Forms.AnalyticalFormField(0, exact_ddsol_func_1D, geometry_1D, "δdsol_2D")
dδdsol_1D = Forms.AnalyticalFormField(1, exact_dddsol_func_1D, geometry_1D, "dδdsol_2D")
δdδdsol_1D = Forms.AnalyticalFormField(0, forcing_function_1D, geometry_1D, "δdδdsol_2D")

# Project the exact polynomial solution. The representation should be exact.
sol_1D_h = Assemblers.solve_L2_projection(zero_form_space_1D, sol_1D, dΩ_1D)
dsol_1D_h = Forms.ExteriorDerivative(sol_1D_h)
δdsol_1D_h = Forms.CoDifferential(dsol_1D_h)
# Also start from the laplacian, because we cannot directly compute the 4-th derivative.
δdsol_1D_hp = Assemblers.solve_L2_projection(zero_form_space_1D, δdsol_1D, dΩ_1D)
dδdsol_1D_h = Forms.ExteriorDerivative(δdsol_1D_hp)
δdδdsol_1D_h = Forms.CoDifferential(dδdsol_1D_h)
# We repeat this starting from the 1-forms, since the CoDifferential is specialised when
# computing a Laplacian.
dsol_1D_hp = Assemblers.solve_L2_projection(one_form_space_1D, dsol_1D, dΩ_1D)
δdsol_1D_h2 = Forms.CoDifferential(dsol_1D_hp)
# Also start from the laplacian, because we cannot directly compute the 4-th derivative.
dδdsol_1D_hp = Assemblers.solve_L2_projection(one_form_space_1D, dδdsol_1D, dΩ_1D)
δdδdsol_1D_h2 = Forms.CoDifferential(dδdsol_1D_hp)

test_δdsol_1D_h = true
test_δdδdsol_1D_h = true
test_δdsol_1D_h2 = true
test_δdδdsol_1D_h2 = true
xi_1d = Quadrature.get_nodes(Quadrature.get_canonical_quadrature_rule(dΩ_1D))
for element_id in 1:Geometry.get_num_elements(geometry_1D)
    values, _ = Forms.evaluate(δdsol_1D_h, element_id, xi_1d)
    values_e, _ = Forms.evaluate(δdsol_1D, element_id, xi_1d)

    values2, _ = Forms.evaluate(δdδdsol_1D_h, element_id, xi_1d)
    values2_e, _ = Forms.evaluate(δdδdsol_1D, element_id, xi_1d)

    values3, _ = Forms.evaluate(δdsol_1D_h2, element_id, xi_1d)
    values3_e, _ = Forms.evaluate(δdsol_1D, element_id, xi_1d)

    values4, _ = Forms.evaluate(δdδdsol_1D_h2, element_id, xi_1d)
    values4_e, _ = Forms.evaluate(δdδdsol_1D, element_id, xi_1d)

    for point in eachindex(values[1], values2[1], values3[1], values4[1])
        if !isapprox(values[1][point], values_e[1][point]; rtol=1e-10, atol=1e-10)
            global test_δdsol_1D_h = false
        end
        if !isapprox(values2[1][point], values2_e[1][point]; rtol=1e-10, atol=1e-10)
            global test_δdδdsol_1D_h = false
        end
        if !isapprox(values3[1][point], values3_e[1][point]; rtol=1e-10, atol=1e-10)
            global test_δdsol_1D_h2 = false
        end
        if !isapprox(values4[1][point], values4_e[1][point]; rtol=1e-10, atol=1e-10)
            global test_δdδdsol_1D_h2 = false
        end
    end
end
@test test_δdsol_1D_h
@test test_δdδdsol_1D_h
@test test_δdsol_1D_h2
@test test_δdδdsol_1D_h2

# 2D ---------------------------------------------------------------------------------------
TP5454_2D = FunctionSpaces.TensorProductSpace((B54_1D, B54_1D))
TP5465_2D = FunctionSpaces.TensorProductSpace((B54_1D, B65_1D))
DS_2D = FunctionSpaces.DirectSumSpace((TP5454_2D, TP5465_2D))
geometry_2D = FunctionSpaces.get_geometry(TP5465_2D)
canonical_qrule_2D = Quadrature.tensor_product_rule((7, 7), Quadrature.gauss_lobatto)
dΩ_2D = Mantis.Quadrature.StandardQuadrature(
    canonical_qrule_2D, Geometry.get_num_elements(geometry_2D)
)

zero_form_space_2D = Forms.FormSpace(0, TP5454_2D, "TP5454-0-2D")
one_form_space_2D = Forms.FormSpace(1, DS_2D, "DS-1-2D")
two_form_space_2D = Forms.FormSpace(2, TP5465_2D, "TP5465-2-2D")

@test_throws ArgumentError Forms.CoDifferential(zero_form_space_2D)
@test_throws MethodError Forms.CoDifferential(two_form_space_2D)

delta_one_form_space_2D = Forms.CoDifferential(one_form_space_2D)
@test isa(delta_one_form_space_2D, Forms.CoDifferential{2, 0, 1})

basic_tests(
    delta_one_form_space_2D,
    (
        2,
        0,
        1,
        "\$\\delta\$(DS-1-2D)",
        one_form_space_2D,
        (one_form_space_2D,),
        geometry_2D,
        9,
        78,
        78,
        DS_2D,
        136,
        78,
    ),
)

# Exact solution to the Biharmonic problem
function exact_sol_func_2D(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [[x^4*y^4 for (x, y) in zip(xs, ys)]]
end
function exact_dsol_func_2D(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [[4*x^3*y^4 for (x, y) in zip(xs, ys)], [4*x^4*y^3 for (x, y) in zip(xs, ys)]]
end
function exact_ddsol_func_2D(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [[-12*x^4*y^2 - 12*x^2*y^4 for (x, y) in zip(xs, ys)]]
end
function exact_dddsol_func_2D(xx::Matrix{Float64}) # grad laplacian
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [
        [-24*x*y^2*(2*x^2 + y^2) for (x, y) in zip(xs, ys)],
        [-24*x^2*y*(x^2 + 2*y^2) for (x, y) in zip(xs, ys)],
    ]
end
function forcing_function_2D(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    return [[24*x^4 + 288*x^2*y^2 + 24*y^4 for (x, y) in zip(xs, ys)]]
end

sol_2D = Forms.AnalyticalFormField(0, exact_sol_func_2D, geometry_2D, "sol")
dsol_2D = Forms.AnalyticalFormField(1, exact_dsol_func_2D, geometry_2D, "dsol")
δdsol_2D = Forms.AnalyticalFormField(0, exact_ddsol_func_2D, geometry_2D, "δdsol_2D")
dδdsol_2D = Forms.AnalyticalFormField(1, exact_dddsol_func_2D, geometry_2D, "dδdsol_2D")
δdδdsol_2D = Forms.AnalyticalFormField(0, forcing_function_2D, geometry_2D, "δdδdsol_2D")

# Project the exact polynomial solution. The representation should be exact.
sol_2D_h = Assemblers.solve_L2_projection(zero_form_space_2D, sol_2D, dΩ_2D)
dsol_2D_h = Forms.ExteriorDerivative(sol_2D_h)
δdsol_2D_h = Forms.CoDifferential(dsol_2D_h)
# Also start from the laplacian, because we cannot directly compute the 4-th derivative.
δdsol_2D_hp = Assemblers.solve_L2_projection(zero_form_space_2D, δdsol_2D, dΩ_2D)
dδdsol_2D_h = Forms.ExteriorDerivative(δdsol_2D_hp)
δdδdsol_2D_h = Forms.CoDifferential(dδdsol_2D_h)
# We repeat this starting from the 1-forms, since the CoDifferential is specialised when
# computing a Laplacian.
dsol_2D_hp = Assemblers.solve_L2_projection(one_form_space_2D, dsol_2D, dΩ_2D)
δdsol_2D_h2 = Forms.CoDifferential(dsol_2D_hp)
# Also start from the laplacian, because we cannot directly compute the 4-th derivative.
dδdsol_2D_hp = Assemblers.solve_L2_projection(one_form_space_2D, dδdsol_2D, dΩ_2D)
δdδdsol_2D_h2 = Forms.CoDifferential(dδdsol_2D_hp)

test_δdsol_2D_h = true
test_δdδdsol_2D_h = true
test_δdsol_2D_h2 = true
test_δdδdsol_2D_h2 = true
xi_2d = Quadrature.get_nodes(Quadrature.get_canonical_quadrature_rule(dΩ_2D))
for element_id in 1:Geometry.get_num_elements(geometry_2D)
    values, _ = Forms.evaluate(δdsol_2D_h, element_id, xi_2d)
    values_e, _ = Forms.evaluate(δdsol_2D, element_id, xi_2d)

    values2, _ = Forms.evaluate(δdδdsol_2D_h, element_id, xi_2d)
    values2_e, _ = Forms.evaluate(δdδdsol_2D, element_id, xi_2d)

    values3, _ = Forms.evaluate(δdsol_2D_h2, element_id, xi_2d)
    values3_e, _ = Forms.evaluate(δdsol_2D, element_id, xi_2d)

    values4, _ = Forms.evaluate(δdδdsol_2D_h2, element_id, xi_2d)
    values4_e, _ = Forms.evaluate(δdδdsol_2D, element_id, xi_2d)

    for point in eachindex(values3[1])
        if !isapprox(values[1][point], values_e[1][point]; rtol=1e-10, atol=1e-10)
            global test_δdsol_2D_h = false
        end
        if !isapprox(values2[1][point], values2_e[1][point]; rtol=1e-10, atol=1e-9)
            global test_δdδdsol_2D_h = false
        end
        if !isapprox(values3[1][point], values3_e[1][point]; rtol=1e-10, atol=1e-10)
            global test_δdsol_2D_h2 = false
        end
        if !isapprox(values4[1][point], values4_e[1][point]; rtol=1e-10, atol=1e-10)
            global test_δdδdsol_2D_h2 = false
        end
    end
end
@test test_δdsol_2D_h
@test test_δdδdsol_2D_h
@test test_δdsol_2D_h2
@test test_δdδdsol_2D_h2

@test_throws MethodError Forms._evaluate_codifferential(two_form_space_2D, 1, xi_2d)

# 3D ---------------------------------------------------------------------------------------
TP545454_3D = FunctionSpaces.TensorProductSpace((B54_1D, B54_1D, B54_1D))
TP546554_3D = FunctionSpaces.TensorProductSpace((B54_1D, B65_1D, B54_1D))
DS_3D = FunctionSpaces.DirectSumSpace((TP546554_3D, TP545454_3D, TP546554_3D))
geometry_3D = FunctionSpaces.get_geometry(TP546554_3D)
canonical_qrule_3D = Quadrature.tensor_product_rule((8, 8, 8), Quadrature.gauss_legendre)
dΩ_3D = Mantis.Quadrature.StandardQuadrature(
    canonical_qrule_3D, Geometry.get_num_elements(geometry_3D)
)

zero_form_space_3D = Forms.FormSpace(0, TP545454_3D, "TP545454-0-3D")
one_form_space_3D = Forms.FormSpace(1, DS_3D, "DS-1-3D")
two_form_space_3D = Forms.FormSpace(2, DS_3D, "DS-2-3D")
three_form_space_3D = Forms.FormSpace(3, TP546554_3D, "TP546554-3-3D")

@test_throws ArgumentError Forms.CoDifferential(zero_form_space_3D)
@test_throws MethodError Forms.CoDifferential(one_form_space_3D)
@test_throws MethodError Forms.CoDifferential(two_form_space_3D)
@test_throws MethodError Forms.CoDifferential(three_form_space_3D)

end

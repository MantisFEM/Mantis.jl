module ExteriorDerivativeTests

using Mantis

using Test

using LinearAlgebra, SparseArrays
import Random

# Common mappings -------------------------------------------------------------

# Crazy mapping
const Lleft = 0.0
const Lright = 1.0
const Lbottom = 0.0
const Ltop = 1.0
const c = 0.2

const crazy_mapping = Geometry.create_curvilinear_mapping(
    (Lleft, Lbottom), (Lright - Lleft, Ltop - Lbottom), c
)

# 'parametric' geometries
const breakpoints = LinRange(Lleft, Lright, 3)
const geo_cart_1 = Geometry.CartesianGeometry(breakpoints)
const geo_cart_2d = Geometry.CartesianGeometry((breakpoints, breakpoints))
const breakpoints2 = [Lbottom, 0.5, 0.6, Ltop]
const geo_cart_2 = Geometry.CartesianGeometry(breakpoints2)
const deg = 2

# 2D tests --------------------------------------------------------------------
# Setup
const B1 = FunctionSpaces.BSplineSpace(geo_cart_1, deg, [-1, deg - 1, -1])
const B2 = FunctionSpaces.BSplineSpace(geo_cart_2, deg, [-1, min(deg - 1, 1), deg - 1, -1])

# Tensor-product B-spline spaces
const TP_Space_2d_tp_geo = FunctionSpaces.TensorProductSpace((B1, B2))
const TP_Space_2d_cart_geo = FunctionSpaces.TensorProductSpace(
    (B1, B2),
    Geometry.CartesianGeometry,  # Convert the geometry to a CartesianGeometry.
)
const TP_Space_2d_crazy_geo = FunctionSpaces.TensorProductSpace(
    (B1, B2), Geometry.CartesianGeometry, crazy_mapping
)

const q_rule = Quadrature.tensor_product_rule((deg + 1, deg + 1), Quadrature.gauss_legendre)

const spaces_2D = (TP_Space_2d_cart_geo, TP_Space_2d_tp_geo, TP_Space_2d_crazy_geo)
@testset "2D" verbose = true begin
    @testset "Error barriers" begin
        for space in spaces_2D
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")
            top_form_space = Forms.FormSpace(2, space, "σ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space)
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space)
            ζ¹.coefficients .= 1.0

            β¹ = Forms.FormField(one_form_space)
            β¹.coefficients .= 1.0

            # Compute exterior derivatives
            @test_throws ArgumentError (d(zero_form_space) ∧ one_form_space) +
                (zero_form_space ∧ top_form_space)

            # Check if compatible spaces error is NOT thrown
            @test begin
                (α⁰ ∧ zero_form_space ∧ one_form_space) + (zero_form_space ∧ one_form_space)
                true
            end

            # Check if incompatible form_rank error is thrown (no method should exist)
            @test_throws MethodError (d(α⁰) ∧ zero_form_space ∧ one_form_space) +
                (zero_form_space ∧ one_form_space)
            @test_throws MethodError d(α⁰ ∧ one_form_space) +
                (zero_form_space ∧ top_form_space)
            @test_throws MethodError (d(α⁰) ∧ one_form_space) +
                (d(zero_form_space) ∧ one_form_space)
        end
    end

    @testset "FormField" begin
        for space in spaces_2D
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space)
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space)
            ζ¹.coefficients .= 1.0

            β¹ = Forms.FormField(one_form_space)
            β¹.coefficients .= 1.0

            # Compute exterior derivatives
            dα⁰ = d(α⁰)
            dζ¹ = d(ζ¹)

            # Perform the tests
            for elem_id in 1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # 0-form
                # Exterior derivative of a unity 0-form is a zero 1-form
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dα⁰, elem_id, Quadrature.get_nodes(q_rule)
                                )[1][1],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dα⁰, elem_id, Quadrature.get_nodes(q_rule)
                                )[1][2],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )

                # 1-form
                # Exterior derivative of a unity 1-form is a zero 2-form
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dζ¹, elem_id, Quadrature.get_nodes(q_rule)
                                )[1][1],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )
            end
        end
    end

    @testset "Wedge product of FormField" begin
        for space in spaces_2D
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space)
            Random.rand!(α⁰.coefficients)

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space)
            Random.rand!(ζ¹.coefficients)

            # Compute exterior derivative of wedge product
            # Reference via Leibniz rule
            dα⁰_wedge_ζ¹_via_leibniz = (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹))

            # Error form
            error_form_d_wedge = d(α⁰ ∧ ζ¹) - ((d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)))

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all
            # elements
            for elem_id in 1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # Evaluate the Leibniz rule form to check we are not in the trivial case
                dα⁰_wedge_ζ¹_via_leibniz_eval, _ = Forms.evaluate(
                    dα⁰_wedge_ζ¹_via_leibniz, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    >(0),
                    [sum(abs.(component)) for component in dα⁰_wedge_ζ¹_via_leibniz_eval],
                )  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of
                # wedge product and explicit Leibniz rule
                error_form_d_wedge_eval, _ = Forms.evaluate(
                    error_form_d_wedge, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox.(
                        [
                            sum(abs.(component_error)) for
                            component_error in error_form_d_wedge_eval
                        ],
                        0.0,
                        atol=1e-12,
                    ),
                )
            end
        end
    end

    @testset "Wedge product of FormField and FormSpace" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for space in spaces_2D
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space)
            Random.rand!(α⁰.coefficients)

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space)
            Random.rand!(ζ¹.coefficients)

            # Compute exterior derivative of wedge product
            d_α⁰_wedge_one_form_space = d(α⁰ ∧ one_form_space)

            # Reference via Leibniz rule
            d_α⁰_wedge_one_form_space_via_leibniz =
                (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space))

            # Error form
            error_form_d_wedge =
                d_α⁰_wedge_one_form_space -
                ((d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)))

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all
            # elements
            for elem_id in 1:1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # Evaluate the Leibniz rule form to check we are not in the trivial case
                d_α⁰_wedge_one_form_space_via_leibniz_eval, _ = Forms.evaluate(
                    d_α⁰_wedge_one_form_space_via_leibniz,
                    elem_id,
                    Quadrature.get_nodes(q_rule),
                )
                @test all(
                    >(0),
                    [
                        sum(abs.(component)) for
                        component in d_α⁰_wedge_one_form_space_via_leibniz_eval
                    ],
                )  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of
                # wedge product and explicit Leibniz rule
                error_form_d_wedge_eval, _ = Forms.evaluate(
                    error_form_d_wedge, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox.(
                        [
                            sum(abs.(component_error)) for
                            component_error in error_form_d_wedge_eval
                        ],
                        0.0,
                        atol=1e-12,
                    ),
                )
            end
        end
    end

    @testset "Binary product of FormField" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for space in spaces_2D
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")

            # Generate the form expressions
            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space)
            Random.rand!(ζ¹.coefficients)

            β¹ = Forms.FormField(one_form_space)
            Random.rand!(β¹.coefficients)

            # Compute exterior derivative of binary product product (subtraction)
            # Reference via explicit expression
            d_β¹_minus_ζ¹_explicit = d(β¹) - d(ζ¹)

            # Error form
            error_form_d_minus = d(β¹ - ζ¹) - (d(β¹) - d(ζ¹))

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                d_β¹_minus_ζ¹_explicit_eval, _ = Forms.evaluate(
                    d_β¹_minus_ζ¹_explicit, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    >(0),
                    [sum(abs.(component)) for component in d_β¹_minus_ζ¹_explicit_eval],
                )  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_minus_eval, _ = Forms.evaluate(
                    error_form_d_minus, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox.(
                        [
                            sum(abs.(component_error)) for
                            component_error in error_form_d_minus_eval
                        ],
                        0.0,
                        atol=1e-12,
                    ),
                )
            end
        end
    end

    @testset "Unary product of FormField" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for space in spaces_2D
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")

            # Generate the form expressions
            # 1-form: random
            ζ¹ = Forms.FormField(one_form_space)
            Random.rand!(ζ¹.coefficients)

            # Compute exterior derivative of unitary product product (multiplication by scalar)
            c = 2.0

            # Reference via explicit expression
            d_c_ζ¹_explicit = c * d(ζ¹)

            # Error form
            error_form_d_times = d(c * ζ¹) - (c * d(ζ¹))

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                d_c_ζ¹_explicit_eval, _ = Forms.evaluate(
                    d_c_ζ¹_explicit, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    >(0), [sum(abs.(component)) for component in d_c_ζ¹_explicit_eval]
                )  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_times_eval, _ = Forms.evaluate(
                    error_form_d_times, elem_id, Quadrature.get_nodes(q_rule)
                )
                @test all(
                    isapprox.(
                        [
                            sum(abs.(component_error)) for
                            component_error in error_form_d_times_eval
                        ],
                        0.0,
                        atol=1e-12,
                    ),
                )
            end
        end
    end
end

# 3D tests --------------------------------------------------------------------
# Setup
const B = FunctionSpaces.BSplineSpace(geo_cart_1, deg, [-1, deg - 1, -1])

# tensor-product B-spline spaces
const TP_Space_2d = FunctionSpaces.TensorProductSpace((B, B))

const TP_Space_2d_crazy = FunctionSpaces.TensorProductSpace(
    (B, B), Geometry.CartesianGeometry, crazy_mapping
)

const TP_Space_3d_cart_tp_geo = FunctionSpaces.TensorProductSpace((B, B, B))
const TP_Space_3d_tpcart_tp_geo_1 = FunctionSpaces.TensorProductSpace((TP_Space_2d, B))
const TP_Space_3d_cart_geo = FunctionSpaces.TensorProductSpace(
    (B, B, B),
    Geometry.CartesianGeometry,  # Converts the geometry to the desired Cartesian one.
)
const TP_Space_3d_crazycart_tp_geo = FunctionSpaces.TensorProductSpace((
    TP_Space_2d_crazy, B
))

const q_rule_3D = Quadrature.tensor_product_rule(
    (deg, deg, deg) .+ 1, Quadrature.gauss_legendre
)

const spaces_3D = (
    TP_Space_3d_cart_tp_geo,
    TP_Space_3d_tpcart_tp_geo_1,
    TP_Space_3d_cart_geo,
    TP_Space_3d_crazycart_tp_geo,
)
@testset "3D" verbose = true begin
    @testset "FormField" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for space in spaces_3D
            dsTP_1_form_3d = FunctionSpaces.DirectSumSpace((space, space, space))
            dsTP_2_form_3d = FunctionSpaces.DirectSumSpace((space, space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_3d, "η")
            two_form_space = Forms.FormSpace(2, dsTP_2_form_3d, "μ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space)
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space)
            ζ¹.coefficients .= 1.0

            # 2-form: constant
            β² = Forms.FormField(two_form_space)
            β².coefficients .= 1.0

            # Exterior derivative of all forms
            dα⁰ = Forms.ExteriorDerivative(α⁰)
            dζ¹ = Forms.ExteriorDerivative(ζ¹)
            dβ² = Forms.ExteriorDerivative(β²)

            for elem_id in 1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # 0-form
                # Exterior derivative of a unity 0-form is a zero 1-form
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dα⁰, elem_id, Quadrature.get_nodes(q_rule_3D)
                                )[1][1],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dα⁰, elem_id, Quadrature.get_nodes(q_rule_3D)
                                )[1][2],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dα⁰, elem_id, Quadrature.get_nodes(q_rule_3D)
                                )[1][3],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )

                # 1-form
                # Exterior derivative of a unity 1-form is a zero 2-form
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dζ¹, elem_id, Quadrature.get_nodes(q_rule_3D)
                                )[1][1],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dζ¹, elem_id, Quadrature.get_nodes(q_rule_3D)
                                )[1][2],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dζ¹, elem_id, Quadrature.get_nodes(q_rule_3D)
                                )[1][3],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )

                # 2-form
                # Exterior derivative of a unity 1-form is a zero 3-form
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dβ², elem_id, Quadrature.get_nodes(q_rule_3D)
                                )[1][1],
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )
            end
        end
    end
end

end

module ExteriorDerivativeTests

using Mantis

using Test

using LinearAlgebra, SparseArrays
using Random

# Flag to check time while testing
const TIME_TESTS = false

# Common mappings -------------------------------------------------------------

# Crazy mapping
const Lleft = 0.0
const Lright = 1.0
const Lbottom = 0.0
const Ltop = 1.0
const c = 0.2
function mapping_ed_test(x::Vector{Float64})
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0

    return [
        x[1] + ((Lright - Lleft) / 2.0) * c * sinpi(x1_new) * sinpi(x2_new),
        x[2] + ((Ltop - Lbottom) / 2.0) * c * sinpi(x1_new) * sinpi(x2_new),
    ]
end

function dmapping_ed_test(x::Vector{Float64})
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0

    return [
        1.0+pi * c * cospi(x1_new) * sinpi(x2_new) ((Lright - Lleft)/(Ltop - Lbottom))*pi*c*sinpi(x1_new)*cospi(x2_new)
        ((Ltop - Lbottom)/(Lright - Lleft))*pi*c*cospi(x1_new)*sinpi(x2_new) 1.0+pi * c * sinpi(x1_new) * cospi(x2_new)
    ]
end

const crazy_mapping = Geometry.Mapping(Val(2), Val(2), mapping_ed_test, dmapping_ed_test)

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
@testset "2D" verbose = true begin
    @testset "Error barriers" begin
        foreach((TP_Space_2d_cart_geo, TP_Space_2d_tp_geo, TP_Space_2d_crazy_geo)) do space
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")
            top_form_space = Forms.FormSpace(2, space, "σ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space, "α")
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space, "ζ")
            ζ¹.coefficients .= 1.0

            β¹ = Forms.FormField(one_form_space, "ζ")
            β¹.coefficients .= 1.0

            # Compute exterior derivatives
            dα⁰ = d(α⁰)
            dζ¹ = d(ζ¹)
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
        foreach((TP_Space_2d_cart_geo, TP_Space_2d_tp_geo, TP_Space_2d_crazy_geo)) do space
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")
            top_form_space = Forms.FormSpace(2, space, "σ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space, "α")
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space, "ζ")
            ζ¹.coefficients .= 1.0

            β¹ = Forms.FormField(one_form_space, "ζ")
            β¹.coefficients .= 1.0

            # Compute exterior derivatives
            dα⁰ = d(α⁰)
            dζ¹ = d(ζ¹)
            # ddα⁰ = d(dα⁰)

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
                                )[1][1]
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
                                )[1][2]
                            ),
                        ),
                        0.0;
                        atol=1e-12,
                    ),
                )
                # ddα⁰_eval = Forms.evaluate(ddα⁰, elem_id, Quadrature.get_nodes(q_rule))
                # num_components_ddα⁰ = 1
                # @test all(isapprox(sum(abs.(ddα⁰_eval[1][1])), 0.0, atol=1e-12))
                # @test size(ddα⁰_eval[1]) == (num_components_ddα⁰, )
                # @test length(ddα⁰_eval[1][1]) == prod(size.(Quadrature.get_nodes(q_rule), 1))

                # 1-form
                # Exterior derivative of a unity 1-form is a zero 2-form
                @test all(
                    isapprox(
                        sum(
                            abs.(
                                Forms.evaluate(
                                    dζ¹, elem_id, Quadrature.get_nodes(q_rule)
                                )[1][1]
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
        foreach((TP_Space_2d_cart_geo, TP_Space_2d_tp_geo, TP_Space_2d_crazy_geo)) do space
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")
            top_form_space = Forms.FormSpace(2, space, "σ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space, "α")
            Random.rand!(α⁰.coefficients)

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space, "ζ")
            Random.rand!(ζ¹.coefficients)

            if TIME_TESTS
                @info "Timing tests for form fields..."
            end

            # Compute exterior derivative of wedge product
            d_α⁰_wedge_ζ¹ = d(α⁰ ∧ ζ¹)
            if TIME_TESTS
                time_spent = @timed d_α⁰_wedge_ζ¹ = d(α⁰ ∧ ζ¹)
                @info "Time spent for d(α⁰ ∧ ζ¹): $(time_spent.time) seconds"
            end

            # Reference via Leibniz rule
            dα⁰_wedge_ζ¹_via_leibniz = (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹))
            if TIME_TESTS
                time_spent = @timed dα⁰_wedge_ζ¹_via_leibniz = (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹))
                @info "Time spent for (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)): $(time_spent.time) seconds"
            end

            # Error form
            error_form_d_wedge = d(α⁰ ∧ ζ¹) - ((d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)))
            if TIME_TESTS
                time_spent = @timed error_form_d_wedge =
                    d(α⁰ ∧ ζ¹) - ((d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)))
                @info "Time spent for d(α⁰ ∧ ζ¹) - (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)): $(time_spent.time) seconds"
            end

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                dα⁰_wedge_ζ¹_via_leibniz_eval, _ = Forms.evaluate(
                    dα⁰_wedge_ζ¹_via_leibniz, elem_id, Quadrature.get_nodes(q_rule)
                )
                if TIME_TESTS
                    time_spent = @timed dα⁰_wedge_ζ¹_via_leibniz_eval, _ = Forms.evaluate(
                        dα⁰_wedge_ζ¹_via_leibniz, elem_id, Quadrature.get_nodes(q_rule)
                    )
                    @info "Time spent for evaluating (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)): $(time_spent.time) seconds"
                end
                @test all(
                    >(0),
                    [sum(abs.(component)) for component in dα⁰_wedge_ζ¹_via_leibniz_eval],
                )  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_wedge_eval, _ = Forms.evaluate(
                    error_form_d_wedge, elem_id, Quadrature.get_nodes(q_rule)
                )
                if TIME_TESTS
                    time_spent = @timed error_form_d_wedge_eval, _ = Forms.evaluate(
                        error_form_d_wedge, elem_id, Quadrature.get_nodes(q_rule)
                    )
                    @info "Time spent for evaluating d(α⁰ ∧ ζ¹) - (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)): $(time_spent.time) seconds"
                end
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
        foreach((TP_Space_2d_cart_geo, TP_Space_2d_tp_geo, TP_Space_2d_crazy_geo)) do space
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space, "α")
            Random.rand!(α⁰.coefficients)

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space, "ζ")
            Random.rand!(ζ¹.coefficients)

            if TIME_TESTS
                @info "Timing tests for form spaces..."
            end

            # Compute exterior derivative of wedge product
            d_α⁰_wedge_one_form_space = d(α⁰ ∧ one_form_space)
            if TIME_TESTS
                time_spent = @timed d_α⁰_wedge_one_form_space = d(α⁰ ∧ one_form_space)
                @info "Time spent for d(α⁰ ∧ one_form_space): $(time_spent.time) seconds"
            end

            # Reference via Leibniz rule
            d_α⁰_wedge_one_form_space_via_leibniz =
                (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space))
            if TIME_TESTS
                time_spent = @timed d_α⁰_wedge_one_form_space_via_leibniz =
                    (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space))
                @info "Time spent for (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)): $(time_spent.time) seconds"
            end

            # Error form
            error_form_d_wedge =
                d_α⁰_wedge_one_form_space -
                ((d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)))
            if TIME_TESTS
                time_spent = @timed error_form_d_wedge =
                    d_α⁰_wedge_one_form_space -
                    ((d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)))
                @info "Time spent for d_α⁰_wedge_one_form_space - (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)): $(time_spent.time) seconds"
            end

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                d_α⁰_wedge_one_form_space_via_leibniz_eval, _ = Forms.evaluate(
                    d_α⁰_wedge_one_form_space_via_leibniz,
                    elem_id,
                    Quadrature.get_nodes(q_rule),
                )
                if TIME_TESTS
                    time_spent = @timed d_α⁰_wedge_one_form_space_via_leibniz_eval, _ = Forms.evaluate(
                        d_α⁰_wedge_one_form_space_via_leibniz,
                        elem_id,
                        Quadrature.get_nodes(q_rule),
                    )
                    @info "Time spent for evaluating (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)): $(time_spent.time) seconds"
                end
                @test all(
                    >(0),
                    [
                        sum(abs.(component)) for
                        component in d_α⁰_wedge_one_form_space_via_leibniz_eval
                    ],
                )  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_wedge_eval, _ = Forms.evaluate(
                    error_form_d_wedge, elem_id, Quadrature.get_nodes(q_rule)
                )
                if TIME_TESTS
                    time_spent = @timed error_form_d_wedge_eval, _ = Forms.evaluate(
                        error_form_d_wedge, elem_id, Quadrature.get_nodes(q_rule)
                    )
                    @info "Time spent for evaluating d_α⁰_wedge_one_form_space - ((d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space))): $(time_spent.time) seconds"
                end
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
        foreach((TP_Space_2d_cart_geo, TP_Space_2d_tp_geo, TP_Space_2d_crazy_geo)) do space
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")
            top_form_space = Forms.FormSpace(2, space, "σ")

            # Generate the form expressions
            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space, "ζ")
            Random.rand!(ζ¹.coefficients)

            β¹ = Forms.FormField(one_form_space, "ζ")
            Random.rand!(β¹.coefficients)

            if TIME_TESTS
                @info "Timing tests for binary product of form fields..."
            end

            # Compute exterior derivative of binary product product (subtraction)
            d_β¹_minus_ζ¹ = d(β¹ - ζ¹)
            if TIME_TESTS
                time_spent = @timed d_β¹_minus_ζ¹ = d(β¹ - ζ¹)
                @info "Time spent for d(β¹ - ζ¹): $(time_spent.time) seconds"
            end

            # Reference via explicit expression
            d_β¹_minus_ζ¹_explicit = d(β¹) - d(ζ¹)
            if TIME_TESTS
                time_spent = @timed d_β¹_minus_ζ¹_explicit = d(β¹) - d(ζ¹)
                @info "Time spent for d(β¹) - d(ζ¹): $(time_spent.time) seconds"
            end

            # Error form
            error_form_d_minus = d(β¹ - ζ¹) - (d(β¹) - d(ζ¹))
            if TIME_TESTS
                time_spent = @timed error_form_d_minus = d(β¹ - ζ¹) - (d(β¹) - d(ζ¹))
                @info "Time spent for d(β¹ - ζ¹) - (d(β¹) - d(ζ¹)): $(time_spent.time) seconds"
            end

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                d_β¹_minus_ζ¹_explicit_eval, _ = Forms.evaluate(
                    d_β¹_minus_ζ¹_explicit, elem_id, Quadrature.get_nodes(q_rule)
                )
                if TIME_TESTS
                    time_spent = @timed d_β¹_minus_ζ¹_explicit_eval, _ = Forms.evaluate(
                        d_β¹_minus_ζ¹_explicit, elem_id, Quadrature.get_nodes(q_rule)
                    )
                    @info "Time spent for evaluating d(β¹) - d(ζ¹): $(time_spent.time) seconds"
                end
                @test all(
                    >(0),
                    [sum(abs.(component)) for component in d_β¹_minus_ζ¹_explicit_eval],
                )  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_minus_eval, _ = Forms.evaluate(
                    error_form_d_minus, elem_id, Quadrature.get_nodes(q_rule)
                )
                if TIME_TESTS
                    time_spent = @timed error_form_d_minus_eval, _ = Forms.evaluate(
                        error_form_d_minus, elem_id, Quadrature.get_nodes(q_rule)
                    )
                    @info "Time spent for evaluating d(β¹ - ζ¹) - (d(β¹) - d(ζ¹)): $(time_spent.time) seconds"
                end
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
        foreach((TP_Space_2d_cart_geo, TP_Space_2d_tp_geo, TP_Space_2d_crazy_geo)) do space
            dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_2d, "η")
            top_form_space = Forms.FormSpace(2, space, "σ")

            # Generate the form expressions
            # 1-form: random
            ζ¹ = Forms.FormField(one_form_space, "ζ")
            Random.rand!(ζ¹.coefficients)

            if TIME_TESTS
                @info "Timing tests for unary product of form fields..."
            end

            # Compute exterior derivative of unitary product product (multiplication by scalar)
            c = 2.0
            d_c_ζ¹ = d(c * ζ¹)
            if TIME_TESTS
                time_spent = @timed d_c_ζ¹ = d(c * ζ¹)
                @info "Time spent for d(c*ζ¹): $(time_spent.time) seconds"
            end

            # Reference via explicit expression
            d_c_ζ¹_explicit = c * d(ζ¹)
            if TIME_TESTS
                time_spent = @timed d_c_ζ¹_explicit = c * d(ζ¹)
                @info "Time spent for c*d(ζ¹): $(time_spent.time) seconds"
            end

            # Error form
            error_form_d_times = d(c * ζ¹) - (c * d(ζ¹))
            if TIME_TESTS
                time_spent = @timed error_form_d_times = d(c * ζ¹) - (c * d(ζ¹))
                @info "Time spent for d(c*ζ¹) - (c*d(ζ¹)): $(time_spent.time) seconds"
            end

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:1:Geometry.get_num_elements(FunctionSpaces.get_geometry(space))
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                d_c_ζ¹_explicit_eval, _ = Forms.evaluate(
                    d_c_ζ¹_explicit, elem_id, Quadrature.get_nodes(q_rule)
                )
                if TIME_TESTS
                    time_spent = @timed d_c_ζ¹_explicit_eval, _ = Forms.evaluate(
                        d_c_ζ¹_explicit, elem_id, Quadrature.get_nodes(q_rule)
                    )
                    @info "Time spent for evaluating c*d(ζ¹): $(time_spent.time) seconds"
                end
                @test all(
                    >(0), [sum(abs.(component)) for component in d_c_ζ¹_explicit_eval]
                )  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_times_eval, _ = Forms.evaluate(
                    error_form_d_times, elem_id, Quadrature.get_nodes(q_rule)
                )
                if TIME_TESTS
                    time_spent = @timed error_form_d_times_eval, _ = Forms.evaluate(
                        error_form_d_times, elem_id, Quadrature.get_nodes(q_rule)
                    )
                    @info "Time spent for evaluating d(c*ζ¹) - (c*d(ζ¹)): $(time_spent.time) seconds"
                end
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
@testset "3D" verbose = true begin
    @testset "FormField" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        foreach((
            TP_Space_3d_cart_tp_geo,
            TP_Space_3d_tpcart_tp_geo_1,
            TP_Space_3d_cart_geo,
            TP_Space_3d_crazycart_tp_geo,
        )) do space
            dsTP_1_form_3d = FunctionSpaces.DirectSumSpace((space, space, space))
            dsTP_2_form_3d = FunctionSpaces.DirectSumSpace((space, space, space))

            # Create form spaces
            zero_form_space = Forms.FormSpace(0, space, "ν")
            one_form_space = Forms.FormSpace(1, dsTP_1_form_3d, "η")
            two_form_space = Forms.FormSpace(2, dsTP_2_form_3d, "μ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Forms.FormField(zero_form_space, "α")
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Forms.FormField(one_form_space, "ζ")
            ζ¹.coefficients .= 1.0

            # 2-form: constant
            β² = Forms.FormField(two_form_space, "β")
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
                                )[1][1]
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
                                )[1][2]
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
                                )[1][3]
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
                                )[1][1]
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
                                )[1][2]
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
                                )[1][3]
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
                                )[1][1]
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

# -----------------------------------------------------------------------------
end

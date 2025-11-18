module ExteriorDerivativeTests

using Mantis

using Test

using LinearAlgebra, SparseArrays
using Random

# Flag to check time while testing
const TIME_TESTS = false

# Common mappings -------------------------------------------------------------

# Crazy mapping
crazy_c = 0.2
function mapping_ed_test(x::Vector{Float64})
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0

    return [
        x[1] + ((Lright - Lleft) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
        x[2] + ((Ltop - Lbottom) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
    ]
end

function dmapping_ed_test(x::Vector{Float64})
    x1_new = (2.0 / (Lright - Lleft)) * x[1] - 2.0 * Lleft / (Lright - Lleft) - 1.0
    x2_new = (2.0 / (Ltop - Lbottom)) * x[2] - 2.0 * Lbottom / (Ltop - Lbottom) - 1.0

    return [
        1.0+pi * crazy_c * cospi(x1_new) * sinpi(x2_new) ((Lright - Lleft)/(Ltop - Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)
        ((Ltop - Lbottom)/(Lright - Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0+pi * crazy_c * sinpi(x1_new) * cospi(x2_new)
    ]
end

dimension = (2, 2)
crazy_mapping = Mantis.Geometry.Mapping(dimension, mapping_ed_test, dmapping_ed_test)

# 2D tests --------------------------------------------------------------------
@testset "2D" verbose=true begin
    # Domain
    Lleft = 0.0
    Lright = 1.0
    Lbottom = 0.0
    Ltop = 1.0

    # Setup the form spaces
    # First the FEM spaces
    breakpoints1 = [Lleft, 0.5, Lright]
    patch1 = Mantis.Mesh.Patch1D(breakpoints1)
    breakpoints2 = [Lbottom, 0.5, 0.6, Ltop]
    patch2 = Mantis.Mesh.Patch1D(breakpoints2)

    # first B-spline patch
    deg1 = 2
    deg2 = 2
    B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
    # second B-spline patch
    B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
    # tensor-product B-spline patch
    TP_Space_2d = Mantis.FunctionSpaces.TensorProductSpace((B1, B2))


    # Direct sum spaces 2d
    dsTP_0_form_2d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_2d,))
    dsTP_1_form_2d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_2d, TP_Space_2d))
    dsTP_2_form_2d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_2d,))

    # Then the geometry
    # Line 1
    line_1_geo = Mantis.Geometry.CartesianGeometry((breakpoints1,))

    # Line 2
    line_2_geo = Mantis.Geometry.CartesianGeometry((breakpoints2,))

    # Tensor product geometry
    tensor_prod_geo = Mantis.Geometry.TensorProductGeometry((line_1_geo, line_2_geo))
    geo_2d_cart = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2))

    # Crazy geometry
    geom_crazy = Mantis.Geometry.MappedGeometry(geo_2d_cart, crazy_mapping)

    q_rule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1), Mantis.Quadrature.gauss_legendre)

    @testset "Error barriers" begin 
        for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
            # Create form spaces
            zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_2d, "ν")
            one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_2d, "η")
            top_form_space = Mantis.Forms.FormSpace(2, geom, dsTP_2_form_2d, "σ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
            ζ¹.coefficients .= 1.0

            β¹ = Mantis.Forms.FormField(one_form_space, "ζ")
            β¹.coefficients .= 1.0

            # Compute exterior derivatives
            dα⁰ = d(α⁰)
            dζ¹ = d(ζ¹)
            @test_throws ArgumentError (d(zero_form_space) ∧ one_form_space) + (zero_form_space ∧ top_form_space)
                
            # Check if compatible spaces error is NOT thrown
            @test begin
                (α⁰ ∧ zero_form_space ∧ one_form_space) + (zero_form_space ∧ one_form_space)
                true
            end

            # Check if incompatible form_rank error is thrown (no method should exist)
            @test_throws MethodError (d(α⁰) ∧ zero_form_space ∧ one_form_space) + (zero_form_space ∧ one_form_space)
            @test_throws MethodError d(α⁰ ∧ one_form_space) + (zero_form_space ∧ top_form_space)
            @test_throws MethodError (d(α⁰) ∧ one_form_space) + (d(zero_form_space) ∧ one_form_space)
        end
    end

    @testset "FormField" begin
        for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
            # Create form spaces
            zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_2d, "ν")
            one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_2d, "η")
            top_form_space = Mantis.Forms.FormSpace(2, geom, dsTP_2_form_2d, "σ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
            ζ¹.coefficients .= 1.0

            β¹ = Mantis.Forms.FormField(one_form_space, "ζ")
            β¹.coefficients .= 1.0

            # Compute exterior derivatives
            dα⁰ = d(α⁰)
            dζ¹ = d(ζ¹)
            # ddα⁰ = d(dα⁰)

            # Perform the tests
            for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
                # 0-form
                # Exterior derivative of a unity 0-form is a zero 1-form
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dα⁰, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][1])), 0.0, atol=1e-12))
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dα⁰, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][2])), 0.0, atol=1e-12))
                # ddα⁰_eval = Mantis.Forms.evaluate(ddα⁰, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                # num_components_ddα⁰ = 1
                # @test all(isapprox(sum(abs.(ddα⁰_eval[1][1])), 0.0, atol=1e-12))
                # @test size(ddα⁰_eval[1]) == (num_components_ddα⁰, )
                # @test length(ddα⁰_eval[1][1]) == prod(size.(Mantis.Quadrature.get_nodes(q_rule), 1))

                # 1-form
                # Exterior derivative of a unity 1-form is a zero 2-form
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][1])), 0.0, atol=1e-12))
            end
        end
    end

    @testset "Wedge product of FormField" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
            # Create form spaces
            zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_2d, "ν")
            one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_2d, "η")
            top_form_space = Mantis.Forms.FormSpace(2, geom, dsTP_2_form_2d, "σ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
            Random.rand!(α⁰.coefficients)

            # 1-form: constant
            ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
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
                time_spent = @timed error_form_d_wedge = d(α⁰ ∧ ζ¹) - ((d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)))
                @info "Time spent for d(α⁰ ∧ ζ¹) - (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)): $(time_spent.time) seconds"
            end 

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                dα⁰_wedge_ζ¹_via_leibniz_eval, _ = Mantis.Forms.evaluate(dα⁰_wedge_ζ¹_via_leibniz, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                if TIME_TESTS
                    time_spent = @timed dα⁰_wedge_ζ¹_via_leibniz_eval, _ = Mantis.Forms.evaluate(dα⁰_wedge_ζ¹_via_leibniz, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                    @info "Time spent for evaluating (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)): $(time_spent.time) seconds"
                end
                @test all(>(0), [sum(abs.(component)) for component in dα⁰_wedge_ζ¹_via_leibniz_eval])  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_wedge_eval, _ = Mantis.Forms.evaluate(error_form_d_wedge, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                if TIME_TESTS
                    time_spent = @timed error_form_d_wedge_eval, _ = Mantis.Forms.evaluate(error_form_d_wedge, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                    @info "Time spent for evaluating d(α⁰ ∧ ζ¹) - (d(α⁰) ∧ ζ¹) + (α⁰ ∧ d(ζ¹)): $(time_spent.time) seconds"
                end
                @test all(isapprox.([sum(abs.(component_error)) for component_error in error_form_d_wedge_eval], 0.0, atol=1e-12))
            end
        end
    end

    @testset "Wedge product of FormField and FormSpace" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
            # Create form spaces
            zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_2d, "ν")
            one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_2d, "η")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
            Random.rand!(α⁰.coefficients)

            # 1-form: constant
            ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
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
            d_α⁰_wedge_one_form_space_via_leibniz = (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space))
            if TIME_TESTS
                time_spent = @timed d_α⁰_wedge_one_form_space_via_leibniz = (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space))
                @info "Time spent for (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)): $(time_spent.time) seconds"
            end

            # Error form 
            error_form_d_wedge = d_α⁰_wedge_one_form_space - ((d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)))
            if TIME_TESTS
                time_spent = @timed error_form_d_wedge = d_α⁰_wedge_one_form_space - ((d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)))
                @info "Time spent for d_α⁰_wedge_one_form_space - (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)): $(time_spent.time) seconds"
            end 

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                d_α⁰_wedge_one_form_space_via_leibniz_eval, _ = Mantis.Forms.evaluate(d_α⁰_wedge_one_form_space_via_leibniz, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                if TIME_TESTS
                    time_spent = @timed d_α⁰_wedge_one_form_space_via_leibniz_eval, _ = Mantis.Forms.evaluate(d_α⁰_wedge_one_form_space_via_leibniz, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                    @info "Time spent for evaluating (d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space)): $(time_spent.time) seconds"
                end
                @test all(>(0), [sum(abs.(component)) for component in d_α⁰_wedge_one_form_space_via_leibniz_eval])  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_wedge_eval, _ = Mantis.Forms.evaluate(error_form_d_wedge, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                if TIME_TESTS
                    time_spent = @timed error_form_d_wedge_eval, _ = Mantis.Forms.evaluate(error_form_d_wedge, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                    @info "Time spent for evaluating d_α⁰_wedge_one_form_space - ((d(α⁰) ∧ one_form_space) + (α⁰ ∧ d(one_form_space))): $(time_spent.time) seconds"
                end
                @test all(isapprox.([sum(abs.(component_error)) for component_error in error_form_d_wedge_eval], 0.0, atol=1e-12))
            end
        end
    end

    @testset "Binary product of FormField" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
            # Create form spaces
            zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_2d, "ν")
            one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_2d, "η")
            top_form_space = Mantis.Forms.FormSpace(2, geom, dsTP_2_form_2d, "σ")

            # Generate the form expressions
            # 1-form: constant
            ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
            Random.rand!(ζ¹.coefficients)

            β¹ = Mantis.Forms.FormField(one_form_space, "ζ")
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
            for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                d_β¹_minus_ζ¹_explicit_eval, _ = Mantis.Forms.evaluate(d_β¹_minus_ζ¹_explicit, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                if TIME_TESTS
                    time_spent = @timed d_β¹_minus_ζ¹_explicit_eval, _ = Mantis.Forms.evaluate(d_β¹_minus_ζ¹_explicit, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                    @info "Time spent for evaluating d(β¹) - d(ζ¹): $(time_spent.time) seconds"
                end
                @test all(>(0), [sum(abs.(component)) for component in d_β¹_minus_ζ¹_explicit_eval])  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_minus_eval, _ = Mantis.Forms.evaluate(error_form_d_minus, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                if TIME_TESTS
                    time_spent = @timed error_form_d_minus_eval, _ = Mantis.Forms.evaluate(error_form_d_minus, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                    @info "Time spent for evaluating d(β¹ - ζ¹) - (d(β¹) - d(ζ¹)): $(time_spent.time) seconds"
                end
                @test all(isapprox.([sum(abs.(component_error)) for component_error in error_form_d_minus_eval], 0.0, atol=1e-12))
            end
        end
    end

    @testset "Unary product of FormField" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for geom in [geo_2d_cart, tensor_prod_geo, geom_crazy]
            # Create form spaces
            zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_2d, "ν")
            one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_2d, "η")
            top_form_space = Mantis.Forms.FormSpace(2, geom, dsTP_2_form_2d, "σ")

            # Generate the form expressions
            # 1-form: random
            ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
            Random.rand!(ζ¹.coefficients)
            
            if TIME_TESTS
                @info "Timing tests for unary product of form fields..."
            end

            # Compute exterior derivative of unitary product product (multiplication by scalar) 
            c = 2.0
            d_c_ζ¹ = d(c*ζ¹)
            if TIME_TESTS
                time_spent = @timed d_c_ζ¹ = d(c*ζ¹)
                @info "Time spent for d(c*ζ¹): $(time_spent.time) seconds"
            end

            # Reference via explicit expression
            d_c_ζ¹_explicit = c*d(ζ¹)
            if TIME_TESTS
                time_spent = @timed d_c_ζ¹_explicit = c*d(ζ¹)
                @info "Time spent for c*d(ζ¹): $(time_spent.time) seconds"
            end

            # Error form 
            error_form_d_times = d(c*ζ¹) - (c*d(ζ¹))
            if TIME_TESTS
                time_spent = @timed error_form_d_times = d(c*ζ¹) - (c*d(ζ¹))
                @info "Time spent for d(c*ζ¹) - (c*d(ζ¹)): $(time_spent.time) seconds"
            end 

            # Check error of automatic exterior derivative vs explicit Leibniz rule on all elements
            for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom)
                # Evaluate the Leibniz rule form to check we are not in the trivial case (= 0)
                d_c_ζ¹_explicit_eval, _ = Mantis.Forms.evaluate(d_c_ζ¹_explicit, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                if TIME_TESTS
                    time_spent = @timed d_c_ζ¹_explicit_eval, _ = Mantis.Forms.evaluate(d_c_ζ¹_explicit, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                    @info "Time spent for evaluating c*d(ζ¹): $(time_spent.time) seconds"
                end
                @test all(>(0), [sum(abs.(component)) for component in d_c_ζ¹_explicit_eval])  # Check it's not zero just not to check trivial case

                # Evaluate the error between explicit and automatic exterior derivative of wedge product and
                # explicit Leibniz rule
                error_form_d_times_eval, _ = Mantis.Forms.evaluate(error_form_d_times, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                if TIME_TESTS
                    time_spent = @timed error_form_d_times_eval, _ = Mantis.Forms.evaluate(error_form_d_times, elem_id, Mantis.Quadrature.get_nodes(q_rule))
                    @info "Time spent for evaluating d(c*ζ¹) - (c*d(ζ¹)): $(time_spent.time) seconds"
                end
                @test all(isapprox.([sum(abs.(component_error)) for component_error in error_form_d_times_eval], 0.0, atol=1e-12))
            end
        end
    end
end

# 3D tests --------------------------------------------------------------------
@testset "3D" verbose=true begin
    # Setup the geometry

    # Domain
    Lleft = 0.0
    Lright = 1.0
    Lbottom = 0.0
    Ltop = 1.0

    # Setup the form spaces

    # First the FEM spaces
    breakpoints = [Lleft, 0.5, Lright]
    patch = Mantis.Mesh.Patch1D(breakpoints)

    # first B-spline patch
    deg = 2
    B = Mantis.FunctionSpaces.BSplineSpace(patch, deg, [-1, deg-1, -1])

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
    dsTP_1_form_3d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_3d, TP_Space_3d, TP_Space_3d))  # direct sum space
    dsTP_2_form_3d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_3d, TP_Space_3d, TP_Space_3d))
    dsTP_top_form_3d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_3d,))  # direct sum space


    # Quadrature rule
    q_rule = Mantis.Quadrature.tensor_product_rule((deg, deg, deg) .+ 1, Mantis.Quadrature.gauss_legendre)

    @testset "FormField" begin
        # Test on multiple geometries. Type-wise and content/metric wise.
        for geom in [geo_3d_cart, crazy_geo_3d_cart]
            # Create form spaces
            zero_form_space = Mantis.Forms.FormSpace(0, geom, dsTP_0_form_3d, "ν")
            one_form_space = Mantis.Forms.FormSpace(1, geom, dsTP_1_form_3d, "η")
            two_form_space = Mantis.Forms.FormSpace(2, geom, dsTP_2_form_3d, "μ")

            # Generate the form expressions
            # 0-form: constant
            α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
            α⁰.coefficients .= 1.0

            # 1-form: constant
            ζ¹ = Mantis.Forms.FormField(one_form_space, "ζ")
            ζ¹.coefficients .= 1.0

            # 2-form: constant
            β² = Mantis.Forms.FormField(two_form_space, "β")
            β².coefficients .= 1.0

            # Exterior derivative of all forms
            dα⁰ = Mantis.Forms.ExteriorDerivative(α⁰)
            dζ¹ = Mantis.Forms.ExteriorDerivative(ζ¹)
            dβ² = Mantis.Forms.ExteriorDerivative(β²)

            for elem_id in 1:Mantis.Geometry.get_num_elements(geom)
                # 0-form
                # Exterior derivative of a unity 0-form is a zero 1-form
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dα⁰, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][1])), 0.0, atol=1e-12))
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dα⁰, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][2])), 0.0, atol=1e-12))
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dα⁰, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][3])), 0.0, atol=1e-12))

                # 1-form
                # Exterior derivative of a unity 1-form is a zero 2-form
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][1])), 0.0, atol=1e-12))
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][2])), 0.0, atol=1e-12))
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dζ¹, elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][3])), 0.0, atol=1e-12))

                # 2-form
                # Exterior derivative of a unity 1-form is a zero 3-form
                @test all(isapprox(sum(abs.(Mantis.Forms.evaluate(dβ², elem_id, Mantis.Quadrature.get_nodes(q_rule))[1][1])), 0.0, atol=1e-12))

            end
        end
    end
end

# -----------------------------------------------------------------------------
end

module PullbackTests

using Mantis
using Test
using LinearAlgebra
using StaticArrays

include("../../TestHelpers.jl")

############################################################################################
#                                          Setup                                           #
############################################################################################
xi_1D = Points.TensorProductPoints((LinRange(0.0, 1.0, 5),))
xi_2D = Points.TensorProductPoints((LinRange(0.0, 1.0, 5), LinRange(0.0, 1.0, 5)))
xi_3D = Points.TensorProductPoints((
    LinRange(0.0, 1.0, 5), LinRange(0.0, 1.0, 5), LinRange(0.0, 1.0, 5)
))

one_func_1c(x) = [ones(size(x, 1))]
one_func_2c(x) = [ones(size(x, 1)), ones(size(x, 1))]
one_func_3c(x) = [ones(size(x, 1)), ones(size(x, 1)), ones(size(x, 1))]

function approx(v, e)
    return isapprox(v, e; rtol=1e-15)
end

############################################################################################
#                                      Form Pullbacks                                      #
############################################################################################

# FormSpaces and FormFields test pullbacks from the canonical to the parametric domain but
# with different input types for the evaluate (Vector{Vector{}} and Vector{Matrix{}}).
# AnalyticalFormFields test pullbacks from the physical to the canonical domain.
@testset "FormPullbacks" verbose=true begin
    @testset "Identity map" begin
        # The identity map can be implemented by using a single-element cartesian geometry
        # for the form.
        @testset "1D" begin
            X_1D = Forms.create_tensor_product_bspline_de_rham_complex(
                (0.0,), (1.0,), (1,), (4,), (3,)
            )
            num_bases = map(Forms.get_num_basis, X_1D)
            coeffs = map(ones, num_bases)
            X_1D_fields = map(Forms.FormField, X_1D, coeffs)

            @testset "FormSpaces" begin
                for form in X_1D
                    evaluations_fe = FunctionSpaces.evaluate(
                        Forms.get_fe_space(form), 1, xi_1D, 0
                    )[1][1][1]
                    evaluations_fe_ref = deepcopy(evaluations_fe)

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Parametric,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_fe_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_fe, pb, k, xi_1D) approx
                end
            end

            @testset "FormFields" begin
                for form in X_1D_fields
                    evaluations_fe = FunctionSpaces.evaluate(
                        Forms.get_fe_space(Forms.get_form(form)),
                        1,
                        xi_1D,
                        0,
                        Forms.get_coefficients(form),
                    )[1][1]
                    evaluations_fe_ref = deepcopy(evaluations_fe)

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Parametric,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_fe_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_fe, pb, k, xi_1D) approx
                end
            end

            @testset "AnalyticalFormFields" begin
                geos = map(Forms.get_geometry, X_1D)
                X_analytical = map(
                    Forms.AnalyticalFormField,
                    (0, 1),
                    (one_func_1c, one_func_1c),
                    geos,
                    ("0", "1"),
                )
                for form in X_analytical
                    x = Geometry.evaluate(Forms.get_geometry(form), 1, xi_1D)
                    evaluations_f = Forms.get_expression(form)(x)
                    evaluations_f_ref = deepcopy(evaluations_f)

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Physical,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_f_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_f, pb, k, xi_1D) approx
                end
            end

            @testset "ConstantFormSpace" begin
                geos = map(Forms.get_geometry, X_1D)
                X_const = map(Forms.ConstantFormSpace, (0, 1), geos, ("0", "1"))
                for form in X_const
                    evaluations_f = [ones(Float64, Points.get_num_points(xi_1D), 1)]
                    evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_1D), 1)]

                    if Forms.get_form_rank(form) == Forms.get_manifold_dim(form)
                        g, sqrt_g = Geometry.metric(Forms.get_geometry(form), 1, xi_1D)
                        for p in eachindex(sqrt_g)
                            evaluations_f_ref[1][p] *= sqrt_g[p]
                        end
                    end

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Physical,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_f_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_f, pb, k, xi_1D) approx
                end
            end
        end

        @testset "2D" begin
            X_2D = Forms.create_tensor_product_bspline_de_rham_complex(
                (0.0, 0.0), (1.0, 1.0), (1, 1), (3, 3), (2, 2)
            )
            num_bases = map(Forms.get_num_basis, X_2D)
            coeffs = map(ones, num_bases)
            X_2D_fields = map(Forms.FormField, X_2D, coeffs)

            @testset "FormSpaces" begin
                for form in X_2D
                    evaluations_fe = FunctionSpaces.evaluate(
                        Forms.get_fe_space(form), 1, xi_2D, 0
                    )[1][1][1]
                    evaluations_fe_ref = deepcopy(evaluations_fe)

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Parametric,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_fe_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_fe, pb, k, xi_2D) approx
                end
            end

            @testset "FormFields" begin
                for form in X_2D_fields
                    evaluations_fe = FunctionSpaces.evaluate(
                        Forms.get_fe_space(Forms.get_form(form)),
                        1,
                        xi_2D,
                        0,
                        Forms.get_coefficients(form),
                    )[1][1]
                    evaluations_fe_ref = deepcopy(evaluations_fe)

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Parametric,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_fe_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_fe, pb, k, xi_2D) approx
                end
            end

            @testset "AnalyticalFormFields" begin
                geos = map(Forms.get_geometry, X_2D)
                X_analytical = map(
                    Forms.AnalyticalFormField,
                    (0, 1, 2),
                    (one_func_1c, one_func_2c, one_func_1c),
                    geos,
                    ("0", "1", "2"),
                )
                for form in X_analytical
                    x = Geometry.evaluate(Forms.get_geometry(form), 1, xi_2D)
                    evaluations_f = Forms.get_expression(form)(x)
                    evaluations_f_ref = deepcopy(evaluations_f)
                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Physical,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_f_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_f, pb, k, xi_2D) approx
                end
            end

            @testset "ConstantFormSpace" begin
                geos = map(Forms.get_geometry, X_2D)
                X_const = map(
                    Forms.ConstantFormSpace, (0, 2), (first(geos), last(geos)), ("0", "2")
                )
                for form in X_const
                    evaluations_f = [ones(Float64, Points.get_num_points(xi_2D), 1)]
                    evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_2D), 1)]

                    if Forms.get_form_rank(form) == Forms.get_manifold_dim(form)
                        g, sqrt_g = Geometry.metric(Forms.get_geometry(form), 1, xi_2D)
                        for p in eachindex(sqrt_g)
                            evaluations_f_ref[1][p] *= sqrt_g[p]
                        end
                    end

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Physical,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_f_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_f, pb, k, xi_2D) approx
                end
            end
        end

        @testset "3D" begin
            X_3D = Forms.create_tensor_product_bspline_de_rham_complex(
                (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), (2, 2, 2), (1, 1, 1)
            )
            num_bases = map(Forms.get_num_basis, X_3D)
            coeffs = map(ones, num_bases)
            X_3D_fields = map(Forms.FormField, X_3D, coeffs)

            @testset "FormSpaces" begin
                for form in X_3D
                    evaluations_fe = FunctionSpaces.evaluate(
                        Forms.get_fe_space(form), 1, xi_3D, 0
                    )[1][1][1]
                    evaluations_fe_ref = deepcopy(evaluations_fe)

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Parametric,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_fe_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_fe, pb, k, xi_3D) approx
                end
            end
            @testset "FormFields" begin
                for form in X_3D_fields
                    evaluations_fe = FunctionSpaces.evaluate(
                        Forms.get_fe_space(Forms.get_form(form)),
                        1,
                        xi_3D,
                        0,
                        Forms.get_coefficients(form),
                    )[1][1]
                    evaluations_fe_ref = deepcopy(evaluations_fe)

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Parametric,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_fe_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_fe, pb, k, xi_3D) approx
                end
            end
            @testset "AnalyticalFormFields" begin
                geos = map(Forms.get_geometry, X_3D)
                X_analytical = map(
                    Forms.AnalyticalFormField,
                    (0, 1, 2, 3),
                    (one_func_1c, one_func_3c, one_func_3c, one_func_1c),
                    geos,
                    ("0", "1", "2", "3"),
                )
                for form in X_analytical
                    x = Geometry.evaluate(Forms.get_geometry(form), 1, xi_3D)
                    evaluations_f = Forms.get_expression(form)(x)
                    evaluations_f_ref = deepcopy(evaluations_f)

                    if Forms.get_form_rank(form) == 2
                        # Currently, the formpullback from physical to canonical is missing
                        # for 2-forms in 3D.
                        continue
                    end

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Physical,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_f_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_f, pb, k, xi_3D) approx
                end
            end
            @testset "ConstantFormSpace" begin
                geos = map(Forms.get_geometry, X_3D)
                X_const = map(
                    Forms.ConstantFormSpace, (0, 3), (first(geos), last(geos)), ("0", "3")
                )
                for form in X_const
                    evaluations_f = [ones(Float64, Points.get_num_points(xi_3D), 1)]
                    evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_3D), 1)]

                    if Forms.get_form_rank(form) == Forms.get_manifold_dim(form)
                        g, sqrt_g = Geometry.metric(Forms.get_geometry(form), 1, xi_3D)
                        for p in eachindex(sqrt_g)
                            evaluations_f_ref[1][p] *= sqrt_g[p]
                        end
                    end

                    pb = Forms.get_pullback(form)

                    @test Forms.get_pullback_type(form) <: Forms.FormPullback
                    @test pb isa Forms.FormPullback{
                        Forms.get_manifold_dim(form),
                        Forms.get_form_rank(form),
                        Forms.get_expression_rank(form),
                        Forms.Physical,
                        Forms.Canonical,
                    }

                    expected_pullback = Dict(1 => evaluations_f_ref)
                    @test @expected expected_pullback k ->
                        Forms.pullback!(evaluations_f, pb, k, xi_3D) approx
                end
            end
        end
    end

    @testset "2D: F(x,y) = (2x+y, 3x-y)" begin
        function mapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            return SVector{2}(2*x+y, 3*x-y)
        end

        function dmapping(xx::AbstractVector)
            # Note: SMatrix creates the matrix per column.
            return SMatrix{2, 2}(2.0, 3.0, 1.0, -1.0)
        end
        poly_mapping = Geometry.Mapping((2, 2), mapping, dmapping)

        X_2D = Forms.create_tensor_product_bspline_de_rham_complex(
            (0.0, 0.0),
            (1.0, 1.0),
            (4, 4), # 4x4 elements, so a scaling occurs from parameteric to canonical.
            (FunctionSpaces.Bernstein(3), FunctionSpaces.Bernstein(3)),
            (2, 2),
            poly_mapping,
        )
        num_bases = map(Forms.get_num_basis, X_2D)
        coeffs = map(ones, num_bases)
        X_2D_fields = map(Forms.FormField, X_2D, coeffs)

        @testset "FormSpaces" begin
            for form in X_2D
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(form), 1, xi_2D, 0
                )[1][1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                if Forms.get_form_rank(form) == 1
                    evaluations_fe_ref[1] .*= 0.25
                    evaluations_fe_ref[2] .*= 0.25
                elseif Forms.get_form_rank(form) == 2
                    evaluations_fe_ref .*= (1/16)
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_2D) approx
            end
        end

        @testset "FormFields" begin
            for form in X_2D_fields
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(Forms.get_form(form)),
                    1,
                    xi_2D,
                    0,
                    Forms.get_coefficients(form),
                )[1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                if Forms.get_form_rank(form) == 1
                    evaluations_fe_ref[1] .*= 0.25
                    evaluations_fe_ref[2] .*= 0.25
                elseif Forms.get_form_rank(form) == 2
                    evaluations_fe_ref .*= (1/16)
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_2D) approx
            end
        end

        @testset "AnalyticalFormFields" begin
            geos = map(Forms.get_geometry, X_2D)
            X_analytical = map(
                Forms.AnalyticalFormField,
                (0, 1, 2),
                (one_func_1c, one_func_2c, one_func_1c),
                geos,
                ("0", "1", "2"),
            )
            for form in X_analytical
                x = Geometry.evaluate(Forms.get_geometry(form), 1, xi_2D)
                evaluations_f = Forms.get_expression(form)(x)
                evaluations_f_old = deepcopy(evaluations_f)
                evaluations_f_ref = deepcopy(evaluations_f)
                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                if Forms.get_form_rank(form) == 1
                    evaluations_f_ref[1] .*=
                        0.25 .* (2.0 .* evaluations_f_old[1] .+ 3.0 .* evaluations_f_old[2])
                    evaluations_f_ref[2] .*=
                        0.25 .* (1.0 .* evaluations_f_old[1] .- 1.0 .* evaluations_f_old[2])
                elseif Forms.get_form_rank(form) == 2
                    evaluations_f_ref .*= 5.0 * (1/16)
                end
                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_2D) approx
            end
        end
        @testset "ConstantFormSpace" begin
            geos = map(Forms.get_geometry, X_2D)
            X_const = map(
                Forms.ConstantFormSpace, (0, 2), (first(geos), last(geos)), ("0", "2")
            )
            for form in X_const
                evaluations_f = [ones(Float64, Points.get_num_points(xi_2D), 1)]
                evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_2D), 1)]

                if Forms.get_form_rank(form) == Forms.get_manifold_dim(form)
                    g, sqrt_g = Geometry.metric(Forms.get_geometry(form), 1, xi_2D)
                    for p in eachindex(sqrt_g)
                        evaluations_f_ref[1][p] *= sqrt_g[p]
                    end
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_2D) approx
            end
        end
    end

    @testset "1D in 3D: F(x,y) = (2x, -3x, x)" begin
        function mapping(xx::AbstractVector)
            x = xx[1]
            return SVector{3}(2*x, -3*x, x)
        end

        function dmapping(xx::AbstractVector)
            x = xx[1]
            # Note: SMatrix creates the matrix per column.
            return SMatrix{3, 1}(2.0, -3.0, 1.0)
        end
        poly_mapping = Geometry.Mapping((1, 3), mapping, dmapping)

        X_1D = Forms.create_tensor_product_bspline_de_rham_complex(
            (0.0,), (1.0,), (4,), (FunctionSpaces.Bernstein(3),), (2,), poly_mapping
        )
        num_bases = map(Forms.get_num_basis, X_1D)
        coeffs = map(ones, num_bases)
        X_1D_fields = map(Forms.FormField, X_1D, coeffs)

        @testset "FormSpaces" begin
            for form in X_1D
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(form), 1, xi_1D, 0
                )[1][1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                if Forms.get_form_rank(form) == 1
                    evaluations_fe_ref .*= 0.25
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_1D) approx
            end
        end

        @testset "FormFields" begin
            for form in X_1D_fields
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(Forms.get_form(form)),
                    1,
                    xi_1D,
                    0,
                    Forms.get_coefficients(form),
                )[1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                if Forms.get_form_rank(form) == 1
                    evaluations_fe_ref .*= 0.25
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_1D) approx
            end
        end

        @testset "AnalyticalFormFields" begin
            geos = map(Forms.get_geometry, X_1D)
            X_analytical = map(
                Forms.AnalyticalFormField,
                (0, 1),
                (one_func_1c, one_func_1c),
                geos,
                ("0", "1"),
            )
            for form in X_analytical
                xx = Geometry.evaluate(Forms.get_geometry(form), 1, xi_1D)
                evaluations_f = Forms.get_expression(form)(xx)
                evaluations_f_ref = deepcopy(evaluations_f)
                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                if Forms.get_form_rank(form) == 1
                    evaluations_f_ref .*= sqrt(14) * 0.25
                end
                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_1D) approx
            end
        end
        @testset "ConstantFormSpace" begin
            geos = map(Forms.get_geometry, X_1D)
            X_const = map(Forms.ConstantFormSpace, (0, 1), geos, ("0", "1"))
            for form in X_const
                evaluations_f = [ones(Float64, Points.get_num_points(xi_1D), 1)]
                evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_1D), 1)]

                if Forms.get_form_rank(form) == Forms.get_manifold_dim(form)
                    g, sqrt_g = Geometry.metric(Forms.get_geometry(form), 1, xi_1D)
                    for p in eachindex(sqrt_g)
                        evaluations_f_ref[1][p] *= sqrt_g[p]
                    end
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_1D) approx
            end
        end
    end

    @testset "2D in 3D: F(x,y) = (2x+y, 3x-y, x+0.5y)" begin
        function mapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            return SVector{3}(2*x+y, 3*x-y, x+0.5y)
        end

        function dmapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            # Note: SMatrix creates the matrix per column.
            return SMatrix{3, 2}(2.0, 3.0, 1.0, 1.0, -1.0, 0.5)
        end
        poly_mapping = Geometry.Mapping((2, 3), mapping, dmapping)

        X_2D = Forms.create_tensor_product_bspline_de_rham_complex(
            (0.0, 0.0),
            (1.0, 1.0),
            (4, 4), # 4x4 elements, so a scaling occurs from parameteric to canonical.
            (FunctionSpaces.Bernstein(3), FunctionSpaces.Bernstein(3)),
            (2, 2),
            poly_mapping,
        )
        num_bases = map(Forms.get_num_basis, X_2D)
        coeffs = map(ones, num_bases)
        X_2D_fields = map(Forms.FormField, X_2D, coeffs)

        @testset "Get pullbacks of operators" begin
            form = first(Base.tail(X_2D))
            deltaform = Forms.CoDifferential(form)
            dform = d(form)
            hodgeform = Forms.Hodge(form)
            wedgeform = Forms.Wedge(form, first(X_2D))
            binform = form + form

            ops = (deltaform, dform, hodgeform, wedgeform, binform)

            for opform in ops
                pb = Forms.get_pullback(opform)
                @test Forms.get_pullback_type(opform) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(opform),
                    Forms.get_form_rank(opform),
                    Forms.get_expression_rank(opform),
                    Forms.Canonical,
                    Forms.Canonical,
                    typeof(opform),
                }
            end
        end

        @testset "FormSpaces" begin
            for form in X_2D
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(form), 1, xi_2D, 0
                )[1][1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                if Forms.get_form_rank(form) == 1
                    evaluations_fe_ref[1] .*= 0.25
                    evaluations_fe_ref[2] .*= 0.25
                elseif Forms.get_form_rank(form) == 2
                    evaluations_fe_ref .*= 1/16
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_2D) approx
            end
        end

        @testset "FormFields" begin
            for form in X_2D_fields
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(Forms.get_form(form)),
                    1,
                    xi_2D,
                    0,
                    Forms.get_coefficients(form),
                )[1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                if Forms.get_form_rank(form) == 1
                    evaluations_fe_ref[1] .*= 0.25
                    evaluations_fe_ref[2] .*= 0.25
                elseif Forms.get_form_rank(form) == 2
                    evaluations_fe_ref .*= 1/16
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_2D) approx
            end
        end

        @testset "AnalyticalFormFields" begin
            geos = map(Forms.get_geometry, X_2D)
            X_analytical = map(
                Forms.AnalyticalFormField,
                (0, 1, 2),
                (one_func_1c, one_func_3c, one_func_1c),
                geos,
                ("0", "1", "2"),
            )
            for form in X_analytical
                xx = Geometry.evaluate(Forms.get_geometry(form), 1, xi_2D)
                evaluations_f = Forms.get_expression(form)(xx)
                evaluations_f_old = deepcopy(evaluations_f)
                evaluations_f_ref = deepcopy(evaluations_f)
                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                if Forms.get_form_rank(form) == 1
                    evaluations_f_ref[1] .*=
                        0.25 .* (
                            2.0 .* evaluations_f_old[1] .+ 3.0 .* evaluations_f_old[2] .+
                            1.0 .* evaluations_f_old[3]
                        )
                    evaluations_f_ref[2] .*=
                        0.25 .* (
                            1.0 .* evaluations_f_old[1] .- 1.0 .* evaluations_f_old[2] .+
                            0.5 .* evaluations_f_old[3]
                        )
                elseif Forms.get_form_rank(form) == 2
                    evaluations_f_ref .*= sqrt(31.25) * (1/16)
                end
                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_2D) approx
            end
        end
        @testset "ConstantFormSpace" begin
            geos = map(Forms.get_geometry, X_2D)
            X_const = map(
                Forms.ConstantFormSpace, (0, 2), (first(geos), last(geos)), ("0", "2")
            )
            for form in X_const
                evaluations_f = [ones(Float64, Points.get_num_points(xi_2D), 1)]
                evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_2D), 1)]

                if Forms.get_form_rank(form) == Forms.get_manifold_dim(form)
                    g, sqrt_g = Geometry.metric(Forms.get_geometry(form), 1, xi_2D)
                    for p in eachindex(sqrt_g)
                        evaluations_f_ref[1][p] *= sqrt_g[p]
                    end
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_2D) approx
            end
        end
    end

    @testset "3D: F(x,y) = (2x^2+y^3-0.25z^4, 3x^3-y^5+2z^2, x+2y+3z)" begin
        function mapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            z = xx[3]
            return SVector{3}(2*x^2+y^3-0.25*z^4, 3*x^3-y^5+2*z^2, x+2*y+3*z)
        end

        function dmapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            z = xx[3]
            # Note: SMatrix creates the matrix per column.
            return SMatrix{3, 3}(
                4*x+y^3-0.25*z^4,
                9*x^2-y^5+2*z^2,
                1+2*y+3*z,
                2*x^2+3*y^2-0.25*z^4,
                3*x^3-5*y^4+2*z^2,
                x+2+3*z,
                2*x^2+y^3-z^3,
                3*x^3-y^5+2*2*z,
                x+2*y+3,
            )
        end
        poly_mapping = Geometry.Mapping((3, 3), mapping, dmapping)

        X_3D = Forms.create_tensor_product_bspline_de_rham_complex(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (4, 3, 2),
            (
                FunctionSpaces.Bernstein(4),
                FunctionSpaces.Bernstein(3),
                FunctionSpaces.Bernstein(2),
            ),
            (3, 2, 1),
            poly_mapping,
        )
        num_bases = map(Forms.get_num_basis, X_3D)
        coeffs = map(ones, num_bases)
        X_3D_fields = map(Forms.FormField, X_3D, coeffs)

        @testset "FormSpaces" begin
            for form in X_3D
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(form), 1, xi_3D, 0
                )[1][1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                if Forms.get_form_rank(form) == 1
                    evaluations_fe_ref[1] .*= 0.25
                    evaluations_fe_ref[2] .*= (1/3)
                    evaluations_fe_ref[3] .*= 0.5
                elseif Forms.get_form_rank(form) == 2
                    evaluations_fe_ref[1] .*= (1/6)
                    evaluations_fe_ref[2] .*= (1/8)
                    evaluations_fe_ref[3] .*= (1/12)
                elseif Forms.get_form_rank(form) == 3
                    evaluations_fe_ref .*= (1/24)
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_3D) approx
            end
        end

        @testset "FormFields" begin
            for form in X_3D_fields
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(Forms.get_form(form)),
                    1,
                    xi_3D,
                    0,
                    Forms.get_coefficients(form),
                )[1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                if Forms.get_form_rank(form) == 1
                    evaluations_fe_ref[1] .*= 0.25
                    evaluations_fe_ref[2] .*= (1/3)
                    evaluations_fe_ref[3] .*= 0.5
                elseif Forms.get_form_rank(form) == 2
                    evaluations_fe_ref[1] .*= (1/6)
                    evaluations_fe_ref[2] .*= (1/8)
                    evaluations_fe_ref[3] .*= (1/12)
                elseif Forms.get_form_rank(form) == 3
                    evaluations_fe_ref .*= (1/24)
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_3D) approx
            end
        end

        @testset "AnalyticalFormFields" verbose=true begin
            geos = map(Forms.get_geometry, X_3D)
            X_analytical = map(
                Forms.AnalyticalFormField,
                (0, 1, 2, 3),
                (one_func_1c, one_func_3c, one_func_3c, one_func_1c),
                geos,
                ("0", "1", "2", "3"),
            )
            for form in X_analytical
                x = Geometry.evaluate(Forms.get_geometry(form), 1, xi_3D)
                evaluations_f = Forms.get_expression(form)(x)
                evaluations_f_old = deepcopy(evaluations_f)
                evaluations_f_ref = deepcopy(evaluations_f)

                if Forms.get_form_rank(form) == 2
                    # Currently, the formpullback from physical to canonical is missing for
                    # 2-forms in 3D.
                    continue
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                if Forms.get_form_rank(form) == 1
                    J = Geometry.jacobian(Forms.get_geometry(form), 1, xi_3D)
                    for p in eachindex(J)
                        JT = transpose(J[p])
                        evaluations_f_ref[1][p] =
                            JT[1, 1] * evaluations_f_old[1][p] +
                            JT[1, 2] * evaluations_f_old[2][p] +
                            JT[1, 3] * evaluations_f_old[3][p]
                        evaluations_f_ref[2][p] =
                            JT[2, 1] * evaluations_f_old[1][p] +
                            JT[2, 2] * evaluations_f_old[2][p] +
                            JT[2, 3] * evaluations_f_old[3][p]
                        evaluations_f_ref[3][p] =
                            JT[3, 1] * evaluations_f_old[1][p] +
                            JT[3, 2] * evaluations_f_old[2][p] +
                            JT[3, 3] * evaluations_f_old[3][p]
                    end
                elseif Forms.get_form_rank(form) == 3
                    g, sqrt_g = Geometry.metric(Forms.get_geometry(form), 1, xi_3D)
                    for p in eachindex(sqrt_g)
                        evaluations_f_ref[1][p] *= sqrt_g[p]
                    end
                end
                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_3D) approx
            end
        end
        @testset "ConstantFormSpace" begin
            geos = map(Forms.get_geometry, X_3D)
            X_const = map(
                Forms.ConstantFormSpace, (0, 3), (first(geos), last(geos)), ("0", "3")
            )
            for form in X_const
                evaluations_f = [ones(Float64, Points.get_num_points(xi_3D), 1)]
                evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_3D), 1)]

                if Forms.get_form_rank(form) == Forms.get_manifold_dim(form)
                    g, sqrt_g = Geometry.metric(Forms.get_geometry(form), 1, xi_3D)
                    for p in eachindex(sqrt_g)
                        evaluations_f_ref[1][p] *= sqrt_g[p]
                    end
                end

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.FormPullback
                @test pb isa Forms.FormPullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_3D) approx
            end
        end
    end
end

############################################################################################
#                                  ComponentWisePullbacks                                  #
############################################################################################

# FormSpaces and FormFields test pullbacks from the canonical to the parametric domain but
# with different input types for the evaluate (Vector{Vector{}} and Vector{Matrix{}}).
# AnalyticalFormFields test pullbacks from the physical to the canonical domain.
@testset "ComponentWisePullbacks" verbose=true begin
    @testset "2D in 3D: F(x,y) = (2x+y, 3x-y, x+0.5y)" begin
        function mapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            return SVector{3}(2*x+y, 3*x-y, x+0.5y)
        end

        function dmapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            # Note: SMatrix creates the matrix per column.
            return SMatrix{3, 2}(2.0, 3.0, 1.0, 1.0, -1.0, 0.5)
        end
        poly_mapping = Geometry.Mapping((2, 3), mapping, dmapping)

        B1 = FunctionSpaces.create_bspline_space(0.0, 1.0, 4, 3, 2)
        B = FunctionSpaces.TensorProductSpace((B1, B1), poly_mapping)
        zero_form_space = Forms.FormSpace(0, B, "0", Forms.ComponentWisePullback)
        one_form_space = Forms.FormSpace(
            1, FunctionSpaces.DirectSumSpace((B, B)), "1", Forms.ComponentWisePullback
        )
        two_form_space = Forms.FormSpace(2, B, "2", Forms.ComponentWisePullback)
        X_2D = (zero_form_space, one_form_space, two_form_space)
        num_bases = map(Forms.get_num_basis, X_2D)
        coeffs = map(ones, num_bases)
        X_2D_fields = map(Forms.FormField, X_2D, coeffs)

        @testset "Get pullbacks of operators" begin
            deltaform = Forms.CoDifferential(one_form_space)
            dform = d(one_form_space)
            hodgeform = Forms.Hodge(one_form_space)
            wedgeform = Forms.Wedge(one_form_space, first(X_2D))
            binform = one_form_space + one_form_space

            ops = (deltaform, dform, hodgeform, wedgeform, binform)

            for opform in ops
                pb = Forms.get_pullback(opform)
                @test Forms.get_pullback_type(opform) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(opform),
                    Forms.get_form_rank(opform),
                    Forms.get_expression_rank(opform),
                    Forms.Canonical,
                    Forms.Canonical,
                    typeof(opform),
                }
            end
        end

        @testset "FormSpaces" begin
            for form in X_2D
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(form), 1, xi_2D, 0
                )[1][1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_2D) approx
            end
        end

        @testset "FormFields" begin
            for form in X_2D_fields
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(Forms.get_form(form)),
                    1,
                    xi_2D,
                    0,
                    Forms.get_coefficients(form),
                )[1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_2D) approx
            end
        end

        @testset "AnalyticalFormFields" begin
            geos = map(Forms.get_geometry, X_2D)
            X_analytical = map(
                Forms.AnalyticalFormField,
                (0, 1, 2),
                (one_func_1c, one_func_3c, one_func_1c),
                geos,
                ("0", "1", "2"),
                (Forms.ComponentWisePullback, Forms.ComponentWisePullback),
            )
            for form in X_analytical
                xx = Geometry.evaluate(Forms.get_geometry(form), 1, xi_2D)
                evaluations_f = Forms.get_expression(form)(xx)
                evaluations_f_ref = deepcopy(evaluations_f)
                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_2D) approx
            end
        end
        @testset "ConstantFormSpace" begin
            geos = map(Forms.get_geometry, X_2D)
            X_const = map(
                Forms.ConstantFormSpace,
                (0, 2),
                (first(geos), last(geos)),
                ("0", "2"),
                (Forms.ComponentWisePullback, Forms.ComponentWisePullback),
            )
            for form in X_const
                evaluations_f = [ones(Float64, Points.get_num_points(xi_2D), 1)]
                evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_2D), 1)]

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_2D) approx
            end
        end
    end

    @testset "3D: F(x,y) = (2x^2+y^3-0.25z^4, 3x^3-y^5+2z^2, x+2y+3z)" begin
        function mapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            z = xx[3]
            return SVector{3}(2*x^2+y^3-0.25*z^4, 3*x^3-y^5+2*z^2, x+2*y+3*z)
        end

        function dmapping(xx::AbstractVector)
            x = xx[1]
            y = xx[2]
            z = xx[3]
            # Note: SMatrix creates the matrix per column.
            return SMatrix{3, 3}(
                4*x+y^3-0.25*z^4,
                9*x^2-y^5+2*z^2,
                1+2*y+3*z,
                2*x^2+3*y^2-0.25*z^4,
                3*x^3-5*y^4+2*z^2,
                x+2+3*z,
                2*x^2+y^3-z^3,
                3*x^3-y^5+2*2*z,
                x+2*y+3,
            )
        end
        poly_mapping = Geometry.Mapping((3, 3), mapping, dmapping)

        B1 = FunctionSpaces.create_bspline_space(0.0, 1.0, 4, 3, 2)
        B = FunctionSpaces.TensorProductSpace((B1, B1, B1), poly_mapping)
        zero_form_space = Forms.FormSpace(0, B, "0", Forms.ComponentWisePullback)
        one_form_space = Forms.FormSpace(
            1, FunctionSpaces.DirectSumSpace((B, B, B)), "1", Forms.ComponentWisePullback
        )
        two_form_space = Forms.FormSpace(
            2, FunctionSpaces.DirectSumSpace((B, B, B)), "2", Forms.ComponentWisePullback
        )
        three_form_space = Forms.FormSpace(3, B, "3", Forms.ComponentWisePullback)
        X_3D = (zero_form_space, one_form_space, two_form_space, three_form_space)
        num_bases = map(Forms.get_num_basis, X_3D)
        coeffs = map(ones, num_bases)
        X_3D_fields = map(Forms.FormField, X_3D, coeffs)

        @testset "FormSpaces" begin
            for form in X_3D
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(form), 1, xi_3D, 0
                )[1][1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)
                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_3D) approx
            end
        end

        @testset "FormFields" begin
            for form in X_3D_fields
                evaluations_fe = FunctionSpaces.evaluate(
                    Forms.get_fe_space(Forms.get_form(form)),
                    1,
                    xi_3D,
                    0,
                    Forms.get_coefficients(form),
                )[1][1]
                evaluations_fe_ref = deepcopy(evaluations_fe)

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Parametric,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_fe_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_fe, pb, k, xi_3D) approx
            end
        end

        @testset "AnalyticalFormFields" verbose=true begin
            geos = map(Forms.get_geometry, X_3D)
            X_analytical = map(
                Forms.AnalyticalFormField,
                (0, 1, 2, 3),
                (one_func_1c, one_func_3c, one_func_3c, one_func_1c),
                geos,
                ("0", "1", "2", "3"),
                (Forms.ComponentWisePullback, Forms.ComponentWisePullback),
            )
            for form in X_analytical
                x = Geometry.evaluate(Forms.get_geometry(form), 1, xi_3D)
                evaluations_f = Forms.get_expression(form)(x)
                evaluations_f_ref = deepcopy(evaluations_f)

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_3D) approx
            end
        end
        @testset "ConstantFormSpace" begin
            geos = map(Forms.get_geometry, X_3D)
            X_const = map(
                Forms.ConstantFormSpace,
                (0, 3),
                (first(geos), last(geos)),
                ("0", "3"),
                (Forms.ComponentWisePullback, Forms.ComponentWisePullback),
            )
            for form in X_const
                evaluations_f = [ones(Float64, Points.get_num_points(xi_3D), 1)]
                evaluations_f_ref = [ones(Float64, Points.get_num_points(xi_3D), 1)]

                pb = Forms.get_pullback(form)

                @test Forms.get_pullback_type(form) <: Forms.ComponentWisePullback
                @test pb isa Forms.ComponentWisePullback{
                    Forms.get_manifold_dim(form),
                    Forms.get_form_rank(form),
                    Forms.get_expression_rank(form),
                    Forms.Physical,
                    Forms.Canonical,
                }

                expected_pullback = Dict(1 => evaluations_f_ref)
                @test @expected expected_pullback k ->
                    Forms.pullback!(evaluations_f, pb, k, xi_3D) approx
            end
        end
    end
end

end

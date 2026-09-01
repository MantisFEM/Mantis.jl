module RefinementTests

using Mantis
using Test

const PLOT = false

const ref_p1(s) = FunctionSpaces.refinement_degree(s, 1)
const ref_h2(s) = FunctionSpaces.refinement_uniform(s, 2)

@testset "Uniform" verbose = true begin
    struct NewSpace{manifold_dim, num_components, num_patches} <:
           FunctionSpaces.AbstractFESpace{manifold_dim, num_components, num_patches} end

    @test_throws MethodError FunctionSpaces.refinement_uniform(NewSpace{1, 1, 1}, (1,))

    @testset "KnotVector" verbose = true begin
        cart_1d = Geometry.create_cartesian_box((0.0,), (1.0,), (2,))
        knot_1 = FunctionSpaces.KnotVector(cart_1d, 3, [4, 2, 4])
        knot_2 = FunctionSpaces.refinement_uniform(knot_1, 2)
        @test knot_2.multiplicity == [4, 1, 2, 1, 4]
    end

    @testset "B-Splines" verbose = true begin
        bsp_1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (2,), (1,))
        bsp_2 = Hierarchical.Refinement(bsp_1, ref_h2)()
        @test FunctionSpaces.get_num_elements(bsp_2) == 4
        @test FunctionSpaces.get_num_basis(bsp_2) == 6

        coeffs_1 = [0.5, -1.0, 0.75]
        eval_1_1 = FunctionSpaces.evaluate(
            bsp_1, 1, Points.PointSet((LinRange(0, 0.5, 5)))
        )[1][1][1][1]
        for c in axes(eval_1_1, 2)
            eval_1_1[:, c] .*= coeffs_1[c]
        end
        sol_1_1 = sum(eval_1_1; dims=2)

        eval_2_1 = FunctionSpaces.evaluate(
            bsp_2, 1, Points.PointSet((LinRange(0, 1.0, 5)))
        )[1][1][1][1]
        coeffs_2_1 = eval_2_1 \ sol_1_1
        subdiv_2_1 = [
            1.0 0.0 0.0
            0.5 0.5 0.0
            0.0 0.75 0.25
        ]
        @test isapprox(coeffs_2_1, subdiv_2_1 * coeffs_1)

        eval_2_2 = FunctionSpaces.evaluate(
            bsp_2, 1, Points.PointSet((LinRange(0, 1.0, 5)))
        )[1][1][1][1]
        eval_1_2 = FunctionSpaces.evaluate(
            bsp_1, 1, Points.PointSet((LinRange(0.5, 1.0, 5)))
        )[1][1][1][1]
        for c in axes(eval_1_2, 2)
            eval_1_2[:, c] .*= coeffs_1[c]
        end
        sol_1_2 = sum(eval_1_2; dims=2)
        eval_2_2 = FunctionSpaces.evaluate(
            bsp_2, 2, Points.PointSet((LinRange(0, 1.0, 5)))
        )[1][1][1][1]
        coeffs_2_2 = eval_2_2 \ sol_1_2
        subdiv_2_2 = [
            0.5 0.5 0.0
            0.0 0.75 0.25
            0.0 0.25 0.75
        ]
        @test isapprox(coeffs_2_2, subdiv_2_2 * coeffs_1)

        if PLOT
            using GLMakie
            using Makie

            display(Mantis.Plot.plot_basis(bsp_1))
            display(Mantis.Plot.plot_basis(bsp_2))
        end
    end
end

@testset "Degree-Elevation" verbose = true begin
    @test_throws MethodError FunctionSpaces.refinement_degree(NewSpace{1, 1, 1}, 1)

    @testset "KnotVector" verbose = true begin
        cart_1d = Geometry.create_cartesian_box((0.0,), (1.0,), (2,))
        knot_1 = FunctionSpaces.KnotVector(cart_1d, 3, [4, 2, 4])
        knot_2 = FunctionSpaces.refinement_degree(knot_1, 2)
        @test FunctionSpaces.get_geometry(knot_1) === FunctionSpaces.get_geometry(knot_2)
        @test FunctionSpaces.get_polynomial_degree(knot_2) == 5
        @test FunctionSpaces.get_multiplicity(knot_2) == [6, 4, 6]
    end

    @testset "B-Splines" verbose = true begin
        bsp_1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (2,), (1,))
        ref_p1(s) = FunctionSpaces.refinement_degree(s, 1)
        bsp_2 = Hierarchical.Refinement(bsp_1, ref_p1)()
        @test FunctionSpaces.get_num_elements(bsp_2) == 2
        @test FunctionSpaces.get_polynomial_degree(bsp_2) == 3
        @test FunctionSpaces.get_num_basis(bsp_2) == 6

        if PLOT
            using GLMakie
            using Makie

            display(Mantis.Plot.plot_basis(bsp_1))
            display(Mantis.Plot.plot_basis(bsp_2))
        end
    end
end

@testset "Composition" verbose = true begin
    @testset "B-Splines" verbose = true begin
        bsp_1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (2,), (1,))
        ref_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
        ref_p1(s) = FunctionSpaces.refinement_degree(s, 1)
        bsp_2_1 = Hierarchical.Refinement(bsp_1, s -> s |> ref_h2 |> ref_p1)()
        bsp_2_2 = Hierarchical.Refinement(bsp_1, s -> s |> ref_p1 |> ref_h2)()
        @test FunctionSpaces.get_num_elements(bsp_2_1) == 4
        @test FunctionSpaces.get_num_elements(bsp_2_2) == 4
        @test FunctionSpaces.get_polynomial_degree(bsp_2_1) == 3
        @test FunctionSpaces.get_polynomial_degree(bsp_2_2) == 3
        @test FunctionSpaces.get_num_basis(bsp_2_1) == 10
        @test FunctionSpaces.get_num_basis(bsp_2_2) == 8

        if PLOT
            using GLMakie
            using Makie

            display(Mantis.Plot.plot_basis(bsp_1))
            display(Mantis.Plot.plot_basis(bsp_2_1))
            display(Mantis.Plot.plot_basis(bsp_2_2))
        end
    end
end

@testset "Tensor-Product" verbose = true begin
    @testset "B-Splines" verbose = true begin
        ref_p1(s) = FunctionSpaces.refinement_degree(s, 1)
        ref_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
        bsp_1 = FunctionSpaces.create_bspline_space(
            (0.0, 0.0), (1.0, 1.0), (2, 1), (1, 2), (0, 1)
        )
        bsp_2 = Hierarchical.Refinement(bsp_1, (ref_p1, ref_h2))()
        geo_2 = FunctionSpaces.get_geometry(bsp_2)
        @test Geometry.get_factor_num_elements(geo_2) == (2, 2)
        @test FunctionSpaces.get_factor_polynomial_degrees(bsp_2) == (2, 2)
        @test FunctionSpaces.get_factor_num_basis(bsp_2) == (5, 4)

        if PLOT
            using GLMakie
            using Makie

            display(Mantis.Plot.plot_basis(bsp_1))
            display(Mantis.Plot.plot_basis(bsp_2))
        end
    end
end

end

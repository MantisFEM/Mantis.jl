module ScalingsTests

using Mantis
using Test
using LinearAlgebra

ref_h2(s) = FunctionSpaces.refinement_uniform(s, 2)

@testset "Uniform" verbose = true begin
    @testset "B-Splines" verbose = true begin
        @testset "Knot Insertion" verbose = true begin
            bsp_1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (2,), (1,))
            bsp_2 = Hierarchical.Refinement(bsp_1, ref_h2)()
            scal = Hierarchical.MatrixScaling(
                bsp_1, bsp_2, FunctionSpaces.scaling_matrix_uniform
            )

            @test isapprox(
                Hierarchical.get_scaling_matrix(scal),
                [
                    1.0 0.0 0.0 0.0
                    0.5 0.5 0.0 0.0
                    0.0 0.75 0.25 0.0
                    0.0 0.25 0.75 0.0
                    0.0 0.0 0.5 0.5
                    0.0 0.0 0.0 1.0
                ],
            )

            @test Hierarchical.get_children(scal, 1) == [1, 2]
            @test Hierarchical.get_children(scal, 2) == [2, 3, 4]
            @test Hierarchical.get_children(scal, 3) == [3, 4, 5]
            @test Hierarchical.get_children(scal, 4) == [5, 6]

            @test Hierarchical.get_parents(scal, 1) == [1]
            @test Hierarchical.get_parents(scal, 2) == [1, 2]
            @test Hierarchical.get_parents(scal, 3) == [2, 3]
            @test Hierarchical.get_parents(scal, 4) == [2, 3]
            @test Hierarchical.get_parents(scal, 5) == [3, 4]
            @test Hierarchical.get_parents(scal, 6) == [4]

            # Oslo is degree-preserving, so the child must keep the polynomial degree.
            bsp_1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (2,), (1,))
            bsp_1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (3,), (1,))
            @test_throws ArgumentError Hierarchical.MatrixScaling(
                bsp_1, bsp_2, FunctionSpaces.scaling_matrix_uniform
            )
        end
    end
end

@testset "Degree-Elevation" verbose = true begin
    p = 2
    delta = 1
    num_elements = 2
    bsp_1 = FunctionSpaces.create_bspline_space(
        (0.0,), (1.0,), (num_elements,), (p,), (p - 1,)
    )
    ref_p(s) = FunctionSpaces.refinement_degree(s, delta)
    bsp_2 = Hierarchical.Refinement(bsp_1, ref_p)()

    # General scaling matrix
    qr = Quadrature.get_global_quadrature_rules(
        Quadrature.gauss_legendre, num_elements, (p + delta + 1,)
    )[1]
    canonical_scaling_matrix = FunctionSpaces.scaling_matrix_degree(
        FunctionSpaces.get_polynomials(bsp_1), delta
    )
    scal_general = Hierarchical.MatrixScaling(
        bsp_1,
        bsp_2,
        (bsp_1, bsp_2) -> FunctionSpaces.scaling_matrix_approximate(
            bsp_1, bsp_2, canonical_scaling_matrix; tol=1e-14
        ),
    )
    # Degree elevation matrix
    scal_degree = Hierarchical.MatrixScaling(
        bsp_1,
        bsp_2,
        (bsp_1, bsp_2) -> FunctionSpaces.scaling_matrix_degree(bsp_1, bsp_2, delta),
    )

    scal_mat_general = Hierarchical.get_scaling_matrix(scal_general)
    scal_mat_degree = Hierarchical.get_scaling_matrix(scal_degree)
    @test isapprox(scal_mat_general, scal_mat_degree)

    p = 3
    delta = 3
    num_elements = 4
    bsp_1 = FunctionSpaces.create_bspline_space(
        (0.0,), (1.0,), (num_elements,), (p,), (p - 1,)
    )
    bsp_2 = Hierarchical.Refinement(bsp_1, ref_p)()

    # General scaling matrix
    qr = Quadrature.get_global_quadrature_rules(
        Quadrature.gauss_legendre, num_elements, (p + delta + 1,)
    )[1]
    scal_general = Hierarchical.MatrixScaling(
        bsp_1,
        bsp_2,
        (bsp_1, bsp_2) -> Assemblers.scaling_matrix_general(bsp_1, bsp_2, qr; tol=1e-14),
    )
    # Degree elevation matrix
    scal_degree = Hierarchical.MatrixScaling(
        bsp_1,
        bsp_2,
        (bsp_1, bsp_2) -> FunctionSpaces.scaling_matrix_degree(bsp_1, bsp_2, delta),
    )

    scal_mat_general = Hierarchical.get_scaling_matrix(scal_general)
    scal_mat_degree = Hierarchical.get_scaling_matrix(scal_degree)
    @test isapprox(scal_mat_general, scal_mat_degree)
end

@testset "Approximate" verbose = true begin
    p = 2
    num_elements = 4
    bsp_1 = FunctionSpaces.create_bspline_space(
        (0.0,), (1.0,), (num_elements,), (p,), (p - 1,)
    )
    scal_approx = FunctionSpaces.scaling_matrix_approximate(
        bsp_1, bsp_1, Matrix(LinearAlgebra.I, 3, 3)
    )
    @test isapprox(scal_approx, Matrix(LinearAlgebra.I, 6, 6))

    # vector should have length 4
    @test_throws ArgumentError FunctionSpaces.scaling_matrix_approximate(
        bsp_1, bsp_1, [Matrix(LinearAlgebra.I, 3, 3)]
    )

    # child should have more dofs
    bsp_2 = FunctionSpaces.create_bspline_space(
        (0.0,), (1.0,), (num_elements .* 2,), (p,), (p - 1,)
    )
    @test_throws ArgumentError FunctionSpaces.scaling_matrix_approximate(
        bsp_2, bsp_1, Matrix(LinearAlgebra.I, 3, 3)
    )

    canonical_scaling_matrix = FunctionSpaces.scaling_matrix_uniform(
        FunctionSpaces.get_polynomials(bsp_1), 2
    )
    scal_approx = FunctionSpaces.scaling_matrix_approximate(
        bsp_1, bsp_2, canonical_scaling_matrix
    )
    scal_uni = FunctionSpaces.scaling_matrix_uniform(bsp_1, bsp_2)
    @test isapprox(scal_approx, scal_uni)
end

@testset "Composition" verbose = true begin
    @testset "B-Splines" verbose = true begin
        p = 2
        delta = 2
        num_elements = 4
        bsp_1 = FunctionSpaces.create_bspline_space(
            (0.0,), (1.0,), (num_elements,), (p,), (p - 1,)
        )
        ref_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
        ref_p(s) = FunctionSpaces.refinement_degree(s, delta)
        bsp_2 = Hierarchical.Refinement(bsp_1, ref_h2)()
        bsp_3 = Hierarchical.Refinement(bsp_2, ref_p)()
        scal = Hierarchical.MatrixScaling(
            (bsp_1, bsp_2, bsp_3),
            FunctionSpaces.scaling_matrix_uniform,
            (bsp_2, bsp_3) -> FunctionSpaces.scaling_matrix_degree(bsp_2, bsp_3, delta),
        )
        scal_mat = Hierarchical.get_scaling_matrix(scal)
        scal_h2 = Hierarchical.MatrixScaling(
            bsp_1, bsp_2, FunctionSpaces.scaling_matrix_uniform
        )
        scal_p = Hierarchical.MatrixScaling(
            bsp_2,
            bsp_3,
            (bsp_2, bsp_3) -> FunctionSpaces.scaling_matrix_degree(bsp_2, bsp_3, delta),
        )
        scal_h2_mat = Hierarchical.get_scaling_matrix(scal_h2)
        scal_p_mat = Hierarchical.get_scaling_matrix(scal_p)
        @test isapprox(scal_mat, scal_p_mat * scal_h2_mat)

        # Not enough scaling functions
        @test_throws ArgumentError Hierarchical.MatrixScaling(
            (bsp_1, bsp_2, bsp_3), FunctionSpaces.scaling_matrix_uniform
        )
    end
end

@testset "Tensor-Product" verbose = true begin
    bsp_1 = FunctionSpaces.create_bspline_space(
        (0.0, 0.0), (1.0, 1.0), (2, 1), (1, 2), (0, 1)
    )
    ref_p1(s) = FunctionSpaces.refinement_degree(s, 1)
    ref_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
    bsp_2 = Hierarchical.Refinement(bsp_1, (ref_p1, ref_h2))()
    scal_p1 = (p, c) -> FunctionSpaces.scaling_matrix_degree(p, c, 1)
    scal_h2 = FunctionSpaces.scaling_matrix_uniform
    scal = Hierarchical.MatrixScaling(bsp_1, bsp_2, (scal_p1, scal_h2))
    scal_mat = Hierarchical.get_scaling_matrix(scal)
    bsp_1_1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (1,), (0,))
    bsp_1_2 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (1,), (2,), (1,))
    bsp_2_1 = Hierarchical.Refinement(bsp_1_1, ref_p1)()
    bsp_2_2 = Hierarchical.Refinement(bsp_1_2, ref_h2)()
    scal_mat_1 = scal_p1(bsp_1_1, bsp_2_1)
    scal_mat_2 = scal_h2(bsp_1_2, bsp_2_2)
    @test isapprox(scal_mat, kron(scal_mat_2, scal_mat_1))
end

end

module ExtractionTests

using Mantis
using Mantis.Hierarchical
using Test

const PLOT = false

ref_space_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
ref_geo_h2(s) = Geometry.refinement_uniform(s, 2)
ref_space_p1(s) = FunctionSpaces.refinement_degree(s, 1)
scal_space_p1(p, c) = FunctionSpaces.scaling_matrix_degree(p, c, 1)
scal_space_h2 = FunctionSpaces.scaling_matrix_uniform
const SelectionStandard = FunctionSpaces.SelectionStandard
const HB = FunctionSpaces.HB
const THB = FunctionSpaces.THB

const xi = Points.PointSet((LinRange(0, 1, 6),))

############################################################################################
#                                       Test Helpers                                       #
############################################################################################

function extraction_coefficients(space, expected)
    for (e, coeffs) in pairs(expected)
        computed_coeffs = FunctionSpaces.get_extraction_coefficients(space, e)
        if !(computed_coeffs ≈ coeffs)
            println("Element $(e).\nExpected")
            display(coeffs)
            println("but got")
            display(computed_coeffs)
            return false
        end
    end

    return true
end

function basis_indices(space, expected)
    for (e, indices) in pairs(expected)
        computed_indices = FunctionSpaces.get_basis_indices(space, e)
        if !(FunctionSpaces.get_basis_indices(space, e) == indices)
            println("Element $(e).\nExpected")
            println(indices)
            println("but got")
            println(computed_indices)
            return false
        end
    end

    return true
end

function support(space, expected)
    for (b, supp) in pairs(expected)
        computed_supp = FunctionSpaces.get_support(space, b)
        if !(FunctionSpaces.get_support(space, b) == supp)
            println("Basis Function $(b).\nExpected")
            println(supp)
            println("but got")
            println(computed_supp)
            return false
        end
    end

    return true
end

function partition_of_unity(space)
    for e in 1:FunctionSpaces.get_num_elements(space)
        eval = FunctionSpaces.evaluate(space, e, xi)[1][1][1][1]
        if !all(p -> isapprox(p, 1.0), sum(eval; dims=2))
            return false
        end
    end

    return true
end

############################################################################################
#                                          Tests                                           #
############################################################################################

@testset "Uniform 1D" verbose = true begin
    # Level Geometries
    geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))
    geo_l2 = Refinement(geo_l1, ref_geo_h2)()
    geo_l3 = Refinement(geo_l2, ref_geo_h2)()
    relations = Relations(
        Geometry.parent_to_children_uniform(2), Geometry.child_to_parents_uniform(2)
    )
    geo_scal_1 = Scaling(geo_l1, geo_l2, relations)
    geo_scal_2 = Scaling(geo_l2, geo_l3, relations)

    #=
    GEOMETRY 1

    x inactive
    - active

    Level 1: |-------|-------|xxxxxxx|xxxxxxx|
    Level 2: |xxx|xxx|xxx|xxx|---|---|xxx|xxx|
    Level 3: |x|x|x|x|x|x|x|x|x|x|x|x|-|-|-|-|
    
    Final G: |-------|-------|---|---|-|-|-|-|
    =#
    active = ActiveInfo([[1, 2], [5, 6], [13, 14, 15, 16]])
    hgeo1 = Geometry.HierarchicalGeometry(NestedHierarchy(active, geo_scal_1, geo_scal_2))

    #=
    GEOMETRY 2

    x inactive
    - active

    Level 1: |-------|-------|xxxxxxx|xxxxxxx|
    Level 2: |xxx|xxx|xxx|xxx|xxx|---|xxx|xxx|
    Level 3: |x|x|x|x|x|x|x|x|-|-|x|x|-|-|-|-|
    
    Final G: |-------|-------|-|-|---|-|-|-|-|
    =#
    active = ActiveInfo([[1, 2], [6], [9, 10, 13, 14, 15, 16]])
    hgeo2 = Geometry.HierarchicalGeometry(NestedHierarchy(active, geo_scal_1, geo_scal_2))
    @testset "Linear" verbose = true begin
        # FESpaces
        bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (1,), (0,))
        bsp_l2 = Refinement(bsp_l1, ref_space_h2)()
        bsp_l3 = Refinement(bsp_l2, ref_space_h2)()
        bsp_scal_1 = MatrixScaling(bsp_l1, bsp_l2, scal_space_h2)
        bsp_scal_2 = MatrixScaling(bsp_l2, bsp_l3, scal_space_h2)
        scalings = (bsp_scal_1, bsp_scal_2)

        @testset "HB" verbose = true begin
            # GEOMETRY 1

            hgeo = hgeo1
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, HB
            )

            coeffs_expected = Dict(
                1 => [1.0 0.0; 0.0 1.0],
                2 => [1.0 0.0; 0.0 1.0],
                3 => [0.0 1.0; 1.0 0.5],
                4 => [1.0 0.0 0.5; 0.0 1.0 0.0],
                5 => [0.0 1.0; 1.0 0.5],
                6 => [1.0 0.0 0.5; 0.0 1.0 0.0],
                7 => [1.0 0.0; 0.0 1.0],
                8 => [1.0 0.0; 0.0 1.0],
            )

            basis_expected = Dict(
                1 => [3, 1],
                2 => [1, 2],
                3 => [4, 2],
                4 => [4, 5, 2],
                5 => [8, 5],
                6 => [8, 6, 5],
                7 => [6, 7],
                8 => [7, 9],
            )

            support_expected = Dict(
                1 => [1, 2],
                2 => [2, 3, 4],
                3 => [1],
                4 => [3, 4],
                5 => [4, 5, 6],
                6 => [6, 7],
                7 => [7, 8],
                8 => [5, 6],
                9 => [8],
            )

            @test extraction_coefficients(space, coeffs_expected)
            @test basis_indices(space, basis_expected)
            @test support(space, support_expected)

            if PLOT
                using GLMakie, Makie

                display(Mantis.Plot.plot_basis(space; plot_points_per_element=2))
                readline()
            end

            # GEOMETRY 2

            hgeo = hgeo2
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, HB
            )

            coeffs_expected = Dict(
                1 => [1.0 0.0; 0.0 1.0],
                2 => [1.0 0.0; 0.0 1.0],
                3 => [1.0 0.0 0.5; 0.0 1.0 0.0],
                4 => [0.0 0.0 1.0; 1.0 0.5 0.75],
                5 => [1.0 0.5 0.75; 0.0 1.0 0.5],
                6 => [0.0 1.0; 1.0 0.5],
                7 => [1.0 0.0 0.5; 0.0 1.0 0.0],
                8 => [1.0 0.0; 0.0 1.0],
                9 => [1.0 0.0; 0.0 1.0],
            )

            basis_expected = Dict(
                1 => [3, 1],
                2 => [1, 2],
                3 => [4, 5, 2],
                4 => [8, 4, 2],
                5 => [8, 4, 2],
                6 => [9, 5],
                7 => [9, 6, 5],
                8 => [6, 7],
                9 => [7, 10],
            )

            support_expected = Dict(
                1 => [1, 2],
                2 => [2, 3, 4, 5],
                3 => [1],
                4 => [4, 5, 3],
                5 => [3, 6, 7],
                6 => [7, 8],
                7 => [8, 9],
                8 => [4, 5],
                9 => [6, 7],
                10 => [9],
            )

            @test extraction_coefficients(space, coeffs_expected)
            @test basis_indices(space, basis_expected)
            @test support(space, support_expected)

            if PLOT
                using GLMakie, Makie

                display(Mantis.Plot.plot_basis(space; plot_points_per_element=2))
                readline()
            end
        end

        @testset "THB" verbose = true begin
            # GEOMETRY 1

            hgeo = hgeo1
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, THB
            )

            coeffs_expected = Dict(
                1 => [1.0 0.0; 0.0 1.0],
                2 => [1.0 0.0; 0.0 1.0],
                3 => [0.0 1.0; 1.0 0.0],
                4 => [1.0 0.0; 0.0 1.0],
                5 => [0.0 1.0; 1.0 0.0],
                6 => [1.0 0.0; 0.0 1.0],
                7 => [1.0 0.0; 0.0 1.0],
                8 => [1.0 0.0; 0.0 1.0],
            )

            basis_expected = Dict(
                1 => [3, 1],
                2 => [1, 2],
                3 => [4, 2],
                4 => [4, 5],
                5 => [8, 5],
                6 => [8, 6],
                7 => [6, 7],
                8 => [7, 9],
            )

            support_expected = Dict(
                1 => [1, 2],
                2 => [2, 3],
                3 => [1],
                4 => [3, 4],
                5 => [4, 5],
                6 => [6, 7],
                7 => [7, 8],
                8 => [5, 6],
                9 => [8],
            )

            @test extraction_coefficients(space, coeffs_expected)
            @test basis_indices(space, basis_expected)
            @test support(space, support_expected)
            @test partition_of_unity(space)

            if PLOT
                using GLMakie, Makie

                display(Mantis.Plot.plot_basis(space; plot_points_per_element=2))
                readline()
            end

            # GEOMETRY 2

            hgeo = hgeo2
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, THB
            )

            ## Extraction
            coeffs_expected = Dict(
                1 => [1.0 0.0; 0.0 1.0],
                2 => [1.0 0.0; 0.0 1.0],
                3 => [1.0 0.0; 0.0 1.0],
                4 => [0.0 1.0; 1.0 0.0],
                5 => [1.0 0.0; 0.0 1.0],
                6 => [0.0 1.0; 1.0 0.0],
                7 => [1.0 0.0; 0.0 1.0],
                8 => [1.0 0.0; 0.0 1.0],
                9 => [1.0 0.0; 0.0 1.0],
            )

            basis_expected = Dict(
                1 => [3, 1],
                2 => [1, 2],
                3 => [4, 5],
                4 => [8, 2],
                5 => [8, 4],
                6 => [9, 5],
                7 => [9, 6],
                8 => [6, 7],
                9 => [7, 10],
            )

            ## Support
            support_expected = Dict(
                1 => [1, 2],
                2 => [2, 4],
                3 => [1],
                4 => [5, 3],
                5 => [3, 6],
                6 => [7, 8],
                7 => [8, 9],
                8 => [4, 5],
                9 => [6, 7],
                10 => [9],
            )

            @test extraction_coefficients(space, coeffs_expected)
            @test basis_indices(space, basis_expected)
            @test support(space, support_expected)
            @test partition_of_unity(space)

            if PLOT
                using GLMakie, Makie

                display(Mantis.Plot.plot_basis(space; plot_points_per_element=2))
                readline()
            end
        end
    end

    @testset "Quadratic" verbose = true begin
        # FESpaces
        bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (2,), (1,))
        bsp_l2 = Refinement(bsp_l1, ref_space_h2)()
        bsp_l3 = Refinement(bsp_l2, ref_space_h2)()
        bsp_scal_1 = MatrixScaling(bsp_l1, bsp_l2, scal_space_h2)
        bsp_scal_2 = MatrixScaling(bsp_l2, bsp_l3, scal_space_h2)
        scalings = (bsp_scal_1, bsp_scal_2)

        @testset "HB" verbose = true begin
            # GEOMETRY 1

            hgeo = hgeo1
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, HB
            )

            coeffs_expected = Dict(
                1 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                2 => [0.5 0.5 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                3 => [0.0 0.5 0.5; 0.0 0.25 0.75; 0.5 0.125 0.75],
                4 => [0.5 0.0 0.125 0.75; 1.0 0.0 0.0 0.75; 0.5 0.5 0.0 0.5],
                5 => [0.0 0.5 0.5 0.5; 0.0 0.25 0.75 0.375; 0.5 0.125 0.75 0.28125],
                6 => [
                    0.5 0.0 0.125 0.75 0.28125;
                    1.0 0.0 0.0 0.75 0.1875;
                    0.5 0.5 0.0 0.5 0.125
                ],
                7 => [
                    0.5 0.5 0.0 0.5 0.125;
                    0.0 1.0 0.0 0.25 0.0625;
                    0.0 0.5 0.5 0.125 0.03125
                ],
                8 => [0.5 0.5 0.0 0.125 0.03125; 0.0 1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0 0.0],
            )
            basis_expected = Dict(
                1 => [4, 2, 3],
                2 => [2, 3, 1],
                3 => [5, 3, 1],
                4 => [5, 6, 3, 1],
                5 => [7, 5, 6, 1],
                6 => [7, 8, 5, 6, 1],
                7 => [7, 8, 10, 6, 1],
                8 => [8, 10, 9, 6, 1],
            )
            support_expected = Dict(
                1 => [2, 3, 4, 5, 6, 7, 8],
                2 => [1, 2],
                3 => [1, 2, 3, 4],
                4 => [1],
                5 => [3, 4, 5, 6],
                6 => [4, 5, 6, 7, 8],
                7 => [5, 6, 7],
                8 => [6, 7, 8],
                9 => [8],
                10 => [7, 8],
            )

            @test extraction_coefficients(space, coeffs_expected)
            @test basis_indices(space, basis_expected)
            @test support(space, support_expected)

            if PLOT
                using GLMakie, Makie
                display(Mantis.Plot.plot_basis(space))
                readline()
            end

            # GEOMETRY 2

            hgeo = hgeo2
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, HB
            )

            coeffs_expected = Dict(
                1 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                2 => [0.5 0.5 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                3 => [0.5 0.0 0.125 0.75; 1.0 0.0 0.0 0.75; 0.5 0.5 0.0 0.5],
                4 => [0.0 0.5 0.5; 0.0 0.375 0.625; 0.125 0.28125 0.6875],
                5 => [0.125 0.28125 0.6875; 0.25 0.1875 0.75; 0.5 0.125 0.75],
                6 => [0.0 0.5 0.5 0.5; 0.0 0.25 0.75 0.375; 0.5 0.125 0.75 0.28125],
                7 => [
                    0.5 0.0 0.125 0.75 0.28125;
                    1.0 0.0 0.0 0.75 0.1875;
                    0.5 0.5 0.0 0.5 0.125
                ],
                8 => [
                    0.5 0.5 0.0 0.5 0.125;
                    0.0 1.0 0.0 0.25 0.0625;
                    0.0 0.5 0.5 0.125 0.03125
                ],
                9 => [0.5 0.5 0.0 0.125 0.03125; 0.0 1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0 0.0],
            )
            basis_expected = Dict(
                1 => [4, 2, 3],
                2 => [2, 3, 1],
                3 => [5, 6, 3, 1],
                4 => [5, 3, 1],
                5 => [5, 3, 1],
                6 => [7, 5, 6, 1],
                7 => [7, 8, 5, 6, 1],
                8 => [7, 8, 10, 6, 1],
                9 => [8, 10, 9, 6, 1],
            )
            support_expected = Dict(
                1 => [2, 3, 4, 5, 6, 7, 8, 9],
                2 => [1, 2],
                3 => [1, 2, 3, 4, 5],
                4 => [1],
                5 => [4, 5, 3, 6, 7],
                6 => [3, 6, 7, 8, 9],
                7 => [6, 7, 8],
                8 => [7, 8, 9],
                9 => [9],
                10 => [8, 9],
            )

            @test extraction_coefficients(space, coeffs_expected)
            @test basis_indices(space, basis_expected)
            @test support(space, support_expected)

            if PLOT
                using GLMakie, Makie
                display(Mantis.Plot.plot_basis(space))
                readline()
            end
        end

        @testset "THB" verbose = true begin
            # GEOMETRY 1

            hgeo = hgeo1
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, THB
            )

            coeffs_expected = Dict(
                1 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                2 => [0.5 0.5 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                3 => [0.0 0.5 0.5; 0.0 0.25 0.75; 0.5 0.125 0.375],
                4 => [0.5 0.0 0.125 0.375; 1.0 0.0 0.0 0.0; 0.5 0.5 0.0 0.0],
                5 => [0.0 0.5 0.5; 0.0 0.25 0.75; 0.5 0.125 0.375],
                6 => [0.5 0.0 0.125 0.375; 1.0 0.0 0.0 0.0; 0.5 0.5 0.0 0.0],
                7 => [0.5 0.5 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                8 => [0.5 0.5 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            )

            basis_expected = Dict(
                1 => [4, 2, 3],
                2 => [2, 3, 1],
                3 => [5, 3, 1],
                4 => [5, 6, 3, 1],
                5 => [7, 5, 6],
                6 => [7, 8, 5, 6],
                7 => [7, 8, 10],
                8 => [8, 10, 9],
            )

            support_expected = Dict(
                1 => [2, 3, 4],
                2 => [1, 2],
                3 => [1, 2, 3, 4],
                4 => [1],
                5 => [3, 4, 5, 6],
                6 => [4, 5, 6],
                7 => [5, 6, 7],
                8 => [6, 7, 8],
                9 => [8],
                10 => [7, 8],
            )

            @test extraction_coefficients(space, coeffs_expected)
            @test basis_indices(space, basis_expected)
            @test support(space, support_expected)
            @test partition_of_unity(space)

            if PLOT
                using GLMakie, Makie
                display(Mantis.Plot.plot_basis(space))
                readline()
            end

            # GEOMETRY 2

            hgeo = hgeo2
            space = FunctionSpaces.HierarchicalSpace(
                hgeo, hgeo, scalings, SelectionStandard, THB
            )

            coeffs_expected = Dict(
                1 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                2 => [0.5 0.5 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                3 => [0.5 0.0 0.125 0.375; 1.0 0.0 0.0 0.0; 0.5 0.5 0.0 0.0],
                4 => [0.0 0.5 0.5; 0.0 0.375 0.625; 0.125 0.28125 0.59375],
                5 => [0.125 0.28125 0.59375; 0.25 0.1875 0.5625; 0.5 0.125 0.375],
                6 => [0.0 0.5 0.5; 0.0 0.25 0.75; 0.5 0.125 0.375],
                7 => [0.5 0.0 0.125 0.375; 1.0 0.0 0.0 0.0; 0.5 0.5 0.0 0.0],
                8 => [0.5 0.5 0.0; 0.0 1.0 0.0; 0.0 0.5 0.5],
                9 => [0.5 0.5 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            )

            basis_expected = Dict(
                1 => [4, 2, 3],
                2 => [2, 3, 1],
                3 => [5, 6, 3, 1],
                4 => [5, 3, 1],
                5 => [5, 3, 1],
                6 => [7, 5, 6],
                7 => [7, 8, 5, 6],
                8 => [7, 8, 10],
                9 => [8, 10, 9],
            )

            support_expected = Dict(
                1 => [2, 3, 4, 5],
                2 => [1, 2],
                3 => [1, 2, 3, 4, 5],
                4 => [1],
                5 => [4, 5, 3, 6, 7],
                6 => [3, 6, 7],
                7 => [6, 7, 8],
                8 => [7, 8, 9],
                9 => [9],
                10 => [8, 9],
            )

            @test extraction_coefficients(space, coeffs_expected)
            @test basis_indices(space, basis_expected)
            @test support(space, support_expected)
            @test partition_of_unity(space)

            if PLOT
                using GLMakie, Makie
                display(Mantis.Plot.plot_basis(space))
                readline()
            end
        end
    end
end

@testset "Degree-elevation 1D" verbose = true begin
    # Level Geometries
    geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 9),))
    relations = Relations(
        RelationExplicit{Hierarchical.PC}(i -> (i,)),
        RelationExplicit{Hierarchical.CP}(i -> (i,)),
    )
    geo_scal_1 = Scaling(geo_l1, geo_l1, relations)

    #=
    GEOMETRY 1

    x inactive
    - active

    Level 1: |-------|-------|xxxxxxx|xxxxxxx|xxxxxxx|xxxxxxx|xxxxxxx|xxxxxxx|
    Level 2: |xxxxxxx|xxxxxxx|-------|-------|-------|-------|-------|-------|
    Level 3: |xxxxxxx|xxxxxxx|xxxxxxx|xxxxxxx|-------|-------|-------|-------|
    
    Final G: |-------|-------|-------|-------|-------|-------|-------|
    =#
    active = ActiveInfo([[1, 2], [3, 4], [5, 6, 7, 8]])
    hgeo1 = Geometry.HierarchicalGeometry(NestedHierarchy(active, geo_scal_1, geo_scal_1))

    # FESpaces
    bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (8,), (1,), (0,))
    bsp_l2 = Refinement(bsp_l1, ref_space_p1)()
    bsp_l3 = Refinement(bsp_l2, ref_space_p1)()
    bsp_scal_1 = MatrixScaling(bsp_l1, bsp_l2, scal_space_p1)
    bsp_scal_2 = MatrixScaling(bsp_l2, bsp_l3, scal_space_p1)
    scalings = (bsp_scal_1, bsp_scal_2)

    @testset "HB" verbose = true begin
        hgeo = hgeo1
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, HB
        )

        coeffs_expected = Dict(
            1 => [1.0 0.0; 0.0 1.0],
            2 => [1.0 0.0; 0.0 1.0],
            3 => [0.0 0.0 1.0; 1.0 0.0 0.5; 0.0 1.0 0.0],
            4 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            5 => [
                0.0 0.0 0.0 1.0;
                1.0 0.0 0.0 0.3333333333333333;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0
            ],
            6 => [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0],
            7 => [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0],
            8 => [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0],
        )

        basis_expected = Dict(
            1 => [3, 1],
            2 => [1, 2],
            3 => [5, 7, 2],
            4 => [7, 4, 6],
            5 => [15, 17, 8, 6],
            6 => [8, 11, 19, 13],
            7 => [13, 9, 18, 14],
            8 => [14, 12, 10, 16],
        )

        support_expected = Dict(
            1 => [1, 2],
            2 => [2, 3],
            3 => [1],
            4 => [4],
            5 => [3],
            6 => [4, 5],
            7 => [3, 4],
            8 => [5, 6],
            9 => [7],
            10 => [8],
            11 => [6],
            12 => [8],
            13 => [6, 7],
            14 => [7, 8],
            15 => [5],
            16 => [8],
            17 => [5],
            18 => [7],
            19 => [6],
        )

        @test extraction_coefficients(space, coeffs_expected)
        @test basis_indices(space, basis_expected)
        @test support(space, support_expected)

        if PLOT
            using GLMakie, Makie
            display(Mantis.Plot.plot_basis(space))
            readline()
        end
    end

    @testset "THB" verbose = true begin
        hgeo = hgeo1
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, THB
        )

        coeffs_expected = Dict(
            1 => [1.0 0.0; 0.0 1.0],
            2 => [1.0 0.0; 0.0 1.0],
            3 => [0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 0.0],
            4 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            5 => [0.0 0.0 0.0 1.0; 1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0],
            6 => [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0],
            7 => [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0],
            8 => [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0],
        )

        basis_expected = Dict(
            1 => [3, 1],
            2 => [1, 2],
            3 => [5, 7, 2],
            4 => [7, 4, 6],
            5 => [15, 17, 8, 6],
            6 => [8, 11, 19, 13],
            7 => [13, 9, 18, 14],
            8 => [14, 12, 10, 16],
        )

        support_expected = Dict(
            1 => [1, 2],
            2 => [2, 3],
            3 => [1],
            4 => [4],
            5 => [3],
            6 => [4, 5],
            7 => [3, 4],
            8 => [5, 6],
            9 => [7],
            10 => [8],
            11 => [6],
            12 => [8],
            13 => [6, 7],
            14 => [7, 8],
            15 => [5],
            16 => [8],
            17 => [5],
            18 => [7],
            19 => [6],
        )

        @test extraction_coefficients(space, coeffs_expected)
        @test basis_indices(space, basis_expected)
        @test support(space, support_expected)
        @test partition_of_unity(space)

        if PLOT
            using GLMakie, Makie
            display(Mantis.Plot.plot_basis(space))
            readline()
        end
    end
end

@testset "Composition 1D" verbose = true begin
    # Level Geometries
    geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 6),))
    geo_l2 = Refinement(geo_l1, ref_geo_h2)()
    geo_l3 = Refinement(geo_l2, ref_geo_h2)()
    relations = Relations(
        Geometry.parent_to_children_uniform(2), Geometry.child_to_parents_uniform(2)
    )
    geo_scal_1 = Scaling(geo_l1, geo_l2, relations)
    geo_scal_2 = Scaling(geo_l2, geo_l3, relations)

    #=
    GEOMETRY 1

    x inactive
    - active

    Level 1: |-------|-------|xxxxxxx|xxxxxxx|xxxxxxx|
    Level 2: |xxx|xxx|xxx|xxx|---|---|---|---|xxx|xxx|
    Level 3: |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|-|-|-|-|
    
    Final G: |-------|-------|---|---|---|---|-|-|-|-|
    =#
    active = ActiveInfo([[1, 2], [5, 6, 7, 8], [17, 18, 19, 20]])
    hgeo1 = Geometry.HierarchicalGeometry(NestedHierarchy(active, geo_scal_1, geo_scal_2))

    #=
    GEOMETRY 2

    x inactive
    - active

    Level 1: |-------|-------|xxxxxxx|xxxxxxx|xxxxxxx|
    Level 2: |xxx|xxx|xxx|xxx|xxx|xxx|---|---|---|---|
    Level 3: |x|x|x|x|x|x|x|x|-|-|-|-|x|x|x|x|x|x|x|x|
    
    Final G: |-------|-------|-|-|-|-|---|---|---|---|
    =#
    active = ActiveInfo([[1, 2], [7, 8, 9, 10], [9, 10, 11, 12]])
    hgeo2 = Geometry.HierarchicalGeometry(NestedHierarchy(active, geo_scal_1, geo_scal_2))

    # FESpaces
    bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (5,), (1,), (0,))
    bsp_l12 = Refinement(bsp_l1, ref_space_h2)()
    bsp_l2 = Refinement(bsp_l12, ref_space_p1)()
    bsp_l22 = Refinement(bsp_l2, ref_space_h2)()
    bsp_l3 = Refinement(bsp_l22, ref_space_p1)()
    bsp_scal_1 = MatrixScaling((bsp_l1, bsp_l12, bsp_l2), scal_space_h2, scal_space_p1)
    bsp_scal_2 = MatrixScaling((bsp_l2, bsp_l22, bsp_l3), scal_space_h2, scal_space_p1)
    scalings = (bsp_scal_1, bsp_scal_2)

    @testset "HB" verbose = true begin
        hgeo = hgeo1
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, HB
        )

        coeffs_expected = Dict(
            1 => [1.0 0.0; 0.0 1.0],
            2 => [1.0 0.0; 0.0 1.0],
            3 => [0.0 0.0 1.0; 1.0 0.0 0.75; 0.0 1.0 0.5],
            4 => [1.0 0.0 0.0 0.5; 0.0 1.0 0.0 0.25; 0.0 0.0 1.0 0.0],
            5 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            6 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            7 => [
                0.0 0.0 0.0 1.0;
                1.0 0.0 0.0 0.666666666666667;
                0.0 1.0 0.0 0.41666666666666713;
                0.0 0.5 0.5 0.25
            ],
            8 => [
                0.5 0.5 0.0 0.0 0.25;
                0.0 1.0 0.0 0.0 0.08333333333333354;
                0.0 0.0 1.0 0.0 0.0;
                0.0 0.0 0.0 1.0 0.0
            ],
            9 => [
                1.0 0.0 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.5 0.5
            ],
            10 => [
                0.5 0.5 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 1.0
            ],
        )

        basis_expected = Dict(
            1 => [3, 1],
            2 => [1, 2],
            3 => [11, 7, 2],
            4 => [7, 5, 9, 2],
            5 => [9, 8, 10],
            6 => [10, 4, 6],
            7 => [20, 15, 17, 6],
            8 => [15, 17, 16, 19, 6],
            9 => [19, 13, 21, 14],
            10 => [21, 14, 18, 12],
        )

        support_expected = Dict(
            1 => [1, 2],
            2 => [2, 3, 4],
            3 => [1],
            4 => [6],
            5 => [4],
            6 => [6, 7, 8],
            7 => [3, 4],
            8 => [5],
            9 => [4, 5],
            10 => [5, 6],
            11 => [3],
            12 => [10],
            13 => [9],
            14 => [9, 10],
            15 => [7, 8],
            16 => [8],
            17 => [7, 8],
            18 => [10],
            19 => [8, 9],
            20 => [7],
            21 => [9, 10],
        )

        @test extraction_coefficients(space, coeffs_expected)
        @test basis_indices(space, basis_expected)
        @test support(space, support_expected)

        if PLOT
            using GLMakie, Makie
            display(Mantis.Plot.plot_basis(space))
            readline()
        end

        hgeo = hgeo2
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, HB
        )

        coeffs_expected = Dict(
            1 => [1.0 0.0; 0.0 1.0],
            2 => [1.0 0.0; 0.0 1.0],
            3 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            4 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            5 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            6 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            7 => [
                0.0 0.0 0.0 1.0;
                1.0 0.0 0.0 0.9166666666666666;
                0.0 1.0 0.0 0.8333333333333331;
                0.0 0.5 0.5 0.75
            ],
            8 => [
                0.5 0.5 0.0 0.0 0.75;
                0.0 1.0 0.0 0.0 0.6666666666666669;
                0.0 0.0 1.0 0.0 0.5833333333333333;
                0.0 0.0 0.0 1.0 0.5
            ],
            9 => [
                1.0 0.0 0.0 0.0 0.0 0.5;
                0.0 1.0 0.0 0.0 0.0 0.4166666666666666;
                0.0 0.0 1.0 0.0 0.08333333333333352 0.33333333333333315;
                0.0 0.0 0.5 0.5 0.25 0.25
            ],
            10 => [
                0.5 0.5 0.0 0.25 0.25;
                0.0 1.0 0.0 0.41666666666666724 0.16666666666666652;
                0.0 0.0 1.0 0.666666666666667 0.08333333333333323;
                0.0 0.0 0.0 1.0 0.0
            ],
        )

        basis_expected = Dict(
            1 => [3, 1],
            2 => [1, 2],
            3 => [9, 8, 10],
            4 => [10, 4, 6],
            5 => [6, 12, 7],
            6 => [7, 5, 11],
            7 => [13, 21, 18, 2],
            8 => [21, 18, 15, 17, 2],
            9 => [17, 16, 19, 14, 9, 2],
            10 => [19, 14, 20, 9, 2],
        )

        support_expected = Dict(
            1 => [1, 2],
            2 => [2, 7, 8, 9, 10],
            3 => [1],
            4 => [4],
            5 => [6],
            6 => [4, 5],
            7 => [5, 6],
            8 => [3],
            9 => [9, 10, 3],
            10 => [3, 4],
            11 => [6],
            12 => [5],
            13 => [7],
            14 => [9, 10],
            15 => [8],
            16 => [9],
            17 => [8, 9],
            18 => [7, 8],
            19 => [9, 10],
            20 => [10],
            21 => [7, 8],
        )

        @test extraction_coefficients(space, coeffs_expected)
        @test basis_indices(space, basis_expected)
        @test support(space, support_expected)

        if PLOT
            using GLMakie, Makie
            display(Mantis.Plot.plot_basis(space))
            readline()
        end
    end

    @testset "THB" verbose = true begin
        hgeo = hgeo1
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, THB
        )

        coeffs_expected = Dict(
            1 => [1.0 0.0; 0.0 1.0],
            2 => [1.0 0.0; 0.0 1.0],
            3 => [0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 0.0],
            4 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            5 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            6 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            7 => [
                0.0 0.0 0.0 1.0;
                1.0 0.0 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.5 0.5 0.0
            ],
            8 => [
                0.5 0.5 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 1.0
            ],
            9 => [
                1.0 0.0 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.5 0.5
            ],
            10 => [
                0.5 0.5 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 1.0
            ],
        )

        basis_expected = Dict(
            1 => [3, 1],
            2 => [1, 2],
            3 => [11, 7, 2],
            4 => [7, 5, 9],
            5 => [9, 8, 10],
            6 => [10, 4, 6],
            7 => [20, 15, 17, 6],
            8 => [15, 17, 16, 19],
            9 => [19, 13, 21, 14],
            10 => [21, 14, 18, 12],
        )

        support_expected = Dict(
            1 => [1, 2],
            2 => [2, 3],
            3 => [1],
            4 => [6],
            5 => [4],
            6 => [6, 7],
            7 => [3, 4],
            8 => [5],
            9 => [4, 5],
            10 => [5, 6],
            11 => [3],
            12 => [10],
            13 => [9],
            14 => [9, 10],
            15 => [7, 8],
            16 => [8],
            17 => [7, 8],
            18 => [10],
            19 => [8, 9],
            20 => [7],
            21 => [9, 10],
        )

        @test extraction_coefficients(space, coeffs_expected)
        @test basis_indices(space, basis_expected)
        @test support(space, support_expected)
        @test partition_of_unity(space)

        if PLOT
            using GLMakie, Makie
            display(Mantis.Plot.plot_basis(space))
            readline()
        end

        hgeo = hgeo2
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, THB
        )

        coeffs_expected = Dict(
            1 => [1.0 0.0; 0.0 1.0],
            2 => [1.0 0.0; 0.0 1.0],
            3 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            4 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            5 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            6 => [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            7 => [0.0 0.0 0.0 1.0; 1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.5 0.5 0.0],
            8 => [0.5 0.5 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0],
            9 => [
                1.0 0.0 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.5 0.5
            ],
            10 => [
                0.5 0.5 0.0 0.0;
                0.0 1.0 0.0 0.0;
                0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 1.0
            ],
        )

        basis_expected = Dict(
            1 => [3, 1],
            2 => [1, 2],
            3 => [9, 8, 10],
            4 => [10, 4, 6],
            5 => [6, 12, 7],
            6 => [7, 5, 11],
            7 => [13, 21, 18, 2],
            8 => [21, 18, 15, 17],
            9 => [17, 16, 19, 14],
            10 => [19, 14, 20, 9],
        )

        support_expected = Dict(
            1 => [1, 2],
            2 => [2, 7],
            3 => [1],
            4 => [4],
            5 => [6],
            6 => [4, 5],
            7 => [5, 6],
            8 => [3],
            9 => [10, 3],
            10 => [3, 4],
            11 => [6],
            12 => [5],
            13 => [7],
            14 => [9, 10],
            15 => [8],
            16 => [9],
            17 => [8, 9],
            18 => [7, 8],
            19 => [9, 10],
            20 => [10],
            21 => [7, 8],
        )

        @test extraction_coefficients(space, coeffs_expected)
        @test basis_indices(space, basis_expected)
        @test support(space, support_expected)
        @test partition_of_unity(space)

        if PLOT
            using GLMakie, Makie
            display(Mantis.Plot.plot_basis(space))
            readline()
        end
    end
end

# @testset "Uniform 2D" verbose = true begin
#     # Level Geometries
#     rel_1 = Relations(i -> ((i - 1) * 2 + 1, i * 2), i -> (div(i + 1, 2),))
#     geo_l11 = Geometry.create_cartesian_box((0.0,), (1.0,), (5,))
#     geo_l1 = Geometry.TensorProductGeometry((geo_l11, geo_l11))
#     geo_l21 = Refinement(geo_l11, ref_geo_h2)()
#     geo_l2 = Geometry.TensorProductGeometry((geo_l21, geo_l21))
#     scal_11 = Scaling(geo_l11, geo_l21, rel_1)
#     geo_scal_1 = Scaling(geo_l1, geo_l2, (scal_11, scal_11))
#     lin_1 = LinearIndices((5, 5))
#     marked_elements_per_level = [collect(11:25), collect(1:10)]
#     active_info = ActiveInfo(marked_elements_per_level)
#     hgeo = Geometry.HierarchicalGeometry(
#         NestedHierarchy(active_info, geo_scal_1)
#     )
#     @testset "Polar Splines Degree-elevation" verbose = true begin
#         num_els = (5, 5)
#         section_spaces = (FunctionSpaces.Bernstein(2), FunctionSpaces.Bernstein(2))
#         pgeo_1, pgeo_coeffs_1 = FunctionSpaces.create_polar_geometry_data(
#             num_els, (FunctionSpaces.Bernstein(2), FunctionSpaces.Bernstein(2)), (1, 1)
#         )
#         P1 = FunctionSpaces.create_vector_polar_spline_space(
#             num_els, (2, 2), (1, 1), pgeo_1; geom_coeffs_tp=pgeo_coeffs_1
#         )
#         pgeo_2, pgeo_coeffs_2 = FunctionSpaces.create_polar_geometry_data(
#             num_els, (FunctionSpaces.Bernstein(3), FunctionSpaces.Bernstein(3)), (2, 2)
#         )
#         P2 = FunctionSpaces.create_vector_polar_spline_space(
#             num_els, (3, 3), (2, 2), pgeo_2; geom_coeffs_tp=pgeo_coeffs_2
#         )
#         Q = Quadrature.get_global_quadrature_rules(
#             Quadrature.gauss_legendre, prod(num_els), (4, 4)
#         )[1]
#         pgeo_scal_1 = Scaling(
#             pgeo_1, pgeo_2, get_relations(geo_scal_1)
#         )
#         hpgeo = Geometry.HierarchicalGeometry(
#             NestedHierarchy(active_info, pgeo_scal_1)
#         )
#         #Plot.export_geometry_to_vtk(hpgeo, "test")
#         space_scal_1 = MatrixScaling(
#             P1, P2, (p, c) -> Assemblers.scaling_matrix_general(p, c, Q)
#         )
#         # TODO: This requires `get_support` on PolarSpline and GTBSpace, which are not
#         # implemented
#         space = FunctionSpaces.HierarchicalSpace(hpgeo, hgeo, (space_scal_1,),
#         SelectionStandard, THB)
#     end
# end

end

module ExtractionTests

using Mantis
using Mantis.Hierarchical
using Test
include(joinpath(pwd(), "test/TestHelpers.jl"))
include("data/ExtractionTests/data.jl")

const PLOT = false
if PLOT
    using GLMakie, Makie
end

ref_space_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
ref_geo_h2(s) = Geometry.refinement_uniform(s, 2)
ref_space_p1(s) = FunctionSpaces.refinement_degree(s, 1)
scal_space_p1(p, c) = FunctionSpaces.scaling_matrix_degree(p, c, 1)
scal_space_h2 = FunctionSpaces.scaling_matrix_uniform
const SelectionStandard = FunctionSpaces.SelectionStandard
const HB = FunctionSpaces.HB
const THB = FunctionSpaces.THB

############################################################################################
#                                       Test Helpers                                       #
############################################################################################

const xi_1d = Points.PointSet((LinRange(0, 1, 6),))
const xi_2d = Points.TensorProductPoints((LinRange(0, 1, 6), LinRange(0, 1, 6)))

const get_support = FunctionSpaces.get_support
const get_extraction_coefficients = FunctionSpaces.get_extraction_coefficients
const get_basis_indices = FunctionSpaces.get_basis_indices

function partition_of_unity(space::FunctionSpaces.AbstractFESpace{1}, e)
    eval = FunctionSpaces.evaluate(space, e, xi_1d)[1][1][1][1]

    return all(p -> isapprox(p, 1.0), sum(eval; dims=2))
end

function partition_of_unity(space::FunctionSpaces.AbstractFESpace{2}, e)
    eval = FunctionSpaces.evaluate(space, e, xi_2d)[1][1][1][1]

    return all(p -> isapprox(p, 1.0), sum(eval; dims=2))
end

function partition_of_unity_dict(space)
    return Dict(e => true for e in 1:FunctionSpaces.get_num_elements(space))
end

function plot(space::FunctionSpaces.AbstractFESpace{2}, file_name)
    coeffs = ones(FunctionSpaces.get_num_basis(space))
    space_form = Forms.FormField(Forms.FormSpace(0, space, ""), coeffs)
    Plot.export_form_fields_to_vtk((space_form,), file_name)

    return nothing
end

function plot(space::FunctionSpaces.AbstractFESpace{1})
    display(Mantis.Plot.plot_basis(space; plot_points_per_element=2))
    println("Press `Enter` to show the next plot.")
    readline()

    return nothing
end

function test_hb(space, expected)
    @test @expected expected.coeffs e -> get_extraction_coefficients(space, e) isapprox
    @test @expected expected.basis e -> get_basis_indices(space, e)
    @test @expected expected.support b -> get_support(space, b)

    return nothing
end

function test_thb(space, expected)
    @test @expected expected.coeffs e -> get_extraction_coefficients(space, e) isapprox
    @test @expected expected.basis e -> get_basis_indices(space, e)
    @test @expected expected.support b -> get_support(space, b)
    @test @expected partition_of_unity_dict(space) e -> partition_of_unity(space, e)

    return nothing
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
            @testset "Geometry 1" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo1, hgeo1, scalings, SelectionStandard, HB
                )
                test_hb(space, DATA[:uni_1d_p1_hb_g1])
                if PLOT; plot(space); end
            end

            @testset "Geometry 2" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo2, hgeo2, scalings, SelectionStandard, HB
                )
                test_hb(space, DATA[:uni_1d_p1_hb_g2])
                if PLOT; plot(space); end
            end
        end

        @testset "THB" verbose = true begin
            @testset "Geometry 1" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo1, hgeo1, scalings, SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_1d_p1_thb_g1])
                if PLOT; plot(space); end
            end

            @testset "Geometry 2" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo2, hgeo2, scalings, SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_1d_p1_thb_g2])
                if PLOT; plot(space); end
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
            @testset "Geometry 1" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo1, hgeo1, scalings, SelectionStandard, HB
                )
                test_hb(space, DATA[:uni_1d_p2_hb_g1])
                if PLOT; plot(space); end
            end

            @testset "Geometry 2" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo2, hgeo2, scalings, SelectionStandard, HB
                )
                test_hb(space, DATA[:uni_1d_p2_hb_g2])
                if PLOT; plot(space); end
            end
        end

        @testset "THB" verbose = true begin
            @testset "Geometry 1" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo1, hgeo1, scalings, SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_1d_p2_thb_g1])
                if PLOT; plot(space); end
            end

            @testset "Geometry 2" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo2, hgeo2, scalings, SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_1d_p2_thb_g2])
                if PLOT; plot(space); end
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
    hgeo = Geometry.HierarchicalGeometry(NestedHierarchy(active, geo_scal_1, geo_scal_1))

    # FESpaces
    bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (8,), (1,), (0,))
    bsp_l2 = Refinement(bsp_l1, ref_space_p1)()
    bsp_l3 = Refinement(bsp_l2, ref_space_p1)()
    bsp_scal_1 = MatrixScaling(bsp_l1, bsp_l2, scal_space_p1)
    bsp_scal_2 = MatrixScaling(bsp_l2, bsp_l3, scal_space_p1)
    scalings = (bsp_scal_1, bsp_scal_2)

    @testset "HB" verbose = true begin
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, HB
        )
        test_hb(space, DATA[:deg_1d_hb])
        if PLOT; plot(space); end
    end

    @testset "THB" verbose = true begin
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, THB
        )
        test_thb(space, DATA[:deg_1d_thb])
        if PLOT; plot(space); end
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
        @testset "Geometry 1" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                hgeo1, hgeo1, scalings, SelectionStandard, HB
            )
            test_hb(space, DATA[:comp_1d_hb_g1])
            if PLOT; plot(space); end
        end

        @testset "Geometry 2" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                hgeo2, hgeo2, scalings, SelectionStandard, HB
            )
            test_hb(space, DATA[:comp_1d_hb_g2])
            if PLOT; plot(space); end
        end
    end

    @testset "THB" verbose = true begin
        @testset "Geometry 1" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                hgeo1, hgeo1, scalings, SelectionStandard, THB
            )
            test_thb(space, DATA[:comp_1d_thb_g1])

            if PLOT; plot(space); end
        end

        @testset "Geometry 2" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                hgeo2, hgeo2, scalings, SelectionStandard, THB
            )
            test_thb(space, DATA[:comp_1d_thb_g2])
            if PLOT; plot(space); end
        end
    end
end

@testset "Uniform 2D" verbose = true begin
    # Level Geometries
    num_els = (5, 5)
    geo_l1 = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), num_els)
    geo_scal_1 = Geometry.scaling_uniform(geo_l1, (2, 2))
    l1_active = collect((num_els[1] + 1):(num_els[1] * 4))
    l2_active = mapreduce(
        e -> collect(get_children(geo_scal_1, e)),
        vcat,
        setdiff(1:Geometry.get_num_elements(geo_l1), l1_active),
    )
    marked_elements_per_level = [l1_active, l2_active]
    active_info = ActiveInfo(marked_elements_per_level)
    hgeo = Geometry.HierarchicalGeometry(NestedHierarchy(active_info, geo_scal_1))
    Plot.export_geometry_to_vtk(hgeo, "hgeo")
    @testset "Polar Splines (Scalar)" verbose = true begin
        p = 2
        pgeo_1, pgeo_coeffs_1 = FunctionSpaces.create_polar_geometry_data(
            num_els, (p, p), (p-1, p-1)
        )
        P1 = FunctionSpaces.create_scalar_polar_spline_space(
            num_els, (p, p), (p-1, p-1), pgeo_1; geom_coeffs_tp=pgeo_coeffs_1
        )
        scaling = FunctionSpaces.scaling_uniform(P1, (2, 2))
        pscal = Scaling(
            FunctionSpaces.get_geometry(P1),
            FunctionSpaces.get_geometry(get_child(scaling)),
            Hierarchical.get_relations(geo_scal_1),
        )
        pgeo = Geometry.HierarchicalGeometry(
            NestedHierarchy(active_info, pscal; check_tree=false)
        )
        @testset "HB" verbose = true begin
            @testset "Geometry 1" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    pgeo, hgeo, (scaling,), SelectionStandard, HB
                )
                test_hb(space, DATA[:uni_2d_p2_hb_polar_scalar])
                if PLOT; plot(space, "polar-scalar-hb"); end
            end
        end

        @testset "THB" verbose = true begin
            @testset "Geometry 1" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    pgeo, hgeo, (scaling,), SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_2d_p2_thb_polar_scalar])
                if PLOT; plot(space, "polar-scalar-thb"); end
            end
        end
    end
end

end

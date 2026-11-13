module ExtractionTests

using Mantis
using Mantis.Hierarchical
using Test
include("../../TestHelpers.jl")
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
    eval = FunctionSpaces.evaluate(space, e, xi_1d)[1][1][1]

    return all(p -> isapprox(p, 1.0), sum(eval; dims=2))
end

function partition_of_unity(
    space::FunctionSpaces.AbstractFESpace{2, num_components}, e
) where {num_components}
    eval = FunctionSpaces.evaluate(space, e, xi_2d)[1][1]
    for c in 1:num_components
        if !all(p -> isapprox(p, 1.0), sum(eval[c]; dims=2))
            return false
        end
    end

    return true
end

function partition_of_unity_dict(space)
    return Dict(e => true for e in 1:FunctionSpaces.get_num_elements(space))
end

function plot(space::FunctionSpaces.AbstractFESpace{2, 1}, file_name)
    coeffs = ones(FunctionSpaces.get_num_basis(space))
    space_form = Forms.FormField(Forms.FormSpace(0, space, "space"), coeffs)
    Plot.export_form_fields_to_vtk((space_form,), file_name)

    return nothing
end

function plot(space::FunctionSpaces.AbstractFESpace{2, 2}, file_name)
    coeffs = ones(FunctionSpaces.get_num_basis(space))
    space_form = Forms.FormField(Forms.FormSpace(1, space, "space"), coeffs)
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

function uniform_scalings(space, num_subdivisions, num_scalings)
    scaling = FunctionSpaces.scaling_uniform(space, num_subdivisions)
    num_scalings == 1 && return (scaling,)
    child_scalings = uniform_scalings(get_child(scaling), num_subdivisions, num_scalings - 1)

    return (scaling, child_scalings...)
end

# Maps the basis function `basis_id` of the hierarchy of direct sums to the corresponding
# basis function of the direct sum of hierarchies.
function direct_sum_basis_id(hier_of_ds, ds_of_hier, basis_id)
    basis = FunctionSpaces.get_basis(hier_of_ds)
    level, level_id = Hierarchical.convert_to_level_and_level_id(basis, basis_id)
    level_offsets = FunctionSpaces.get_dof_offsets(Hierarchical.get_set(basis, level))
    component_id = findlast(<(level_id), level_offsets)
    component_space = FunctionSpaces.get_component_spaces(ds_of_hier)[component_id]
    component_basis_id = Hierarchical.convert_to_hier_id(
        FunctionSpaces.get_basis(component_space),
        level,
        level_id - level_offsets[component_id],
    )

    return component_basis_id + FunctionSpaces.get_dof_offsets(ds_of_hier, component_id)
end

# Compares the hierarchy of direct sums of `ds` with the direct sum of the hierarchies of the
# components of `ds`. Both should match up to a renumbering of the basis functions.
function test_direct_sum(geometry, ds, num_subdivisions, ::Type{B}) where {B}
    num_scalings = Geometry.get_num_levels(geometry) - 1
    hier_of_ds = FunctionSpaces.HierarchicalSpace(
        geometry,
        geometry,
        uniform_scalings(ds, num_subdivisions, num_scalings),
        SelectionStandard,
        B,
    )
    component_spaces = map(FunctionSpaces.get_component_spaces(ds)) do space
        scalings = uniform_scalings(space, num_subdivisions, num_scalings)

        return FunctionSpaces.HierarchicalSpace(
            geometry, geometry, scalings, SelectionStandard, B
        )
    end
    ds_of_hier = FunctionSpaces.DirectSumSpace(component_spaces)

    num_basis = FunctionSpaces.get_num_basis(hier_of_ds)
    @test num_basis == FunctionSpaces.get_num_basis(ds_of_hier)
    basis_map = map(b -> direct_sum_basis_id(hier_of_ds, ds_of_hier, b), 1:num_basis)
    @test sort(basis_map) == 1:num_basis
    for e in 1:FunctionSpaces.get_num_elements(hier_of_ds)
        @test sort(basis_map[get_basis_indices(hier_of_ds, e)]) ==
            sort(get_basis_indices(ds_of_hier, e))
        level, level_id = Geometry.convert_to_level_and_level_id(geometry, e)
        level_space = Hierarchical.get_set(FunctionSpaces.get_basis(hier_of_ds), level)
        for (c, component_space) in enumerate(component_spaces)
            basis_ids = basis_map[get_basis_indices(hier_of_ds, e, c)]
            expected_basis_ids = get_basis_indices(ds_of_hier, e, c)
            @test sort(basis_ids) == sort(expected_basis_ids)
            perm = indexin(expected_basis_ids, basis_ids)
            # The level-wise direct sums evaluate their component spaces, so they have
            # identity extraction blocks, and the extraction of the level-wise component
            # space is missing.
            level_coeffs = get_extraction_coefficients(
                FunctionSpaces.get_component_spaces(level_space)[c], level_id
            )
            @test level_coeffs * get_extraction_coefficients(hier_of_ds, e, c, c)[:, perm] ≈
                get_extraction_coefficients(component_space, e)
        end
    end

    for b in 1:num_basis
        @test sort(get_support(hier_of_ds, b)) == sort(get_support(ds_of_hier, basis_map[b]))
    end

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
    hgeo1 = Geometry.HierarchicalGeometry(TreeHierarchy(active, geo_scal_1, geo_scal_2))

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
    hgeo2 = Geometry.HierarchicalGeometry(TreeHierarchy(active, geo_scal_1, geo_scal_2))
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
                if PLOT
                    plot(space)
                end
            end

            @testset "Geometry 2" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo2, hgeo2, scalings, SelectionStandard, HB
                )
                test_hb(space, DATA[:uni_1d_p1_hb_g2])
                if PLOT
                    plot(space)
                end
            end
        end

        @testset "THB" verbose = true begin
            @testset "Geometry 1" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo1, hgeo1, scalings, SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_1d_p1_thb_g1])
                if PLOT
                    plot(space)
                end
            end

            @testset "Geometry 2" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo2, hgeo2, scalings, SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_1d_p1_thb_g2])
                if PLOT
                    plot(space)
                end
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
                if PLOT
                    plot(space)
                end
            end

            @testset "Geometry 2" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo2, hgeo2, scalings, SelectionStandard, HB
                )
                test_hb(space, DATA[:uni_1d_p2_hb_g2])
                if PLOT
                    plot(space)
                end
            end
        end

        @testset "THB" verbose = true begin
            @testset "Geometry 1" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo1, hgeo1, scalings, SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_1d_p2_thb_g1])
                if PLOT
                    plot(space)
                end
            end

            @testset "Geometry 2" verbose = true begin
                space = FunctionSpaces.HierarchicalSpace(
                    hgeo2, hgeo2, scalings, SelectionStandard, THB
                )
                test_thb(space, DATA[:uni_1d_p2_thb_g2])
                if PLOT
                    plot(space)
                end
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
    
    Final G: |-------|-------|-------|-------|-------|-------|-------|-------|
    =#
    active = ActiveInfo([[1, 2], [3, 4], [5, 6, 7, 8]])
    hgeo = Geometry.HierarchicalGeometry(TreeHierarchy(active, geo_scal_1, geo_scal_1))

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
        if PLOT
            plot(space)
        end
    end

    @testset "THB" verbose = true begin
        space = FunctionSpaces.HierarchicalSpace(
            hgeo, hgeo, scalings, SelectionStandard, THB
        )
        test_thb(space, DATA[:deg_1d_thb])
        if PLOT
            plot(space)
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
    hgeo1 = Geometry.HierarchicalGeometry(TreeHierarchy(active, geo_scal_1, geo_scal_2))

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
    hgeo2 = Geometry.HierarchicalGeometry(TreeHierarchy(active, geo_scal_1, geo_scal_2))

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
            if PLOT
                plot(space)
            end
        end

        @testset "Geometry 2" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                hgeo2, hgeo2, scalings, SelectionStandard, HB
            )
            test_hb(space, DATA[:comp_1d_hb_g2])
            if PLOT
                plot(space)
            end
        end
    end

    @testset "THB" verbose = true begin
        @testset "Geometry 1" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                hgeo1, hgeo1, scalings, SelectionStandard, THB
            )
            test_thb(space, DATA[:comp_1d_thb_g1])

            if PLOT
                plot(space)
            end
        end

        @testset "Geometry 2" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                hgeo2, hgeo2, scalings, SelectionStandard, THB
            )
            test_thb(space, DATA[:comp_1d_thb_g2])
            if PLOT
                plot(space)
            end
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
    hgeo = Geometry.HierarchicalGeometry(TreeHierarchy(active_info, geo_scal_1))
    @testset "Polar Splines (Scalar)" verbose = true begin
        p = 2
        polar_geo, geo_coeffs = FunctionSpaces.create_polar_geometry_data(
            num_els, (p, p), (p - 1, p - 1)
        )
        P1 = FunctionSpaces.create_scalar_polar_spline_space(
            num_els, (p, p), (p - 1, p - 1), polar_geo; geom_coeffs_tp=geo_coeffs
        )
        scaling = FunctionSpaces.scaling_uniform(P1, (2, 2))
        pscal = Scaling(
            FunctionSpaces.get_geometry(P1),
            FunctionSpaces.get_geometry(get_child(scaling)),
            Hierarchical.get_relations(geo_scal_1),
        )
        pgeo = Geometry.HierarchicalGeometry(
            TreeHierarchy(active_info, pscal; check_tree=false)
        )
        # The level-wise parametric geometries have to be those of the level spaces.
        param_scal = Scaling(
            FunctionSpaces.get_parametric_geometry(P1),
            FunctionSpaces.get_parametric_geometry(get_child(scaling)),
            Hierarchical.get_relations(geo_scal_1),
        )
        param_geo = Geometry.HierarchicalGeometry(
            TreeHierarchy(active_info, param_scal; check_tree=false)
        )

        @testset "HB" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                pgeo, param_geo, (scaling,), SelectionStandard, HB
            )
            test_hb(space, DATA[:uni_2d_p2_hb_polar_scalar])
            if PLOT
                plot(space, "polar-scalar-hb")
            end
        end

        @testset "THB" verbose = true begin
            space = FunctionSpaces.HierarchicalSpace(
                pgeo, param_geo, (scaling,), SelectionStandard, THB
            )
            test_thb(space, DATA[:uni_2d_p2_thb_polar_scalar])
            if PLOT
                plot(space, "polar-scalar-thb")
            end
        end
    end

    @testset "Polar Splines (Vector)" verbose = true begin
        # TODO: Waiting for multi-component extraction overhaul. 
        # https://github.com/MantisFEM/Mantis.jl/discussions/421
        p = 2
        polar_geo, geo_coeffs = FunctionSpaces.create_polar_geometry_data(
            num_els, (p, p), (p - 1, p - 1)
        )
        section_spaces = (FunctionSpaces.Bernstein(p), FunctionSpaces.Bernstein(p))
        P1 = FunctionSpaces.create_vector_polar_spline_space(
            num_els, (p, p), (p - 1, p - 1), polar_geo; geom_coeffs_tp=geo_coeffs
        )
        scaling = FunctionSpaces.scaling_uniform(P1, (2, 2))
        pscal = Scaling(
            FunctionSpaces.get_geometry(P1),
            FunctionSpaces.get_geometry(get_child(scaling)),
            Hierarchical.get_relations(geo_scal_1),
        )
        pgeo = Geometry.HierarchicalGeometry(
            TreeHierarchy(active_info, pscal; check_tree=false)
        )
        @testset "HB" verbose = true begin
            # space = FunctionSpaces.HierarchicalSpace(
            #     pgeo, hgeo, (scaling,), SelectionStandard, HB
            # )
            @test_skip test_hb(space, DATA[:uni_2d_p2_hb_polar_vector])
            # if PLOT; plot(space, "polar-vector-hb"); end
        end

        @testset "THB" verbose = true begin
            # space = FunctionSpaces.HierarchicalSpace(
            #     pgeo, hgeo, (scaling,), SelectionStandard, THB
            # )
            @test_skip test_thb(space, DATA[:uni_2d_p2_thb_polar_vector])
            # if PLOT; plot(space, "polar-vector-thb"); end
        end
    end
end

@testset "Direct Sum <-> Hierarchical Correspondence" verbose = true begin
    @testset "Uniform 1D" verbose = true begin
        # Level Geometries
        geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))
        geo_scal_1 = Geometry.scaling_uniform(geo_l1, 2)
        geo_scal_2 = Geometry.scaling_uniform(get_child(geo_scal_1), 2)

        # Same geometries as in the "Uniform 1D" tests.
        active_1 = ActiveInfo([[1, 2], [5, 6], [13, 14, 15, 16]])
        active_2 = ActiveInfo([[1, 2], [6], [9, 10, 13, 14, 15, 16]])

        # FESpaces
        bsp_p1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (1,), (0,))
        bsp_p2 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (2,), (1,))
        ds = FunctionSpaces.DirectSumSpace((bsp_p1, bsp_p2))

        for (g, active) in enumerate((active_1, active_2)), B in (HB, THB)
            @testset "$(nameof(B)) Geometry $g" verbose = true begin
                hgeo = Geometry.HierarchicalGeometry(
                    TreeHierarchy(active, geo_scal_1, geo_scal_2)
                )
                test_direct_sum(hgeo, ds, (2,), B)
            end
        end
    end

    @testset "Uniform 2D" verbose = true begin
        # FESpaces
        num_els = (5, 5)
        function tp(p)
            return FunctionSpaces.TensorProductSpace(
                ntuple(
                    d -> FunctionSpaces.create_bspline_space(
                        (0.0,), (1.0,), (num_els[d],), (p,), (p - 1,)
                    ),
                    2,
                ),
            )
        end
        ds = FunctionSpaces.DirectSumSpace((tp(2), tp(3)))

        # Level Geometries
        geo_l1 = FunctionSpaces.get_parametric_geometry(ds)
        geo_l2 = FunctionSpaces.get_parametric_geometry(
            FunctionSpaces.refinement_uniform(ds, (2, 2))
        )
        geo_scal_1 = Geometry.scaling_uniform(geo_l1, geo_l2, (2, 2))

        # Same geometry as in the "Uniform 2D" tests.
        l1_active = collect((num_els[1] + 1):(num_els[1] * 4))
        l2_active = mapreduce(
            e -> collect(get_children(geo_scal_1, e)),
            vcat,
            setdiff(1:Geometry.get_num_elements(geo_l1), l1_active),
        )
        active_info = ActiveInfo([l1_active, l2_active])
        hgeo = Geometry.HierarchicalGeometry(TreeHierarchy(active_info, geo_scal_1))

        for B in (HB, THB)
            @testset "$(nameof(B))" verbose = true begin
                test_direct_sum(hgeo, ds, (2, 2), B)
            end
        end
    end
end

end

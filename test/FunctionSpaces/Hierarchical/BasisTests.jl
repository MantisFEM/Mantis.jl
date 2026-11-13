module BasisTests

using Mantis
using Test

ref_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
const SelectionStandard = FunctionSpaces.SelectionStandard
const SelectionSimple = FunctionSpaces.SelectionSimple
const HB = FunctionSpaces.HB
const THB = FunctionSpaces.THB

@testset "Uniform 1D" verbose = true begin
    # Geometries
    geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))
    geo_l2 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 9),))
    geo_l3 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 17),))
    geo_scal_1 = Geometry.scaling_uniform(geo_l1, geo_l2, 2)
    geo_scal_2 = Geometry.scaling_uniform(geo_l2, geo_l3, 2)

    # FESpaces
    bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (1,), (0,))
    bsp_l2 = Hierarchical.Refinement(bsp_l1, ref_h2)()
    bsp_l3 = Hierarchical.Refinement(bsp_l2, ref_h2)()
    bsp_scal_1 = Hierarchical.MatrixScaling(
        bsp_l1, bsp_l2, FunctionSpaces.scaling_matrix_uniform
    )
    bsp_scal_2 = Hierarchical.MatrixScaling(
        bsp_l2, bsp_l3, FunctionSpaces.scaling_matrix_uniform
    )

    #=
    x inactive
    - active

    Level 1: |-------|xxxxxxx|xxxxxxx|-------|
    Level 2: |xxx|xxx|xxx|xxx|---|---|xxx|xxx|
    Level 3: |x|x|x|x|-|-|-|-|x|x|x|x|x|x|x|x|
    
    Final G: |-------|-|-|-|-|---|---|-------|
    =#
    active = Hierarchical.ActiveInfo([[1, 4], [5, 6], [5, 6, 7, 8]])
    hierarchy = Hierarchical.TreeHierarchy(active, geo_scal_1, geo_scal_2)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)

    # Standard selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionStandard
    )
    # The numbering of hierarhical basis functions is an implementationd detail and can
    # _not_ be relied on, so we sort it here and throughout the rest of the code.
    @test sort.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 4, 5], [5, 6], [6, 7, 8]]

    # Simple selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionSimple
    )
    @test sort.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 4, 5], [5, 6], [6, 7, 8]]

    #=
    x inactive
    - active

    Level 1: |-------|-------|xxxxxxx|-------|
    Level 2: |xxx|xxx|xxx|xxx|---|---|xxx|xxx|
    Level 3: |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|
    
    Final G: |-------|-------|---|---|-------|
    =#
    active = Hierarchical.ActiveInfo([[1, 2, 4], [5, 6], Int[]])
    hierarchy = Hierarchical.TreeHierarchy(active, geo_scal_1, geo_scal_2)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)

    # Standard selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionStandard
    )
    @test sort.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 3, 4, 5], [6], Int[]]

    # Simple selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionSimple
    )
    @test sort.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 3, 4, 5], Int[], Int[]] # No parent is deactivated, so no child is active.

    #=
    x inactive
    - active

    Level 1: |-------|-------|xxxxxxx|-------|
    Level 2: |xxx|xxx|xxx|xxx|xxx|xxx|xxx|xxx|
    Level 3: |x|x|x|x|x|x|x|x|-|-|-|-|x|x|x|x|
    
    Final G: |-------|-------|-|-|-|-|-------|
    =#
    active = Hierarchical.ActiveInfo([[1, 2, 4], Int[], [9, 10, 11, 12]])
    hierarchy = Hierarchical.TreeHierarchy(active, geo_scal_1, geo_scal_2)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)

    # Standard selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionStandard
    )
    @test sort.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 3, 4, 5], Int[], [10, 11, 12]]

    # Simple selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionSimple
    )
    @test sort.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 3, 4, 5], Int[], Int[]] # No parent is deactivated, so no child is active.

    # Errors
    struct SelectionUnknown <: FunctionSpaces.SelectionAlgorithm end
    @test_throws MethodError FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionUnknown
    )
    for selection_type in (SelectionStandard, SelectionSimple)
        # The geometry has 3 levels, but the spaces only have 2.
        @test_throws ArgumentError FunctionSpaces.create_basis(
            hgeo, (bsp_scal_1,), selection_type
        )
        # The space on level 1 is built on the geometry of level 2.
        @test_throws ArgumentError FunctionSpaces.create_basis(
            hgeo, (bsp_scal_2, bsp_scal_2), selection_type
        )
        # Same number of elements, but the geometry on level 1 is not that of the space.
        geo_l1_other = Geometry.CartesianGeometry((LinRange(0.0, 2.0, 5),))
        hgeo_other = Geometry.HierarchicalGeometry(
            Hierarchical.ActiveInfo([[1, 2, 3, 4], Int[]]),
            (Geometry.scaling_uniform(geo_l1_other, geo_l2, 2),),
        )
        @test_throws ArgumentError FunctionSpaces.create_basis(
            hgeo_other, (bsp_scal_1,), selection_type
        )
    end
end

@testset "Update" verbose = true begin
    # Geometries
    geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))
    geo_scal_1 = Geometry.scaling_uniform(geo_l1, 2)
    geo_scal_2 = Geometry.scaling_uniform(Hierarchical.get_child(geo_scal_1), 2)

    # FESpaces
    bsp_l1 = FunctionSpaces.create_bspline_space(0.0, 1.0, 4, 1, 0)
    bsp_scal_1 = FunctionSpaces.scaling_uniform(bsp_l1, 2)
    bsp_scal_2 = FunctionSpaces.scaling_uniform(Hierarchical.get_child(bsp_scal_1), 2)
    bsp_scalings = (bsp_scal_1, bsp_scal_2)

    # Initially the geometry will be this:

    #=
    x inactive
    - active

    Level 1: |-------|xxxxxxx|xxxxxxx|-------|
    Level 2: |xxx|xxx|---|---|---|---|xxx|xxx|
    
    Final G: |-------|---|---|---|---|-------|
    =#

    # And we will refine it such that it ends-up as:

    #=
    x inactive
    - active

    Level 1: |-------|xxxxxxx|xxxxxxx|-------|
    Level 2: |xxx|xxx|xxx|xxx|---|---|xxx|xxx|
    Level 3: |x|x|x|x|-|-|-|-|x|x|x|x|x|x|x|x|
    
    Final G: |-------|-|-|-|-|---|---|-------|
    =#
    active = Hierarchical.ActiveInfo([[1, 4], [3, 4, 5, 6]])
    hierarchy = Hierarchical.TreeHierarchy(active, geo_scal_1)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)
    # Sanity check on geometry updates
    @test Geometry.get_num_elements(hgeo) == 6
    @test Hierarchical.get_scalings(Geometry.get_hierarchy(hgeo)) == (geo_scal_1,)
    @test Hierarchical.get_level_ids(Geometry.get_hierarchy(hgeo)) == [[1, 4], [3, 4, 5, 6]]
    hgeo = Geometry.refine(hgeo, 2, [3, 4], geo_scal_2)
    @test Geometry.get_num_elements(hgeo) == 8
    @test Hierarchical.get_scalings(Geometry.get_hierarchy(hgeo)) ==
        (geo_scal_1, geo_scal_2)
    @test Hierarchical.get_level_ids(Geometry.get_hierarchy(hgeo)) ==
        [[1, 4], [5, 6], [5, 6, 7, 8]]
    # Test space updates
    active = Hierarchical.ActiveInfo([[1, 2, 3, 4], Int[]])
    hierarchy = Hierarchical.TreeHierarchy(active, geo_scal_1)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)
    space = FunctionSpaces.HierarchicalSpace(
        hgeo, hgeo, (bsp_scal_1,), SelectionStandard, HB
    )
    basis = FunctionSpaces.get_basis(space)
    @test Hierarchical.get_num_objects(basis) == 5
    @test Hierarchical.get_level_ids(basis) == [[1, 2, 3, 4, 5], Int[]]
    space = FunctionSpaces.refine(space, 1, [2, 3])
    basis = FunctionSpaces.get_basis(space)
    @test Hierarchical.get_num_objects(basis) == 7
    @test sort.(Hierarchical.get_level_ids(basis)) == [[1, 2, 4, 5], [4, 5, 6]]
    space = FunctionSpaces.refine(space, 2, [3, 4])
    basis = FunctionSpaces.get_basis(space)
    @test Hierarchical.get_num_objects(basis) == 9
    @test sort.(Hierarchical.get_level_ids(basis)) == [[1, 2, 4, 5], [5, 6], [6, 7, 8]]
end

@testset "Direct sum" verbose = true begin
    # Geometries
    geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))
    geo_scal_1 = Geometry.scaling_uniform(geo_l1, 2)
    geo_scal_2 = Geometry.scaling_uniform(Hierarchical.get_child(geo_scal_1), 2)
    active = Hierarchical.ActiveInfo([[1, 4], [5, 6], [5, 6, 7, 8]])
    hierarchy = Hierarchical.TreeHierarchy(active, geo_scal_1, geo_scal_2)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)

    # FESpaces
    function uniform_scalings(space, num_subdivisions)
        scal_1 = FunctionSpaces.scaling_uniform(space, num_subdivisions)
        scal_2 = FunctionSpaces.scaling_uniform(Hierarchical.get_child(scal_1), num_subdivisions)

        return (scal_1, scal_2)
    end

    bsp_p1 = FunctionSpaces.create_bspline_space(0.0, 1.0, 4, 1, 0)
    bsp_p2 = FunctionSpaces.create_bspline_space(0.0, 1.0, 4, 2, 1)
    ds = FunctionSpaces.DirectSumSpace((bsp_p1, bsp_p2))
    ds_scalings = uniform_scalings(ds, (2,))
    component_scalings = map(s -> uniform_scalings(s, 2), (bsp_p1, bsp_p2))

    # The active basis of the hierarchy of direct sums is that of each component hierarchy,
    # with the level-wise ids shifted by the offsets of the level-wise direct sum spaces.
    for selection_type in (SelectionStandard, SelectionSimple)
        ds_basis = FunctionSpaces.create_basis(hgeo, ds_scalings, selection_type)
        component_bases = map(
            scalings -> FunctionSpaces.create_basis(hgeo, scalings, selection_type),
            component_scalings,
        )
        @test Hierarchical.get_num_objects(ds_basis) ==
            sum(Hierarchical.get_num_objects, component_bases)
        for l in 1:Hierarchical.get_num_levels(ds_basis)
            ds_space = Hierarchical.get_set(ds_basis, l)
            expected = mapreduce(vcat, 1:2) do c
                level_ids = Hierarchical.get_level_ids(component_bases[c], l)

                return level_ids .+ FunctionSpaces.get_dof_offsets(ds_space, c)
            end
            @test sort(Hierarchical.get_level_ids(ds_basis, l)) == sort(expected)
        end
    end
end

end

module BasisTests

using Mantis
using Test

ref_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
const SelectionStandard = FunctionSpaces.SelectionStandard
const SelectionSimple = FunctionSpaces.SelectionSimple

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
    hierarchy = Hierarchical.NestedHierarchy(active, geo_scal_1, geo_scal_2)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)

    # Standard selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionStandard
    )
    @test sort!.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 4, 5], [5, 6], [6, 7, 8]] # Hierarchical does not guarantee sorted ids

    # Simple selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionSimple
    )
    @test sort!.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
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
    hierarchy = Hierarchical.NestedHierarchy(active, geo_scal_1, geo_scal_2)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)

    # Standard selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionStandard
    )
    @test sort!.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 3, 4, 5], [6], Int[]]

    # Simple selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionSimple
    )
    @test sort!.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
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
    hierarchy = Hierarchical.NestedHierarchy(active, geo_scal_1, geo_scal_2)
    hgeo = Geometry.HierarchicalGeometry(hierarchy)

    # Standard selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionStandard
    )
    @test sort!.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 3, 4, 5], Int[], [10, 11, 12]]

    # Simple selection
    active_basis = FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionSimple
    )
    @test sort!.(Hierarchical.get_level_ids(Hierarchical.get_active_info(active_basis))) ==
        [[1, 2, 3, 4, 5], Int[], Int[]] # No parent is deactivated, so no child is active.

    # Errors
    struct SelectionUnknown <: FunctionSpaces.SelectionAlgorithm end
    @test_throws MethodError FunctionSpaces.create_basis(
        hgeo, (bsp_scal_1, bsp_scal_2), SelectionUnknown
    )
end

end

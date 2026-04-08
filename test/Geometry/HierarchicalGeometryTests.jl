module HierarchicalGeometryTests

using Mantis
using Test

############################################################################################
#                                          Setup                                           #
############################################################################################

# 1D two-level hierarchy
#   Level 1: 2 coarse elements on [0, 1]  (spacing 0.5)
#   Level 2: 4 fine   elements on [0, 1]  (spacing 0.25)
#   Active elements: level-1 element 1  ([0.0, 0.5])  +
#                    level-2 elements 3 ([0.5, 0.75]) and 4 ([0.75, 1.0])
#   Hierarchical ordering:
#     hier_id 1 → level 1, level_id 1  → [0.0,  0.5 ]
#     hier_id 2 → level 2, level_id 3  → [0.5,  0.75]
#     hier_id 3 → level 2, level_id 4  → [0.75, 1.0 ]
geo_l1_1d = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 3),))   # 2 elements
geo_l2_1d = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))   # 4 elements
active_1d  = Hierarchy.ActiveInfo([[1], [3, 4]])
hgeo_1d    = Geometry.HierarchicalGeometry((geo_l1_1d, geo_l2_1d), active_1d)

# 2D two-level hierarchy
#   Level 1: 2×2 = 4  coarse elements on [0,1]^2  (spacing 0.5)
#   Level 2: 4×4 = 16 fine   elements on [0,1]^2  (spacing 0.25)
#   Active elements: level-1 elements 1, 2, 3  +  level-2 element 16
#   Hierarchical ordering:
#     hier_id 1 → level 1, level_id 1  → x=[0.0,0.5 ], y=[0.0,0.5 ]
#     hier_id 2 → level 1, level_id 2  → x=[0.5,1.0 ], y=[0.0,0.5 ]
#     hier_id 3 → level 1, level_id 3  → x=[0.0,0.5 ], y=[0.5,1.0 ]
#     hier_id 4 → level 2, level_id 16 → x=[0.75,1.0], y=[0.75,1.0]
geo_l1_2d = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3)))
geo_l2_2d = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5), LinRange(0.0, 1.0, 5)))
active_2d  = Hierarchy.ActiveInfo([[1, 2, 3], [16]])
hgeo_2d    = Geometry.HierarchicalGeometry((geo_l1_2d, geo_l2_2d), active_2d)

############################################################################################
#                                       Constructor                                        #
############################################################################################

@testset "Constructor" begin
    # Type parameters are inferred from the level geometries.
    @test hgeo_1d isa Geometry.HierarchicalGeometry{1, 1, 1}
    @test hgeo_2d isa Geometry.HierarchicalGeometry{2, 2, 1}

    # Mismatched number of levels raises an ArgumentError.
    # wrong_active has 2 levels but only 1 geometry is provided.
    wrong_active = Hierarchy.ActiveInfo([[1, 2, 3], [11, 12, 15, 16]])
    @test_throws ArgumentError Geometry.HierarchicalGeometry((geo_l1_2d,), wrong_active)
end

############################################################################################
#                                         Getters                                          #
############################################################################################

@testset "Getters" begin
    # get_geometries returns the original tuple of level geometries.
    @test Geometry.get_geometries(hgeo_1d) === (geo_l1_1d, geo_l2_1d)
    @test Geometry.get_geometries(hgeo_2d) === (geo_l1_2d, geo_l2_2d)

    # get_active_elements returns the ActiveInfo object.
    @test Geometry.get_active_elements(hgeo_1d) === active_1d
    @test Geometry.get_active_elements(hgeo_2d) === active_2d

    # get_level_geometry returns the geometry for a specific level.
    @test Geometry.get_level_geometry(hgeo_1d, 1) === geo_l1_1d
    @test Geometry.get_level_geometry(hgeo_1d, 2) === geo_l2_1d
    @test Geometry.get_level_geometry(hgeo_2d, 2) === geo_l2_2d

    # get_num_levels reflects the number of entries in the geometry tuple.
    @test Geometry.get_num_levels(hgeo_1d) == 2
    @test Geometry.get_num_levels(hgeo_2d) == 2

    # get_num_elements returns the total number of active hierarchical elements.
    @test Geometry.get_num_elements(hgeo_1d) == 3   # 1 from level 1, 2 from level 2
    @test Geometry.get_num_elements(hgeo_2d) == 4   # 3 from level 1, 1 from level 2
end

############################################################################################
#                                       Indexing                                           #
############################################################################################

@testset "convert_to_level_and_level_id" begin
    # 1D: verify every hier_id maps to the correct (level, level_id).
    @test Geometry.convert_to_level_and_level_id(hgeo_1d, 1) == (1, 1)
    @test Geometry.convert_to_level_and_level_id(hgeo_1d, 2) == (2, 3)
    @test Geometry.convert_to_level_and_level_id(hgeo_1d, 3) == (2, 4)

    # 2D: first three are level-1 elements, last one is level-2.
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 1) == (1, 1)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 2) == (1, 2)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 3) == (1, 3)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 4) == (2, 16)
end

############################################################################################
#                                    Element properties                                    #
############################################################################################

@testset "get_element_lengths" begin
    # Coarse element: spacing 0.5
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_1d, 1), (0.5,),  rtol=1e-14))
    # Fine elements: spacing 0.25
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_1d, 2), (0.25,), rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_1d, 3), (0.25,), rtol=1e-14))

    # 2D: coarse spacing 0.5 in both directions
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 1), (0.5,  0.5 ), rtol=1e-14))
    # Fine element (level 2, elem 16): spacing 0.25 in both directions
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 4), (0.25, 0.25), rtol=1e-14))
end

@testset "get_element_measure" begin
    @test isapprox(Geometry.get_element_measure(hgeo_1d, 1), 0.5,    rtol=1e-14)
    @test isapprox(Geometry.get_element_measure(hgeo_1d, 2), 0.25,   rtol=1e-14)
    @test isapprox(Geometry.get_element_measure(hgeo_2d, 1), 0.25,   rtol=1e-14)  # 0.5×0.5
    @test isapprox(Geometry.get_element_measure(hgeo_2d, 4), 0.0625, rtol=1e-14)  # 0.25×0.25
end

@testset "get_element_vertices" begin
    # 1D: coarse element covers [0.0, 0.5]
    verts = Geometry.get_element_vertices(hgeo_1d, 1)
    @test all(isapprox.(verts[1], (0.0, 0.5), rtol=1e-14))
    # 1D: second fine element covers [0.5, 0.75]
    verts = Geometry.get_element_vertices(hgeo_1d, 2)
    @test all(isapprox.(verts[1], (0.5, 0.75), rtol=1e-14))
    # 1D: third fine element covers [0.75, 1.0]
    verts = Geometry.get_element_vertices(hgeo_1d, 3)
    @test all(isapprox.(verts[1], (0.75, 1.0), rtol=1e-14))

    # 2D: level-2 element 16 is at x=[0.75,1.0], y=[0.75,1.0]
    verts = Geometry.get_element_vertices(hgeo_2d, 4)
    @test all(isapprox.(verts[1], (0.75, 1.0), rtol=1e-14))
    @test all(isapprox.(verts[2], (0.75, 1.0), rtol=1e-14))
end

############################################################################################
#                                       Evaluation                                         #
############################################################################################

# Shared parametric evaluation points (3 points in [0,1]^d).
xi_1d = Points.CartesianPoints(([0.0, 0.5, 1.0],))
xi_2d = Points.CartesianPoints(([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]))

@testset "evaluate" begin
    # Results must match direct evaluation on the corresponding level geometry.
    for (hier_id, level, level_id) in [(1, 1, 1), (2, 2, 3), (3, 2, 4)]
        ref = Geometry.evaluate(Geometry.get_level_geometry(hgeo_1d, level), level_id, xi_1d)
        got = Geometry.evaluate(hgeo_1d, hier_id, xi_1d)
        @test all(isapprox.(got, ref, rtol=1e-14))
    end

    # 2D: test one coarse and the single fine element.
    for (hier_id, level, level_id) in [(1, 1, 1), (4, 2, 16)]
        ref = Geometry.evaluate(Geometry.get_level_geometry(hgeo_2d, level), level_id, xi_2d)
        got = Geometry.evaluate(hgeo_2d, hier_id, xi_2d)
        @test all(isapprox.(got, ref, rtol=1e-14))
    end
end

@testset "jacobian" begin
    # Cartesian geometry: Jacobian is diagonal (scaling), constant over points.
    # Coarse 1D element: J = [[0.5]]
    jac_coarse = Geometry.jacobian(hgeo_1d, 1, xi_1d)
    for p in eachindex(jac_coarse)
        @test all(isapprox.(jac_coarse[p], [0.5;;], rtol=1e-14))
    end

    # Fine 1D element: J = [[0.25]]
    jac_fine = Geometry.jacobian(hgeo_1d, 2, xi_1d)
    for p in eachindex(jac_fine)
        @test all(isapprox.(jac_fine[p], [0.25;;], rtol=1e-14))
    end

    # Must agree with direct evaluation on the level geometry.
    for (hier_id, level, level_id) in [(1, 1, 1), (2, 2, 3), (3, 2, 4)]
        ref = Geometry.jacobian(Geometry.get_level_geometry(hgeo_1d, level), level_id, xi_1d)
        got = Geometry.jacobian(hgeo_1d, hier_id, xi_1d)
        for p in eachindex(ref)
            @test all(isapprox.(got[p], ref[p], rtol=1e-14))
        end
    end
end

@testset "hessian" begin
    # Cartesian geometry is affine: Hessian is identically zero.
    for hier_id in 1:Geometry.get_num_elements(hgeo_1d)
        hess = Geometry.hessian(hgeo_1d, hier_id, xi_1d)
        for p in eachindex(hess)
            @test all(isapprox.(hess[p][1], [0.0;;], atol=1e-14))
        end
    end

    # Must agree with direct evaluation on the level geometry.
    for (hier_id, level, level_id) in [(1, 1, 1), (4, 2, 16)]
        ref = Geometry.hessian(Geometry.get_level_geometry(hgeo_2d, level), level_id, xi_2d)
        got = Geometry.hessian(hgeo_2d, hier_id, xi_2d)
        for p in eachindex(ref)
            for d in eachindex(ref[p])
                @test all(isapprox.(got[p][d], ref[p][d], atol=1e-14))
            end
        end
    end
end

end

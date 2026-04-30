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
active_1d = Hierarchy.ActiveInfo([[1], [3, 4]])
hgeo_1d = Geometry.HierarchicalGeometry((geo_l1_1d, geo_l2_1d), active_1d)

# 2D two-level hierarchy
#   Level 1: 2×2 = 4  coarse elements on [0,1]^2  (spacing 0.5)
#   Level 2: 4×4 = 16 fine   elements on [0,1]^2  (spacing 0.25)
#   Active elements: level-1 elements 1, 2, 3  +  level-2 element 16
#   Hierarchical ordering:
#     hier_id 1 → level 1, level_id 1  → x=[0.00,0.50], y=[0.00,0.50]
#     hier_id 2 → level 1, level_id 2  → x=[0.50,1.00], y=[0.00,0.50]
#     hier_id 3 → level 1, level_id 3  → x=[0.00,0.50], y=[0.50,1.00]
#     hier_id 4 → level 2, level_id 11 → x=[0.50,0.75], y=[0.75,0.75]
#     hier_id 5 → level 2, level_id 12 → x=[0.75,1.00], y=[0.50,0.75]
#     hier_id 6 → level 2, level_id 15 → x=[0.50,0.75], y=[0.75,1.00]
#     hier_id 7 → level 2, level_id 16 → x=[0.75,1.00], y=[0.75,1.00]
geo_l1_2d = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3)))
geo_l2_2d = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5), LinRange(0.0, 1.0, 5)))
active_2d = Hierarchy.ActiveInfo([[1, 2, 3], [11, 12, 15, 16]])
hgeo_2d = Geometry.HierarchicalGeometry((geo_l1_2d, geo_l2_2d), active_2d)

############################################################################################
#                                       Constructor                                        #
############################################################################################

@testset "Constructor" begin
    # Type parameters are inferred from the level geometries.
    @inferred Geometry.HierarchicalGeometry((geo_l1_1d, geo_l2_1d), active_1d)
    @inferred Geometry.HierarchicalGeometry((geo_l1_2d, geo_l2_2d), active_2d)

    # Mismatched number of levels raises an ArgumentError.
    # active_2d has 2 levels but only 1 geometry is provided.
    @test_throws ArgumentError Geometry.HierarchicalGeometry((geo_l1_2d,), active_2d)
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
    @test Geometry.get_level_geometry(hgeo_2d, 1) === geo_l1_2d
    @test Geometry.get_level_geometry(hgeo_2d, 2) === geo_l2_2d

    # get_num_levels reflects the number of entries in the geometry tuple.
    @test Geometry.get_num_levels(hgeo_1d) == 2
    @test Geometry.get_num_levels(hgeo_2d) == 2

    # get_num_elements returns the total number of active hierarchical elements.
    @test Geometry.get_num_elements(hgeo_1d) == 3   # 1 from level 1, 2 from level 2
    @test Geometry.get_num_elements(hgeo_2d) == 7   # 3 from level 1, 4 from level 2
end

############################################################################################
#                                       Indexing                                           #
############################################################################################

@testset "Conversions" begin
    @test Geometry.convert_to_level_and_level_id(hgeo_1d, 1) == (1, 1)
    @test Geometry.convert_to_level_and_level_id(hgeo_1d, 2) == (2, 3)
    @test Geometry.convert_to_level_and_level_id(hgeo_1d, 3) == (2, 4)

    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 1) == (1, 1)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 2) == (1, 2)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 3) == (1, 3)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 4) == (2, 11)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 5) == (2, 12)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 6) == (2, 15)
    @test Geometry.convert_to_level_and_level_id(hgeo_2d, 7) == (2, 16)
end

############################################################################################
#                                    Element properties                                    #
############################################################################################

@testset "get_element_lengths" begin
    # Coarse element: spacing 0.5
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_1d, 1), (0.5,), rtol=1e-14))
    # Fine elements: spacing 0.25
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_1d, 2), (0.25,), rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_1d, 3), (0.25,), rtol=1e-14))

    # 2D: coarse spacing 0.5 in both directions
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 1), (0.5, 0.5), rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 2), (0.5, 0.5), rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 3), (0.5, 0.5), rtol=1e-14))
    # Fine element (level 2, elem 16): spacing 0.25 in both directions
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 4), (0.25, 0.25), rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 5), (0.25, 0.25), rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 6), (0.25, 0.25), rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_lengths(hgeo_2d, 7), (0.25, 0.25), rtol=1e-14))
end

@testset "get_element_measure" begin
    @test isapprox(Geometry.get_element_measure(hgeo_1d, 1), 0.5, rtol=1e-14)
    @test isapprox(Geometry.get_element_measure(hgeo_1d, 2), 0.25, rtol=1e-14)
    @test isapprox(Geometry.get_element_measure(hgeo_1d, 3), 0.25, rtol=1e-14)

    @test isapprox(Geometry.get_element_measure(hgeo_2d, 1), 0.25, rtol=1e-14)  # 0.5×0.5
    @test isapprox(Geometry.get_element_measure(hgeo_2d, 2), 0.25, rtol=1e-14)  # 0.5×0.5
    @test isapprox(Geometry.get_element_measure(hgeo_2d, 3), 0.25, rtol=1e-14)  # 0.5×0.5
    @test isapprox(Geometry.get_element_measure(hgeo_2d, 4), 0.0625, rtol=1e-14)  # 0.25×0.25
    @test isapprox(Geometry.get_element_measure(hgeo_2d, 5), 0.0625, rtol=1e-14)  # 0.25×0.25
    @test isapprox(Geometry.get_element_measure(hgeo_2d, 6), 0.0625, rtol=1e-14)  # 0.25×0.25
    @test isapprox(Geometry.get_element_measure(hgeo_2d, 7), 0.0625, rtol=1e-14)  # 0.25×0.25
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

    # 2D: level-1 element 1 is at x=[0.00,0.50], y=[0.00,0.50]
    verts = Geometry.get_element_vertices(hgeo_2d, 1)
    @test all(isapprox.(verts[1], (0.0, 0.5), rtol=1e-14))
    @test all(isapprox.(verts[2], (0.0, 0.5), rtol=1e-14))
    # 2D: level-1 element 2 is at x=[0.50,1.00], y=[0.00,0.50]
    verts = Geometry.get_element_vertices(hgeo_2d, 2)
    @test all(isapprox.(verts[1], (0.5, 1.0), rtol=1e-14))
    @test all(isapprox.(verts[2], (0.0, 0.5), rtol=1e-14))
    # 2D: level-1 element 3 is at x=[0.00,0.50], y=[0.50,1.00]
    verts = Geometry.get_element_vertices(hgeo_2d, 3)
    @test all(isapprox.(verts[1], (0.0, 0.5), rtol=1e-14))
    @test all(isapprox.(verts[2], (0.5, 1.0), rtol=1e-14))
    # 2D: level-2 element 11 is at x=[0.50,0.75], y=[0.50,0.75]
    verts = Geometry.get_element_vertices(hgeo_2d, 4)
    @test all(isapprox.(verts[1], (0.5, 0.75), rtol=1e-14))
    @test all(isapprox.(verts[2], (0.5, 0.75), rtol=1e-14))
    # 2D: level-2 element 12 is at x=[0.75,1.0], y=[0.50,0.75]
    verts = Geometry.get_element_vertices(hgeo_2d, 5)
    @test all(isapprox.(verts[1], (0.75, 1.0), rtol=1e-14))
    @test all(isapprox.(verts[2], (0.5, 0.75), rtol=1e-14))
    # 2D: level-2 element 15 is at x=[0.50,0.75], y=[0.75,1.0]
    verts = Geometry.get_element_vertices(hgeo_2d, 6)
    @test all(isapprox.(verts[1], (0.5, 0.75), rtol=1e-14))
    @test all(isapprox.(verts[2], (0.75, 1.0), rtol=1e-14))
    # 2D: level-2 element 16 is at x=[0.75,1.0], y=[0.75,1.0]
    verts = Geometry.get_element_vertices(hgeo_2d, 7)
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
    for (points, geo) in zip((xi_1d, xi_2d), (hgeo_1d, hgeo_2d))
        for el_id in 1:Geometry.get_num_elements(geo)
            lvl, lvl_id = Geometry.convert_to_level_and_level_id(geo, el_id)
            @test all(
                isapprox.(
                    Geometry.evaluate(geo, el_id, points),
                    Geometry.evaluate(
                        Geometry.get_level_geometry(geo, lvl), lvl_id, points
                    ),
                ),
            )
        end
    end
end

@testset "jacobian" begin
    for (points, geo) in zip((xi_1d, xi_2d), (hgeo_1d, hgeo_2d))
        for el_id in 1:Geometry.get_num_elements(geo)
            lvl, lvl_id = Geometry.convert_to_level_and_level_id(geo, el_id)
            @test all(
                isapprox.(
                    Geometry.jacobian(geo, el_id, points),
                    Geometry.jacobian(
                        Geometry.get_level_geometry(geo, lvl), lvl_id, points
                    ),
                ),
            )
        end
    end
end

@testset "hessian" begin
    for (points, geo) in zip((xi_1d, xi_2d), (hgeo_1d, hgeo_2d))
        for el_id in 1:Geometry.get_num_elements(geo)
            lvl, lvl_id = Geometry.convert_to_level_and_level_id(geo, el_id)
            h_eval = Geometry.hessian(geo, el_id, points)
            lvl_eval = Geometry.hessian(
                Geometry.get_level_geometry(geo, lvl), lvl_id, points
            )
            for i in eachindex(h_eval)
                @test all(isapprox.(h_eval[i], lvl_eval[i]))
            end
        end
    end
end

end

module HierarchicalGeometryTests

using Mantis
using Mantis.Hierarchical
using Test

############################################################################################
#                                          Setup                                           #
############################################################################################

# 1D
geo_l1_1d = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 3),))
geo_l2_1d = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))

active_1d = ActiveInfo([[1], [3, 4]])
relations_1d = Relations(
    Geometry.parent_to_children_uniform(geo_l1_1d, 2),
    Geometry.child_to_parents_uniform(geo_l2_1d, 2),
)
scaling_1d = Scaling(geo_l1_1d, geo_l2_1d, relations_1d)

hierarchy_1d = NestedHierarchy(active_1d, scaling_1d)
hgeo_1d = Geometry.HierarchicalGeometry(hierarchy_1d)

# 2D
geo_l1_2d = Geometry.TensorProductGeometry((geo_l1_1d, geo_l1_1d))
geo_l2_2d = Geometry.TensorProductGeometry((geo_l2_1d, geo_l2_1d))
scaling_2d = Scaling(geo_l1_2d, geo_l2_2d, (scaling_1d, scaling_1d))
active_2d = ActiveInfo([[1, 2, 3], [11, 12, 15, 16]])
hierarchy_2d = NestedHierarchy(active_2d, scaling_2d)

hgeo_2d = Geometry.HierarchicalGeometry(hierarchy_2d)

############################################################################################
#                                       Constructor                                        #
############################################################################################

@testset "Constructor" begin
    @test @inferred(Geometry.HierarchicalGeometry(hierarchy_1d)) === hgeo_1d
    @test @inferred(Geometry.HierarchicalGeometry(hierarchy_2d)) === hgeo_2d
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

    @test Geometry.get_hierarchy(hgeo_1d) === hierarchy_1d
    @test Geometry.get_hierarchy(hgeo_2d) === hierarchy_2d
end

############################################################################################
#                                       Indexing                                           #
############################################################################################

@testset "Conversions" begin
    function level_and_level_id(expected, hgeo)
        for k in keys(expected)
            val = Geometry.convert_to_level_and_level_id(hgeo, k)
            if val != expected[k]
                println(stderr, "Error on key $(k): expected $(expected[k]), got $(val)")
                return false
            end
        end

        return true
    end

    expected_1d = Dict(1 => (1, 1), 2 => (2, 3), 3 => (2, 4))
    @test level_and_level_id(expected_1d, hgeo_1d)

    expected_2d = Dict(
        1 => (1, 1),
        2 => (1, 2),
        3 => (1, 3),
        4 => (2, 11),
        5 => (2, 12),
        6 => (2, 15),
        7 => (2, 16),
    )
    @test level_and_level_id(expected_2d, hgeo_2d)
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
xi_1d = Points.TensorProductPoints(([0.0, 0.5, 1.0],))
xi_2d = Points.TensorProductPoints(([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]))

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

@testset "Type stability" begin
    @test_nowarn @inferred Geometry.get_hierarchy(hgeo_1d)
    @test_nowarn @inferred Geometry.get_geometries(hgeo_1d)
    @test_nowarn @inferred Geometry.get_active_elements(hgeo_1d)
    @test_nowarn @inferred Geometry.get_level_geometry(hgeo_1d, 1)
    @test_nowarn @inferred Geometry.get_num_levels(hgeo_1d)
    @test_nowarn @inferred Geometry.get_num_elements(hgeo_1d)
    @test_nowarn @inferred Geometry.convert_to_level_and_level_id(hgeo_1d, 2)
    @test_nowarn @inferred Geometry.get_element_measure(hgeo_1d, 2)
    @test_nowarn @inferred Geometry.get_element_lengths(hgeo_1d, 2)
    @test_nowarn @inferred Geometry.get_element_vertices(hgeo_1d, 2)
end

end

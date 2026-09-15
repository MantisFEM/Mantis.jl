module SkeletonGeometryTests

using Mantis
using Test

############################################################################################
#                                         Helpers                                          #
############################################################################################
"""
    corner_points(dim)

Points at the corners of the canonical `dim`-dimensional element, as a
`Points.TensorProductPoints`.
"""
corner_points(dim) = Points.TensorProductPoints(ntuple(_ -> [0.0, 1.0], dim))

"""
    evaluated_points(geometry, element_id, xi)

Return the evaluation of `geometry` on `element_id` as a vector of coordinate tuples, which
is easier to compare than the `(num_points, image_dim)` matrix that `evaluate` returns.
"""
function evaluated_points(geometry, element_id, xi)
    values = Geometry.evaluate(geometry, element_id, xi)
    return [Tuple(values[i, :]) for i in axes(values, 1)]
end

############################################################################################
#                            1D skeleton, i.e. of a 2D geometry                            #
############################################################################################
@testset "skeleton of a 2D geometry" begin
    # Two unit squares meeting along x = 1, so the mesh spans [0, 2] x [0, 1] and has
    # 7 edges: 3 vertical and 4 horizontal.
    topology = Topology.MeshTopology([(1, 2, 3, 4), (2, 5, 6, 3)], Topology.QUAD)
    parent = Geometry.CartesianGeometry(
        (
            (LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3)),
            (LinRange(1.0, 2.0, 3), LinRange(0.0, 1.0, 3)),
        ),
        topology,
    )
    skeleton = Geometry.SkeletonGeometry(parent)

    @test Geometry.get_manifold_dim(skeleton) == 1
    @test Geometry.get_parent_geometry(skeleton) === parent
    # The patches of the skeleton are the edges of the parent.
    @test Geometry.get_num_patches(skeleton) == size(topology, 2) == 7
    # Each parent patch is 2 x 2, so every edge carries 2 elements.
    @test Geometry.get_num_elements(skeleton) == 14

    xi = corner_points(1)

    @testset "every skeleton element lies on a parent edge" begin
        for element_id in 1:Geometry.get_num_elements(skeleton)
            points = evaluated_points(skeleton, element_id, xi)
            @test length(points) == 2

            # An edge of an axis-aligned mesh is constant along one coordinate, and its
            # points must stay inside the parent domain.
            constant_x = points[1][1] == points[2][1]
            constant_y = points[1][2] == points[2][2]
            @test constant_x || constant_y
            @test all(p -> 0.0 <= p[1] <= 2.0 && 0.0 <= p[2] <= 1.0, points)
        end
    end

    @testset "the skeleton covers the parent's edges" begin
        # Collect every evaluated point; the four corners of the mesh have to appear.
        all_points = Set{Tuple{Float64, Float64}}()
        for element_id in 1:Geometry.get_num_elements(skeleton)
            union!(all_points, evaluated_points(skeleton, element_id, xi))
        end
        for corner in ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0))
            @test corner in all_points
        end
        # The interface x = 1 is a single skeleton patch, not one per parent patch.
        @test (1.0, 0.0) in all_points
    end
end

############################################################################################
#                            2D skeleton, i.e. of a 3D geometry                            #
############################################################################################
@testset "skeleton of a 3D geometry" begin
    # A single unit cube: its skeleton is the six faces of the cube.
    parent = Geometry.CartesianGeometry(
        (LinRange(0.0, 1.0, 2), LinRange(0.0, 1.0, 2), LinRange(0.0, 1.0, 2))
    )
    skeleton = Geometry.SkeletonGeometry(parent)

    @test Geometry.get_manifold_dim(skeleton) == 2
    @test Geometry.get_num_patches(skeleton) == 6
    @test Geometry.get_num_elements(skeleton) == 6

    xi = corner_points(2)

    @testset "each face is a distinct side of the cube" begin
        # A face of the unit cube is flat: exactly one coordinate is constant, and it is
        # either 0 or 1. Record which side each skeleton face lies on.
        sides = Set{Tuple{Int, Float64}}()
        for element_id in 1:6
            points = evaluated_points(skeleton, element_id, xi)
            @test length(points) == 4
            @test length(Set(points)) == 4 # the four corners are distinct

            constant_dims = [
                dim for dim in 1:3 if all(p -> p[dim] == points[1][dim], points)
            ]
            @test length(constant_dims) == 1
            dim = only(constant_dims)
            @test points[1][dim] in (0.0, 1.0)
            push!(sides, (dim, points[1][dim]))
        end

        # All six sides of the cube are covered, each exactly once.
        @test length(sides) == 6
    end
end

############################################################################################
#                                  Unsupported skeletons                                   #
############################################################################################
@testset "skeleton of a 1D geometry" begin
    # The skeleton of a 1D mesh would be a set of points, which is not a geometry.
    parent = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 3),))
    @test_throws ArgumentError Geometry.SkeletonGeometry(parent)
end

end

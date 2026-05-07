module CartesianPointsTests

using Mantis
using Test

@test_throws ArgumentError Points.CartesianPoints(())
@test_throws ArgumentError Points.CartesianPoints(([1], Int[]))
@test_throws ArgumentError Points.CartesianPoints(([1], ["wrong_type"]))
@test_throws TypeError Points.CartesianPoints(1:2, 1:2; iteration_order=(1, 2, 3))

manifold_dims = [1, 2, 3]
num_const_points = [(2,), (4, 3), (2, 1, 5)]
num_points = [prod(num_const_points[dim]) for dim in 1:3]
for i in 1:3
    manifold_dim = manifold_dims[i]
    points = ntuple(dim -> rand(num_const_points[i][dim]), manifold_dim)
    range = 1:num_points[i]
    xi = Points.CartesianPoints(points)

    @test Points.get_manifold_dim(xi) == manifold_dim
    @test firstindex(xi) == 1
    @test lastindex(xi) == num_points[i]
    @test keys(xi) == 1:num_points[i]
    @test Points.get_constituent_points(xi) == points
    @test Points.get_constituent_num_points(xi) == num_const_points[i]
    @test Points.get_num_points(xi) == num_points[i]
    for (j, original_point) in zip(eachindex(xi), Iterators.product(points...))
        @test j == range[j]
        @test xi[j] == original_point
    end
end

points = (1:2, 1:2, 1:2)
order_1 = (1, 2, 3)
order_2 = (1, 3, 2)
order_3 = (2, 1, 3)
order_4 = (2, 3, 1)
order_5 = (3, 1, 2)
order_6 = (3, 2, 1)
points_0 = Points.CartesianPoints(points)
points_1 = Points.CartesianPoints(points; iteration_order=order_1)
points_2 = Points.CartesianPoints(points; iteration_order=order_2)
points_3 = Points.CartesianPoints(points; iteration_order=order_3)
points_4 = Points.CartesianPoints(points; iteration_order=order_4)
points_5 = Points.CartesianPoints(points; iteration_order=order_5)
points_6 = Points.CartesianPoints(points; iteration_order=order_6)
num_points = Points.get_num_points(points_0)

for p in 1:num_points
    @test points_0[p] == points_1[p]
end

# points_1
@test points_1[1] == (1, 1, 1)
@test points_1[2] == (2, 1, 1)
@test points_1[3] == (1, 2, 1)
@test points_1[4] == (2, 2, 1)
@test points_1[5] == (1, 1, 2)
@test points_1[6] == (2, 1, 2)
@test points_1[7] == (1, 2, 2)
@test points_1[8] == (2, 2, 2)

# points_2
@test points_2[1] == (1, 1, 1)
@test points_2[2] == (2, 1, 1)
@test points_2[3] == (1, 1, 2)
@test points_2[4] == (2, 1, 2)
@test points_2[5] == (1, 2, 1)
@test points_2[6] == (2, 2, 1)
@test points_2[7] == (1, 2, 2)
@test points_2[8] == (2, 2, 2)

# points_3
@test points_3[1] == (1, 1, 1)
@test points_3[2] == (1, 2, 1)
@test points_3[3] == (2, 1, 1)
@test points_3[4] == (2, 2, 1)
@test points_3[5] == (1, 1, 2)
@test points_3[6] == (1, 2, 2)
@test points_3[7] == (2, 1, 2)
@test points_3[8] == (2, 2, 2)

# points_4
@test points_4[1] == (1, 1, 1)
@test points_4[2] == (1, 1, 2)
@test points_4[3] == (2, 1, 1)
@test points_4[4] == (2, 1, 2)
@test points_4[5] == (1, 2, 1)
@test points_4[6] == (1, 2, 2)
@test points_4[7] == (2, 2, 1)
@test points_4[8] == (2, 2, 2)

# points_5
@test points_5[1] == (1, 1, 1)
@test points_5[2] == (1, 2, 1)
@test points_5[3] == (1, 1, 2)
@test points_5[4] == (1, 2, 2)
@test points_5[5] == (2, 1, 1)
@test points_5[6] == (2, 2, 1)
@test points_5[7] == (2, 1, 2)
@test points_5[8] == (2, 2, 2)

# points_6
@test points_6[1] == (1, 1, 1)
@test points_6[2] == (1, 1, 2)
@test points_6[3] == (1, 2, 1)
@test points_6[4] == (1, 2, 2)
@test points_6[5] == (2, 1, 1)
@test points_6[6] == (2, 1, 2)
@test points_6[7] == (2, 2, 1)
@test points_6[8] == (2, 2, 2)

# Test different number of points

points = Mantis.Points.CartesianPoints(1:2, 1:3, 1:4; iteration_order=(3, 1, 2))
@test points[1] == (1, 1, 1)
@test points[2] == (1, 2, 1)
@test points[3] == (1, 3, 1)
@test points[4] == (1, 1, 2)
@test points[5] == (1, 2, 2)
@test points[6] == (1, 3, 2)
@test points[7] == (1, 1, 3)
@test points[8] == (1, 2, 3)
@test points[9] == (1, 3, 3)
@test points[10] == (1, 1, 4)
@test points[11] == (1, 2, 4)
@test points[12] == (1, 3, 4)
@test points[13] == (2, 1, 1)
@test points[14] == (2, 2, 1)
@test points[15] == (2, 3, 1)
@test points[16] == (2, 1, 2)
@test points[17] == (2, 2, 2)
@test points[18] == (2, 3, 2)
@test points[19] == (2, 1, 3)
@test points[20] == (2, 2, 3)
@test points[21] == (2, 3, 3)
@test points[22] == (2, 1, 4)
@test points[23] == (2, 2, 4)
@test points[24] == (2, 3, 4)

# Test heterogeneous types
@test typeof(Points.CartesianPoints([1, 2], [1.0, 2.0]).constituent_points) ==
    Tuple{Vector{Float64}, Vector{Float64}}
@test typeof(Points.CartesianPoints(LinRange(0, 1, 2), [1.0, 2.0]).constituent_points) ==
    Tuple{LinRange{Float64, Int}, Vector{Float64}}
@test typeof(Points.CartesianPoints(LinRange(0, 1, 2), [1, 2]).constituent_points) ==
    Tuple{LinRange{Float64, Int}, Vector{Float64}}
@test typeof(
    Points.CartesianPoints(
        1:2, [1, 2], LinRange(0, 1, 2), zeros(Float32, 2)
    ).constituent_points,
) == Tuple{Vector{Float64}, Vector{Float64}, LinRange{Float64, Int}, Vector{Float64}}

end

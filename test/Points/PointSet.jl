module PointSetTests

using Mantis
using Test

@test_throws ArgumentError Points.PointSet([1, 2], [1, 2, 3])

manifold_dims = [1, 2, 3]
num_points = [7, 8, 9]
for i in 1:3
    manifold_dim = manifold_dims[i]
    points = [tuple(rand(manifold_dim)...) for i in 1:num_points[i]]
    range = 1:num_points[i]
    xi = Points.PointSet(points)

    @test Points.get_manifold_dim(xi) == manifold_dim
    @test firstindex(xi) == 1
    @test lastindex(xi) == num_points[i]
    @test keys(xi) == 1:num_points[i]
    @test Points.get_num_points(xi) == num_points[i]
    for (j, original_point) in zip(eachindex(xi), points)
        @test j == range[j]
        @test xi[j] == original_point
    end
end

# Test heterogeneous types
@test typeof(Points.PointSet([1, 2], [1.0, 2.0]).input_points) ==
    Tuple{Vector{Float64}, Vector{Float64}}
@test typeof(Points.PointSet(LinRange(0, 1, 2), [1.0, 2.0]).input_points) ==
    Tuple{LinRange{Float64, Int}, Vector{Float64}}
@test typeof(Points.PointSet(LinRange(0, 1, 2), [1, 2]).input_points) ==
    Tuple{LinRange{Float64, Int}, Vector{Float64}}
@test typeof(
    Points.PointSet(1:2, [1, 2], LinRange(0, 1, 2), zeros(Float32, 2)).input_points
) == Tuple{Vector{Float64}, Vector{Float64}, LinRange{Float64, Int}, Vector{Float64}}

end

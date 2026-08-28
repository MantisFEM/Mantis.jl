module AbstractPointsTests

using Mantis
using Test

struct UnknownPoints{N, T} <: Points.AbstractPoints{N, T}
    points::NTuple{N, Vector{T}}
end

unknown_points = UnknownPoints(([1.0, 2.0],))

@test_throws MethodError Points.get_num_points(unknown_points)
@test isone(Points.get_manifold_dim(unknown_points))
@test eltype(unknown_points) == Float64

@test_throws ArgumentError Points._construction_checks(())
@test_throws ArgumentError Points._construction_checks(([1], Int[]))
@test_throws MethodError Points._construction_checks(([1], ["1"]))

end

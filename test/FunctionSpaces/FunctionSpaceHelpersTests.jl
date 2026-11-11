module FunctionSpaceHelpersTests

using Mantis

using Test

# Points of the canonical element at which the spaces are compared.
function canonical_points(manifold_dim)
    return Points.TensorProductPoints(ntuple(_ -> [0.0, 0.3, 1.0], manifold_dim))
end

# The dimension of a univariate B-spline space with `N` elements, degree `p` and regularity
# `k` at the interior breakpoints.
bspline_dim(N, p, k) = N * (p + 1) - (k + 1) * (N - 1)

# Checks that `space`, created on `geometry` with `create_bspline_space`, has `geometry` as
# its physical geometry and the same basis functions as `reference`, a space with the same
# breakpoints, degrees and regularities that was created without `geometry`.
function check_space(space, geometry, reference)
    @test FunctionSpaces.get_geometry(space) === geometry
    @test FunctionSpaces.get_parametric_geometry(space) ==
        Geometry.get_parametric_geometry(geometry)
    @test FunctionSpaces.get_num_basis(space) == FunctionSpaces.get_num_basis(reference)
    @test FunctionSpaces.get_dof_partition(space) ==
        FunctionSpaces.get_dof_partition(reference)
    ξ = canonical_points(FunctionSpaces.get_manifold_dim(space))
    for element_id in 1:FunctionSpaces.get_num_elements(space)
        values, basis_ids = FunctionSpaces.evaluate(space, element_id, ξ, 1)
        reference_values, reference_basis_ids = FunctionSpaces.evaluate(
            reference, element_id, ξ, 1
        )
        @test basis_ids == reference_basis_ids
        @test values ≈ reference_values
    end

    return nothing
end

@testset "Curve in the plane" begin
    # The half circle θ ↦ (cos θ, sin θ), with θ ∈ [0, π].
    half_circle = Geometry.MappedGeometry(
        Geometry.create_cartesian_box((0.0,), (1.0π,), (16,)),
        Geometry.Mapping((1, 2), x -> [cos(x[1]), sin(x[1])], x -> [-sin(x[1]), cos(x[1])]),
    )
    space = FunctionSpaces.create_bspline_space(half_circle, (2,), (1,))
    @test space isa FunctionSpaces.BSplineSpace
    @test FunctionSpaces.get_num_basis(space) == bspline_dim(16, 2, 1)
    reference = FunctionSpaces.create_bspline_space(0.0, 1.0π, 16, 2, 1)
    check_space(space, half_circle, reference)
end

@testset "Curvilinear square" begin
    square = Geometry.create_curvilinear_square((0.0, 0.0), (1.0, 1.0), (4, 12))
    space = FunctionSpaces.create_bspline_space(square, (2, 3), (1, 0))
    @test space isa FunctionSpaces.TensorProductSpace
    @test FunctionSpaces.get_num_basis(space) ==
        bspline_dim(4, 2, 1) * bspline_dim(12, 3, 0)
    reference = FunctionSpaces.create_bspline_space(
        (0.0, 0.0), (1.0, 1.0), (4, 12), (2, 3), (1, 0)
    )
    check_space(space, square, reference)

    # The numbers of boundary degrees of freedom are passed on to the factor spaces.
    space = FunctionSpaces.create_bspline_space(
        square, (2, 3), (1, 0); n_dofs_left=(2, 1), n_dofs_right=(1, 2)
    )
    reference = FunctionSpaces.create_bspline_space(
        (0.0, 0.0),
        (1.0, 1.0),
        (4, 12),
        (2, 3),
        (1, 0);
        n_dofs_left=(2, 1),
        n_dofs_right=(1, 2),
    )
    check_space(space, square, reference)
end

@testset "Surface in space" begin
    # Half of a cylinder, (θ, z) ↦ (cos θ, sin θ, z), with (θ, z) ∈ [0, π] × [0, 1].
    cylinder = Geometry.MappedGeometry(
        Geometry.create_cartesian_box((0.0, 0.0), (1.0π, 1.0), (8, 3)),
        Geometry.Mapping(
            (2, 3),
            x -> [cos(x[1]), sin(x[1]), x[2]],
            x -> [
                -sin(x[1]) 0.0
                cos(x[1]) 0.0
                0.0 1.0
            ],
        ),
    )
    space = FunctionSpaces.create_bspline_space(cylinder, (2, 2), (1, 1))
    @test Geometry.get_image_dim(FunctionSpaces.get_geometry(space)) == 3
    @test FunctionSpaces.get_num_basis(space) == bspline_dim(8, 2, 1) * bspline_dim(3, 2, 1)
    reference = FunctionSpaces.create_bspline_space(
        (0.0, 0.0), (1.0π, 1.0), (8, 3), (2, 2), (1, 1)
    )
    check_space(space, cylinder, reference)
end

@testset "Curvilinear cube" begin
    cube = Geometry.MappedGeometry(
        Geometry.create_cartesian_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 3, 2)),
        Geometry.create_curvilinear_mapping((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.1),
    )
    space = FunctionSpaces.create_bspline_space(cube, (2, 2, 1), (1, 1, 0))
    @test FunctionSpaces.get_num_basis(space) ==
        bspline_dim(2, 2, 1) * bspline_dim(3, 2, 1) * bspline_dim(2, 1, 0)
    reference = FunctionSpaces.create_bspline_space(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 3, 2), (2, 2, 1), (1, 1, 0)
    )
    check_space(space, cube, reference)
end

@testset "Non-uniform Cartesian geometry" begin
    breakpoints = ([0.0, 0.2, 0.6, 1.2, 2.0], [0.0, 0.5, 1.0])
    geometry = Geometry.CartesianGeometry(breakpoints)
    space = FunctionSpaces.create_bspline_space(geometry, (3, 2), (2, 1))
    reference = FunctionSpaces.TensorProductSpace((
        FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry((breakpoints[1],)), 3, 2),
        FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry((breakpoints[2],)), 2, 1),
    ))
    check_space(space, geometry, reference)
end

@testset "Geometry without a parametric geometry" begin
    line = Geometry.create_cartesian_box((0.0,), (1.0,), (3,))
    geometry = Geometry.TensorProductGeometry((line, line))
    @test_throws MethodError FunctionSpaces.create_bspline_space(geometry, (2, 2), (1, 1))
end

end

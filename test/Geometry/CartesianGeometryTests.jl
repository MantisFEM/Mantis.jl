module CartesianGeometryTests

using Mantis

import ReadVTK
using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")

# Constructor, property, and getters and setters tests -------------------------------------
function basic_tests(geometry, answers)
    @test Geometry.get_breakpoints(geometry) == answers[1]  # Breakpoints on first patch.
    @test collect(Geometry.get_constituent_num_elements(geometry)) == answers[2]

    @test Geometry.get_num_patches(geometry) == answers[3]
    @test Geometry.get_num_elements(geometry) == answers[4]
    @test Geometry.get_manifold_dim(geometry) == answers[5]
    @test Geometry.get_image_dim(geometry) == answers[6]
    @test all(Geometry.get_num_elements_per_patch(geometry) .== answers[7])
    @test Geometry.get_num_elements(geometry, 1) == answers[8]
    @test all(isapprox.(Geometry.get_element_lengths(geometry, 1), answers[9], rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_measure(geometry, 1), answers[10], rtol=1e-14))

    patch_id, local_element_id = Geometry.get_patch_and_local_element_id(
        geometry, answers[12]
    )
    @test (patch_id, local_element_id) == answers[11]
    @test Geometry.get_global_element_id(geometry, patch_id, local_element_id) ==
        answers[12]

    @test all(
        all.([
            isapprox.(
                Geometry.get_element_vertices(geometry, 1)[i], answers[13][i], rtol=1e-14
            ) for i in eachindex(answers[13])
        ]),
    )

    return nothing
end

# Reduction test, single-patch, single element, 1D.
geometry1 = Geometry.CartesianGeometry(([-1, 1],))
answers_1 = (([-1, 1],), [(1,)], 1, 1, 1, 1, (1,), 1, (2.0,), 2.0, (1, 1), 1, ((-1, 1),))
basic_tests(geometry1, answers_1)

# Vector{Float64} input. Single-patch 3D.
geometryVF = Geometry.CartesianGeometry((
    [0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [-0.75, 0.0, 0.25, 0.75]
))
answers_VF = (
    ([0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [-0.75, 0.0, 0.25, 0.75]),
    [(2, 2, 3)],
    1,
    12,
    3,
    3,
    (12,),
    12,
    (1.0, 1.0, 0.75),
    0.75,
    (1, 4),
    4,
    ((0.0, 1.0), (0.5, 1.5), (-0.75, 0.0)),
)
basic_tests(geometryVF, answers_VF)

# LinRange input. Single-patch, 2D.
geometryLR = Geometry.CartesianGeometry((LinRange(0.5, 2.5, 5), LinRange(-0.75, 0.75, 3)))
answers_LR = (
    ([0.5, 1.0, 1.5, 2.0, 2.5], [-0.75, 0.0, 0.75]),
    [(4, 2)],
    1,
    8,
    2,
    2,
    (8,),
    8,
    (0.5, 0.75),
    0.375,
    (1, 7),
    7,
    ((0.5, 1.0), (-0.75, 0.0)),
)
basic_tests(geometryLR, answers_LR)

# LinRange input. Single-patch, 4D.
geometry1p4D = Geometry.CartesianGeometry((
    LinRange(0.5, 2.5, 5),
    LinRange(-0.75, 0.75, 3),
    LinRange(1.5, 2.5, 4),
    LinRange(10.5, 20.5, 6),
))
answers_1p4D = (
    (
        LinRange(0.5, 2.5, 5),
        LinRange(-0.75, 0.75, 3),
        LinRange(1.5, 2.5, 4),
        LinRange(10.5, 20.5, 6),
    ),
    [(4, 2, 3, 5)],
    1,
    120,
    4,
    4,
    (120,),
    120,
    (0.5, 0.75, 1.0 / 3.0, 2.0),
    0.25,
    (1, 120),
    120,
    ((0.5, 1.0), (-0.75, 0.0), (1.5, 1.5 + 1 / 3), (10.5, 12.5)),
)
basic_tests(geometry1p4D, answers_1p4D)

# Multi-patch input. 2 patches, 2D. Homogeneous input. First patch has more element than the
# second.
geometryMP2 = Geometry.CartesianGeometry((
    (LinRange(-0.5, 2.5, 16), LinRange(-0.5, 2.5, 31)),
    (LinRange(2.5, 3.0, 4), LinRange(-0.5, 2.5, 9)),
))
answers_MP2 = (
    ((LinRange(-0.5, 2.5, 16), LinRange(-0.5, 2.5, 31))),
    [(15, 30), (3, 8)],
    2,
    474,
    2,
    2,
    (450, 24),
    450,
    (0.2, 0.1),
    0.02,
    (2, 13),
    463,
    ((-0.5, -0.3), (-0.5, -0.4)),
)
basic_tests(geometryMP2, answers_MP2)

# Multi-patch input. 2 patches, 2D. Heterogeneous input.
geometryMP = Geometry.CartesianGeometry((
    (LinRange(0.5, 2.5, 5), [-0.75, 0.1, 0.75]),
    (LinRange(0.0, 1.0, 4), LinRange(0.0, 1.0, 6)),
))
answers_MP = (
    (([0.5, 1.0, 1.5, 2.0, 2.5], [-0.75, 0.1, 0.75])),
    [(4, 2), (3, 5)],
    2,
    23,
    2,
    2,
    (8, 15),
    8,
    (0.5, 0.85),
    0.425,
    (2, 15),
    23,
    ((0.5, 1.0), (-0.75, 0.1)),
)
basic_tests(geometryMP, answers_MP)
@test Geometry.get_breakpoints(geometryMP, 2) ==
    (([0.0, 1 / 3, 2 / 3, 1.0], [0.0, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 1.0]))
@test Geometry.get_patch_and_local_element_id(geometryMP, 10) == (2, 2)

for i in 1:Geometry.get_num_elements(geometryMP)
    jac = Geometry.jacobian(geometryMP, i, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0])))
    hess = Geometry.hessian(geometryMP, i, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0])))
    if i <= 4
        for p in axes(jac, 1)
            @test all(isapprox.(jac[p][:, :], [0.5 0.0; 0.0 0.85], rtol=1e-14))
        end
    elseif i <= 8
        for p in axes(jac, 1)
            @test all(isapprox.(jac[p][:, :], [0.5 0.0; 0.0 0.65], rtol=1e-14))
        end
    else
        for p in axes(jac, 1)
            @test all(isapprox.(jac[p][:, :], [1/3 0.0; 0.0 1/5], rtol=1e-14))
        end
    end
    for p in eachindex(hess)
        @test all(isapprox.(hess[p][1][:, :], [0.0 0.0; 0.0 0.0], atol=1e-14))
        @test all(isapprox.(hess[p][2][:, :], [0.0 0.0; 0.0 0.0], atol=1e-14))
    end
end

# Multi-patch input. 100 patches, 1D.
geometryMP100 = Geometry.CartesianGeometry(
    ntuple(100) do i
        return (LinRange((i - 1) * 1.0, i * 1.0, i + 1),)
    end,
)
answers_MP100 = (
    (LinRange(0.0, 1.0, 2),),
    [(i,) for i in 1:100],
    100,
    5050,
    1,
    1,
    Tuple(1:100),
    1,
    (1.0,),
    1.0,
    (100, 100),
    5050,
    ((0.0, 1.0),),
)
basic_tests(geometryMP100, answers_MP100)
@test Geometry.get_breakpoints(geometryMP100, 67) == (LinRange(66.0, 67.0, 68),)
@test Geometry.get_patch_id(geometryMP100, 10) == 4
@test Geometry.get_patch_and_local_element_id(geometryMP100, 10) == (4, 4)
@test Geometry.get_patch_and_local_element_id(geometryMP100, 12) == (5, 2)

all_jac_MP100 = true
all_hess_MP100 = true
for i in 1:Geometry.get_num_elements(geometryMP100)
    jac = Geometry.jacobian(geometryMP100, i, Points.CartesianPoints(([0.0, 1.0],)))
    hess = Geometry.hessian(geometryMP100, i, Points.CartesianPoints(([0.0, 1.0],)))

    for p in eachindex(jac, hess)
        if !all(isapprox.(jac[p][:, :], [1.0 / i], rtol=1e-14))
            all_jac_MP100 = false
        end
        if !all(isapprox.(hess[p][1][:, :], [0.0], rtol=1e-14))
            all_hess_MP100 = false
        end
    end
end
@test all_jac_MP100
@test all_hess_MP100

# Test errors:
# Element_id is too high
@test_throws ArgumentError Geometry.get_patch_id(geometryLR, 9)
@test_throws ArgumentError Geometry.get_patch_id(geometryMP100, 5051)
# Element_id is too high
@test_throws ArgumentError Geometry.get_patch_and_local_element_id(geometryLR, 9)
@test_throws ArgumentError Geometry.get_patch_and_local_element_id(geometryMP100, 5051)
# Patch id is too high
@test_throws ArgumentError Geometry.get_global_element_id(geometryLR, 2, 1)
@test_throws ArgumentError Geometry.get_global_element_id(geometryMP100, 101, 3)
# Element_id is too high for the geometry
@test_throws ArgumentError Geometry.get_global_element_id(geometryLR, 1, 12)
@test_throws ArgumentError Geometry.get_global_element_id(geometryMP100, 30, 6000)
# Element_id is too high for the patch (but not for the geometry).
@test_throws ArgumentError Geometry.get_global_element_id(geometryMP100, 30, 45)

# Comparison to reference data.
for nx in 1:3
    for ny in 1:3
        geometry = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 2.0), (nx, ny))

        # Set file name and path
        file_name = "cartesian_test_nx_$(nx)_ny_$(ny).vtu"
        output_file_path = Mantis.GeneralHelpers.export_path(
            output_directory_tree, file_name
        )
        # Generate the vtk file
        Plot.plot(
            geometry;
            vtk_filename=output_file_path[1:(end - 4)],  # Remove the file extension.
            n_subcells=1,
            degree=1,
            ascii=false,
            compress=false,
        )

        # Read the cell data from the reference file.
        reference_points, reference_cells = get_point_cell_data(
            reference_directory_tree, file_name
        )
        # Read the cell data from the output file.
        output_points, output_cells = get_point_cell_data(output_file_path)

        # Check if cell data is point-wise identical.
        @test all(isapprox.(reference_points, output_points; rtol=rtol))
        @test all(isequal.(reference_cells, output_cells))
    end
end

end

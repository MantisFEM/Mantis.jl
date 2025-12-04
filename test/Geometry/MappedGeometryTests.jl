module MappedGeometryTests

using Mantis

import ReadVTK
using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")

# Test MappedCartesianGeometry ------------------------------------------------
for nx in 1:3
    for ny in 1:3
        breakpoints = (
            collect(LinRange(0.0, 1.0, nx + 1)), collect(LinRange(0.0, 2.0, ny + 1))
        )
        geom = Geometry.CartesianGeometry(breakpoints)

        # Define the mapping ϕ of the geometry and its derivative.
        # ϕ(x,y) = [(x + 0.2)*cos(y), (x + 0.2)*sin(y)\
        function mapping(x::AbstractVector)
            return [(x[1] + 0.2) * cos(x[2]), (x[1] + 0.2) * sin(x[2])]
        end
        function dmapping(x::AbstractVector)
            return [cos(x[2]) -(x[1] + 0.2)*sin(x[2]); sin(x[2]) (x[1] + 0.2)*cos(x[2])]
        end

        dimension = (2, 2)
        curved_mapping = Geometry.Mapping(dimension, mapping, dmapping)
        mapped_geometry = Geometry.MappedGeometry(geom, curved_mapping)

        # Generate the plot
        file_name = "mapped_cartesian_test_nx_$(nx)_ny_$(ny).vtu"
        output_file_path = Mantis.GeneralHelpers.export_path(
            output_directory_tree, file_name
        )
        Plot.plot(
            mapped_geometry;
            vtk_filename=output_file_path[1:(end - 4)],
            n_subcells=1,
            degree=3,
            ascii=false,
            compress=false,
        )

        # Test geometry
        # Read the cell data from the reference file
        reference_points, reference_cells = get_point_cell_data(
            reference_directory_tree, file_name
        )
        # Read the cell data from the output file
        output_points, output_cells = get_point_cell_data(output_file_path)
        # Check if cell data is identical
        @test all(isapprox.(reference_points, output_points; rtol=rtol))
        @test all(isequal.(reference_cells, output_cells))
    end
end
# -----------------------------------------------------------------------------

# Constructor, property, and getters and setters tests -------------------------------------
function basic_tests(geometry, answers)
    @test Geometry.get_num_patches(geometry) == answers[1]
    @test Geometry.get_num_elements(geometry) == answers[2]
    @test Geometry.get_manifold_dim(geometry) == answers[3]
    @test Geometry.get_image_dim(geometry) == answers[4]
    @test all(Geometry.get_num_elements_per_patch(geometry) .== answers[5])
    @test Geometry.get_num_elements(geometry, 1) == answers[6]
    @test all(isapprox.(Geometry.get_element_lengths(geometry, 1), answers[7], rtol=1e-14))
    @test all(isapprox.(Geometry.get_element_measure(geometry, 1), answers[8], rtol=1e-14))

    patch_id, local_element_id = Geometry.get_patch_and_local_element_id(
        geometry, answers[10]
    )
    @test (patch_id, local_element_id) == answers[9]
    @test Geometry.get_global_element_id(geometry, patch_id, local_element_id) ==
        answers[10]

    @test all(
        all.([
            isapprox.(
                Geometry.get_element_vertices(geometry, 1)[i], answers[11][i], rtol=1e-14
            ) for i in eachindex(answers[11])
        ]),
    )

    return nothing
end

# Mappings to create the deformed geometries. The mappings are defined with reference
# to the unit square [0,1]x[0,1] as parametric domain.
function mapping_patch_1_slant(x::AbstractVector{Float64}, slant_factor=0.25)
    return [x[1] + slant_factor * x[1] * x[2], x[2]]
end
function dmapping_patch_1_slant(x::AbstractVector{Float64}, slant_factor=0.25)
    return [
        [1.0 + slant_factor * x[2] slant_factor * x[1]]
        [0.0 1.0]
    ]
end
function ddmapping_patch_1_slant(x::AbstractVector{Float64}, slant_factor=0.25)
    return (
        [
            [0.0 slant_factor]
            [slant_factor 0.0]
        ],
        [
            [0.0 0.0]
            [0.0 0.0]
        ],
    )
end
mapping_patch_1_slanted = Geometry.Mapping(
    (2, 2), mapping_patch_1_slant, dmapping_patch_1_slant, ddmapping_patch_1_slant
)
function mapping_patch_2_slant(x::AbstractVector{Float64}, slant_factor=0.25)
    return [x[1] + 1.0 + slant_factor * (1.0 - x[1]) * x[2], x[2]]
end
function dmapping_patch_2_slant(x::AbstractVector{Float64}, slant_factor=0.25)
    return [
        [1.0 - slant_factor * x[2] slant_factor * (1.0 - x[1])]
        [0.0 1.0]
    ]
end
function ddmapping_patch_2_slant(x::AbstractVector{Float64}, slant_factor=0.25)
    return (
        [
            [0.0 -slant_factor]
            [-slant_factor 0.0]
        ],
        [
            [0.0 0.0]
            [0.0 0.0]
        ],
    )
end
mapping_patch_2_slanted = Geometry.Mapping(
    (2, 2), mapping_patch_2_slant, dmapping_patch_2_slant, ddmapping_patch_1_slant
)
num_elements_per_dim_per_patch = ((4, 4), (5, 6))
geom_cart_patch_1 = Geometry.CartesianGeometry((
    0.0:(1.0 / num_elements_per_dim_per_patch[1][1]):1.0,
    0.0:(1.0 / num_elements_per_dim_per_patch[1][2]):1.0,
))
geom_cart_patch_2 = Geometry.CartesianGeometry((
    0.0:(1.0 / num_elements_per_dim_per_patch[2][1]):1.0,
    0.0:(1.0 / num_elements_per_dim_per_patch[2][2]):1.0,
))

# Reduction test
function mapping_I(x::AbstractVector{Float64}, slant_factor=0.25)
    return [x[1], x[2]]
end
function dmapping_I(x::AbstractVector{Float64}, slant_factor=0.25)
    return [
        [0.0 0.0]
        [0.0 0.0]
    ]
end
mapping_I_obj = Geometry.Mapping((1, 1), mapping_I, dmapping_I)
geometry1 = Geometry.MappedGeometry(
    Geometry.CartesianGeometry((LinRange(0.0, 1.0, 2),)), mapping_I_obj
)
answers_1 = (1, 1, 1, 1, (1,), 1, (1.0,), 1.0, (1, 1), 1, ((0.0, 1.0),))
basic_tests(geometry1, answers_1)

# Mapped, explicit geometry and mapping per patch.
geom_slanted_2patch = Geometry.MappedGeometry(
    (geom_cart_patch_1, geom_cart_patch_2),
    (mapping_patch_1_slanted, mapping_patch_2_slanted),
)
answers_geom_slanted_2patch = (
    2, 46, 2, 2, (16, 30), 16, (0.25, 0.25), 0.0625, (1, 5), 5, ((0.0, 0.25), (0.0, 0.25))
)
basic_tests(geom_slanted_2patch, answers_geom_slanted_2patch)

# Mapped, one parametric domain with multiple mappings.
geom_slanted_2patch_oneref = Geometry.MappedGeometry(
    Geometry.CartesianGeometry(((LinRange(0.0, 1.0, 5), LinRange(0.0, 1.0, 7)),)),
    (mapping_patch_1_slanted, mapping_patch_2_slanted),
)
answers_geom_slanted_2patch_oneref = (
    2,
    48,
    2,
    2,
    (24, 24),
    24,
    (0.25, 1.0 / 6.0),
    1.0 / 24.0,
    (2, 2),
    26,
    ((0.0, 0.25), (0.0, 1.0 / 6.0)),
)
basic_tests(geom_slanted_2patch_oneref, answers_geom_slanted_2patch_oneref)

# Mapped, multiple patches with one map.
geom_slanted_2patch_onemap = Geometry.MappedGeometry(
    (
        Geometry.CartesianGeometry(((LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),)),
        Geometry.CartesianGeometry(((LinRange(0.25, 0.5, 4), LinRange(0.0, 1.0, 7)),)),
        Geometry.CartesianGeometry(((LinRange(0.5, 0.75, 5), LinRange(0.0, 1.0, 7)),)),
        Geometry.CartesianGeometry(((LinRange(0.75, 1.0, 6), LinRange(0.0, 1.0, 7)),)),
    ),
    mapping_patch_1_slanted,
)
answers_geom_slanted_2patch_onemap = (
    4,
    84,
    2,
    2,
    (12, 18, 24, 30),
    12,
    (0.125, 1.0 / 6.0),
    1 / 48,
    (2, 3),
    15,
    ((0.0, 0.125), (0.0, 1.0 / 6.0)),
)
basic_tests(geom_slanted_2patch_onemap, answers_geom_slanted_2patch_onemap)

# Mapped, single mapping, single patch.
geom_slanted_2patch_11 = Geometry.MappedGeometry(geom_cart_patch_1, mapping_patch_1_slanted)
answers_geom_slanted_2patch_11 = (
    1, 16, 2, 2, (16,), 16, (0.25, 0.25), 0.0625, (1, 16), 16, ((0.0, 0.25), (0.0, 0.25))
)
basic_tests(geom_slanted_2patch_11, answers_geom_slanted_2patch_11)

# Surface embedded in 3D.
geo(x) = [x[1], x[2], x[1] * x[2]]
dgeo(x) = [[1.0 0.0]; [0.0 1.0]; [x[2] x[1]]]
ddgeo(x) = (
    [
        [0.0 0.0]
        [0.0 0.0]
    ],
    [
        [0.0 0.0]
        [0.0 0.0]
    ],
    [
        [0.0 1.0]
        [1.0 0.0]
    ],
)
mapping2to3 = Mantis.Geometry.Mapping((2, 3), geo, dgeo, ddgeo)
geometry2to3 = Mantis.Geometry.MappedGeometry(geom_cart_patch_1, mapping2to3)
answers_geometry2to3 = (
    1, 16, 2, 3, (16,), 16, (0.25, 0.25), 0.0625, (1, 14), 14, ((0.0, 0.25), (0.0, 0.25))
)
basic_tests(geometry2to3, answers_geometry2to3)

for (k, IJ) in enumerate(CartesianIndices((4, 4)))
    jac = Geometry.jacobian(
        geometry2to3, k, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
    )
    hess = Geometry.hessian(
        geometry2to3, k, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
    )

    i, j = Tuple(IJ)
    xans = [
        x_i for _ in (1, 2) for x_i in LinRange((i - 1) * 1.0 / 16.0, i * 1.0 / 16.0, 2)
    ]
    yans = [
        y_i for y_i in LinRange((j - 1) * 1.0 / 16.0, j * 1.0 / 16.0, 2) for _ in (1, 2)
    ]
    jactest = true
    for p in eachindex(jac)
        if !all(isapprox.(jac[p][:, :], [0.25 0.0; 0.0 0.25; yans[p] xans[p]], rtol=1e-14))
            jactest = false
        end
    end
    @test jactest

    hesstest = true
    for p in eachindex(hess)
        if !all(isapprox.(hess[p][1][:, :], [0.0 0.0; 0.0 0.0], atol=1e-14))
            println(1)
            hesstest = false
        end
        if !all(isapprox.(hess[p][2][:, :], [0.0 0.0; 0.0 0.0], atol=1e-14))
            println(2)
            hesstest = false
        end
        if !all(isapprox.(hess[p][3][:, :], [0.0 1.0/16.0; 1.0/16.0 0.0], rtol=1e-14))
            println(3)
            hesstest = false
        end
    end
    @test hesstest
end

# Non-matching number of patches
@test_throws ArgumentError Geometry.MappedGeometry(
    Geometry.CartesianGeometry((
        (LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),
        (LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),
    )),
    (mapping_patch_1_slanted, mapping_patch_2_slanted, mapping_patch_2_slanted),
)

end

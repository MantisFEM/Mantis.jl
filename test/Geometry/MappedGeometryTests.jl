module MappedGeometryTests

using Mantis

# import ReadVTK
using Test
using StaticArrays

include("../TestHelpers.jl")

# # Refer to the following file for method and variable definitions.
# include("GeometryTestsHelpers.jl")

# # Test MappedCartesianGeometry ------------------------------------------------
# for nx in 1:3
#     for ny in 1:3
#         breakpoints = (
#             collect(LinRange(0.0, 1.0, nx + 1)), collect(LinRange(0.0, 2.0, ny + 1))
#         )
#         geom = Geometry.CartesianGeometry(breakpoints)

#         # Define the mapping ϕ of the geometry and its derivative.
#         # ϕ(x,y) = [(x + 0.2)*cos(y), (x + 0.2)*sin(y)\
#         function mapping(x::AbstractVector)
#             return [(x[1] + 0.2) * cos(x[2]), (x[1] + 0.2) * sin(x[2])]
#         end
#         function dmapping(x::AbstractVector)
#             return [cos(x[2]) -(x[1] + 0.2)*sin(x[2]); sin(x[2]) (x[1] + 0.2)*cos(x[2])]
#         end

#         dimension = (2, 2)
#         curved_mapping = Geometry.Mapping(dimension, mapping, dmapping)
#         mapped_geometry = Geometry.MappedGeometry(geom, curved_mapping)

#         # Generate the plot
#         file_name = "mapped_cartesian_test_nx_$(nx)_ny_$(ny).vtu"
#         output_file_path = Mantis.GeneralHelpers.export_path(
#             output_directory_tree, file_name
#         )
#         Plot.plot(
#             mapped_geometry;
#             vtk_filename=output_file_path[1:(end - 4)],
#             n_subcells=1,
#             degree=3,
#             ascii=false,
#             compress=false,
#         )

#         # Test geometry
#         # Read the cell data from the reference file
#         reference_points, reference_cells = get_point_cell_data(
#             reference_directory_tree, file_name
#         )
#         # Read the cell data from the output file
#         output_points, output_cells = get_point_cell_data(output_file_path)
#         # Check if cell data is identical
#         @test all(isapprox.(reference_points, output_points; rtol=rtol))
#         @test all(isequal.(reference_cells, output_cells))
#     end
# end
# # -----------------------------------------------------------------------------

const zeros_2x2 = zeros(SMatrix{2, 2})

@testset "1D" verbose=true begin
    @testset "1 patch, 1 element, identity map" verbose=true begin
        function mapping_I(x::AbstractVector{Float64})
            return x[1], x[2]
        end
        function dmapping_I(x::AbstractVector{Float64})
            return zeros_2x2
        end
        mapping_I_obj = Geometry.Mapping((1, 1), mapping_I, dmapping_I)
        geometry = Geometry.MappedGeometry(
            Geometry.CartesianGeometry((LinRange(0.0, 1.0, 2),)), mapping_I_obj
        )

        # Geometry.jl
        @test Geometry.get_manifold_dim(geometry) == 1
        @test Geometry.get_image_dim(geometry) == 1
        @test Geometry.get_num_patches(geometry) == 1
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 1
        @test Geometry.get_num_elements(geometry, 1) == 1
        @test Geometry.get_num_elements_per_patch(geometry) == (1,)
        @test Geometry.get_element_measure(geometry, 1) == 1.0
        @test Geometry.get_element_lengths(geometry, 1) == (1.0,)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 1.0),)
    end
end

# Mappings to create the deformed geometries.

# Two-patch slant
# The mappings are defined with reference to the unit square [0,1]x[0,1] as parametric
# domain.
const slant_factor = 0.25
function mapping_patch_1_slant(x::AbstractVector{Float64})
    return [x[1] + slant_factor * x[1] * x[2], x[2]]
end
function dmapping_patch_1_slant(x::AbstractVector{Float64})
    return SMatrix{2, 2, Float64, 4}(
        1.0 + slant_factor * x[2], 0.0, slant_factor * x[1], 1.0
    )
end
const Hp1 = SMatrix{2, 2, Float64, 4}(0.0, slant_factor, slant_factor, 0.0)
function ddmapping_patch_1_slant(x::AbstractVector{Float64})
    return Hp1, zeros_2x2
end
const mapping_patch_1_slanted = Geometry.Mapping(
    (2, 2), mapping_patch_1_slant, dmapping_patch_1_slant, ddmapping_patch_1_slant
)

function mapping_patch_2_slant(x::AbstractVector{Float64})
    return [x[1] + 1.0 + slant_factor * (1.0 - x[1]) * x[2], x[2]]
end
function dmapping_patch_2_slant(x::AbstractVector{Float64})
    return SMatrix{2, 2, Float64, 4}(
        1.0 - slant_factor * x[2], 0.0, slant_factor * (1.0 - x[1]), 1.0
    )
end
const Hp2 = SMatrix{2, 2, Float64, 4}(0.0, -slant_factor, -slant_factor, 0.0)
function ddmapping_patch_2_slant(x::AbstractVector{Float64})
    return Hp2, zeros_2x2
end
const mapping_patch_2_slanted = Geometry.Mapping(
    (2, 2), mapping_patch_2_slant, dmapping_patch_2_slant, ddmapping_patch_1_slant
)
const num_elements_per_dim_per_patch = ((4, 4), (5, 6))
const geom_cart_patch_1 = Geometry.CartesianGeometry((
    0.0:(1.0 / num_elements_per_dim_per_patch[1][1]):1.0,
    0.0:(1.0 / num_elements_per_dim_per_patch[1][2]):1.0,
))
const geom_cart_patch_2 = Geometry.CartesianGeometry((
    0.0:(1.0 / num_elements_per_dim_per_patch[2][1]):1.0,
    0.0:(1.0 / num_elements_per_dim_per_patch[2][2]):1.0,
))

# Left patch is rotated, right is trivial.
function mapping_patch_1_geo2mapped(x::AbstractVector{Float64})
    return SVector{2}(-x[2], x[1])
end
function dmapping_patch_1_geo2mapped(x::AbstractVector{Float64})
    return SMatrix{2, 2}(0.0, -1.0, 1.0, 0.0)
end
const mapping_obj_1_geo2mapped = Geometry.Mapping(
    (2, 2), mapping_patch_1_geo2mapped, dmapping_patch_1_geo2mapped
)
function mapping_patch_2_geo2mapped(x::AbstractVector{Float64})
    return SVector{2}(x[1], x[2])
end
function dmapping_patch_2_geo2mapped(x::AbstractVector{Float64})
    return SMatrix{2, 2}(1.0, 0.0, 0.0, 1.0)
end
const mapping_obj_2_geo2mapped = Geometry.Mapping(
    (2, 2), mapping_patch_2_geo2mapped, dmapping_patch_2_geo2mapped
)
const geo2mapped_topology = Topology.MeshTopology(
    ((1, 2, 3, 4), (1, 5, 6, 2)), Topology.QUAD
)

@testset "2D" verbose=true begin
    @testset "Tuples of mappings and patches" verbose = true begin
        geometry = Geometry.MappedGeometry(
            (geom_cart_patch_1, geom_cart_patch_2),
            (mapping_patch_1_slanted, mapping_patch_2_slanted),
            Topology.MeshTopology([(1, 2, 3, 4), (2, 5, 6, 3)], Topology.QUAD, Val(2)),
        )

        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 2
        @test Geometry.get_num_patches(geometry) == 2
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 46
        @test Geometry.get_num_elements(geometry, 1) == 16
        @test Geometry.get_num_elements_per_patch(geometry) == (16, 30)
        @test Geometry.get_element_measure(geometry, 1) == 0.0625
        @test Geometry.get_element_lengths(geometry, 1) == (0.25, 0.25)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 0.25), (0.0, 0.25))
    end

    @testset "Tuples of mappings with one geometry" verbose = true begin
        geometry = Geometry.MappedGeometry(
            Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5), LinRange(0.0, 1.0, 7)),),
            (mapping_patch_1_slanted, mapping_patch_2_slanted),
            Topology.MeshTopology(((1, 2, 3, 4), (2, 5, 6, 3)), Topology.QUAD),
        )

        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 2
        @test Geometry.get_num_patches(geometry) == 2
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 48
        @test Geometry.get_num_elements(geometry, 1) == 24
        @test Geometry.get_num_elements_per_patch(geometry) == (24, 24)
        @test Geometry.get_element_measure(geometry, 1) == 1.0 / 24.0
        @test Geometry.get_element_lengths(geometry, 1) == (0.25, 1.0 / 6.0)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 0.25), (0.0, 1.0 / 6.0))
    end

    @testset "Tuples of geometries with one mapping" verbose = true begin
        geometry = Geometry.MappedGeometry(
            (
                Geometry.CartesianGeometry((LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7))),
                Geometry.CartesianGeometry((LinRange(0.25, 0.5, 4), LinRange(0.0, 1.0, 7))),
                Geometry.CartesianGeometry((LinRange(0.5, 0.75, 5), LinRange(0.0, 1.0, 7))),
                Geometry.CartesianGeometry((LinRange(0.75, 1.0, 6), LinRange(0.0, 1.0, 7))),
            ),
            mapping_patch_1_slanted,
            Topology.MeshTopology(
                [(1, 2, 3, 4), (2, 5, 6, 3), (5, 7, 8, 6), (7, 9, 10, 8)],
                Topology.QUAD,
                Val(4),
            ),
        )

        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 2
        @test Geometry.get_num_patches(geometry) == 4
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 84
        @test Geometry.get_num_elements(geometry, 1) == 12
        @test Geometry.get_num_elements_per_patch(geometry) == (12, 18, 24, 30)
        @test Geometry.get_element_measure(geometry, 1) == 1.0 / 48.0
        @test Geometry.get_element_lengths(geometry, 1) == (0.125, 1.0 / 6.0)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 0.125), (0.0, 1.0 / 6.0))
    end

    @testset "One mapping with one geometry" verbose = true begin
        geometry = Geometry.MappedGeometry(geom_cart_patch_1, mapping_patch_1_slanted)

        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 2
        @test Geometry.get_num_patches(geometry) == 1
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 16
        @test Geometry.get_num_elements(geometry, 1) == 16
        @test Geometry.get_num_elements_per_patch(geometry) == (16,)
        @test Geometry.get_element_measure(geometry, 1) == 0.0625
        @test Geometry.get_element_lengths(geometry, 1) == (0.25, 0.25)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 0.25), (0.0, 0.25))
    end

    @testset "Rotated patch, multi-patch base" begin
        geometry = Mantis.Geometry.MappedGeometry(
            Mantis.Geometry.CartesianGeometry(
                (
                    (LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3)),
                    (LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 4)),
                ),
                geo2mapped_topology,
            ),
            (mapping_obj_1_geo2mapped, mapping_obj_2_geo2mapped),
            geo2mapped_topology,
        )

        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 2
        @test Geometry.get_num_patches(geometry) == 2
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 10
        @test Geometry.get_num_elements(geometry, 1) == 4
        @test Geometry.get_num_elements_per_patch(geometry) == (4, 6)
        @test Geometry.get_element_measure(geometry, 1) == 0.25
        @test Geometry.get_element_lengths(geometry, 1) == (0.5, 0.5)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 0.5), (0.0, 0.5))
    end

    @testset "Rotated patch, tuple of geometries as base" begin
        geometry = Mantis.Geometry.MappedGeometry(
            (
                Mantis.Geometry.CartesianGeometry((
                    LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 3)
                )),
                Mantis.Geometry.CartesianGeometry((
                    LinRange(0.0, 1.0, 3), LinRange(0.0, 1.0, 4)
                )),
            ),
            (mapping_obj_1_geo2mapped, mapping_obj_2_geo2mapped),
            Topology.MeshTopology([(1, 2, 3, 4), (1, 5, 6, 2)], Topology.QUAD, Val(2)),
        )

        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 2
        @test Geometry.get_num_patches(geometry) == 2
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 10
        @test Geometry.get_num_elements(geometry, 1) == 4
        @test Geometry.get_num_elements_per_patch(geometry) == (4, 6)
        @test Geometry.get_element_measure(geometry, 1) == 0.25
        @test Geometry.get_element_lengths(geometry, 1) == (0.5, 0.5)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 0.5), (0.0, 0.5))
    end

    @testset "Geometry helpers: curvilinear square" verbose=true begin
        # Curvilinear mapping
        geometry = Geometry.create_curvilinear_square((0.0, 0.0), (1.0, 1.0), (4, 4))

        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 2
        @test Geometry.get_num_patches(geometry) == 1
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 16
        @test Geometry.get_num_elements(geometry, 1) == 16
        @test Geometry.get_num_elements_per_patch(geometry) == (16,)
        @test Geometry.get_element_measure(geometry, 1) == 0.0625
        @test Geometry.get_element_lengths(geometry, 1) == (0.25, 0.25)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 0.25), (0.0, 0.25))
    end
end

const ddg = (
    zeros(SMatrix{2, 2, Float64, 4}),
    zeros(SMatrix{2, 2, Float64, 4}),
    SMatrix{2, 2, Float64, 4}(0.0, 1.0, 1.0, 0.0),
)
const ddgs = (
    zeros(SMatrix{2, 2, Float64, 4}),
    zeros(SMatrix{2, 2, Float64, 4}),
    SMatrix{2, 2, Float64, 4}(0.0, 1/16, 1/16, 0.0),
)
@testset "3D" verbose=true begin
    @testset "surface" verbose=true begin
        geo(x) = (x[1], x[2], x[1] * x[2])
        dgeo(x) = SMatrix{3, 2, Float64, 6}(1.0, 0.0, x[2], 0.0, 1.0, x[1])
        ddgeo(x) = ddg
        mapping = Mantis.Geometry.Mapping((2, 3), geo, dgeo, ddgeo)
        geometry = Mantis.Geometry.MappedGeometry(geom_cart_patch_1, mapping)

        # Geometry.jl
        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 3
        @test Geometry.get_num_patches(geometry) == 1
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 16
        @test Geometry.get_num_elements(geometry, 1) == 16
        @test Geometry.get_num_elements_per_patch(geometry) == (16,)
        @test Geometry.get_element_measure(geometry, 1) == 0.0625
        @test Geometry.get_element_lengths(geometry, 1) == (0.25, 0.25)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 0.25), (0.0, 0.25))

        xi = Points.TensorProductPoints((LinRange(0.0, 1.0, 2), LinRange(0.0, 1.0, 2)))

        expected_evaluate = Dict{Int, Matrix{Float64}}()
        expected_jacobian = Dict{Int, Vector{SMatrix{3, 2, Float64, 6}}}()
        expected_hessian = Dict{Int, Vector{NTuple{3, SMatrix{2, 2, Float64, 4}}}}()

        for (k, IJ) in enumerate(CartesianIndices((4, 4)))
            i, j = Tuple(IJ)
            xans = [
                x_i for _ in (1, 2) for
                x_i in LinRange((i - 1) * 1.0 / 16.0, i * 1.0 / 16.0, 2)
            ]
            yans = [
                y_i for y_i in LinRange((j - 1) * 1.0 / 16.0, j * 1.0 / 16.0, 2) for
                _ in (1, 2)
            ]

            temp = zeros(4, 3)
            for (e, (xei, yei)) in enumerate(zip(xans, yans))
                temp[e, :] = [xei*4, yei*4, xei*yei*4^2]
            end
            expected_evaluate[k] = temp
            expected_jacobian[k] = [
                SMatrix{3, 2, Float64, 6}(0.25, 0.0, yans[p], 0.0, 0.25, xans[p]) for
                p in eachindex(xans, yans)
            ]
            expected_hessian[k] = [ddgs for p in eachindex(xans, yans)]
        end
        @test @expected expected_evaluate k -> Geometry.evaluate(geometry, k, xi)
        @test @expected expected_jacobian k -> Geometry.jacobian(geometry, k, xi)
        @test @expected expected_hessian k -> Geometry.hessian(geometry, k, xi)
    end
end

@testset "Construction errors" begin
    # Non-matching number of patches
    @test_throws MethodError Geometry.MappedGeometry(
        Geometry.CartesianGeometry(
            (
                (LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),
                (LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),
            ),
            Topology.MeshTopology(((1, 2, 3, 4), (1, 5, 6, 2)), Topology.QUAD),
        ),
        (mapping_patch_1_slanted, mapping_patch_2_slanted, mapping_patch_2_slanted),
        Topology.MeshTopology(((1, 2, 3, 4), (1, 5, 6, 2)), Topology.QUAD),
    )
    @test_throws ArgumentError Geometry.MappedGeometry(
        Geometry.CartesianGeometry(
            (
                (LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),
                (LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),
            ),
            Topology.MeshTopology(((1, 2, 3, 4), (1, 5, 6, 2)), Topology.QUAD),
        ),
        (mapping_patch_1_slanted, mapping_patch_2_slanted, mapping_patch_2_slanted),
        Topology.MeshTopology(((1, 2, 3, 4), (1, 5, 6, 2), (5, 7, 8, 6)), Topology.QUAD),
    )
    # No topology
    @test_throws MethodError Geometry.MappedGeometry(
        Geometry.CartesianGeometry((
            (LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),
            (LinRange(0.0, 0.25, 3), LinRange(0.0, 1.0, 7)),
        )),
        (mapping_patch_1_slanted, mapping_patch_2_slanted),
    )
end

end

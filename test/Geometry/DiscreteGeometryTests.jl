module FEGeometryTests

using Mantis

import ReadVTK
import LinearAlgebra

using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")
include("../TestHelpers.jl")

function run_tests(geometry, file_name; degree=4)
    output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
    Plot.plot(geometry; vtk_filename=output_file_path, degree=degree)
    reference_points, reference_cells = get_point_cell_data(
        reference_directory_tree, file_name * ".vtu"
    )
    output_points, output_cells = get_point_cell_data(output_file_path * ".vtu")
    @test all(isapprox.(reference_points, output_points; atol=atol))
    @test all(isequal.(reference_cells, output_cells))

    return nothing
end

############################################################################################
#                                         Annulus                                          #
############################################################################################

starting_points = (0.0, 0.0)
box_sizes = (4.0, 1.0)
num_elements = (4, 1)
deg = (2, 1)
Wt = pi / 2
section_spaces = (
    FunctionSpaces.GeneralizedTrigonometric(deg[1], Wt), FunctionSpaces.Bernstein(deg[2])
)
regularities = (1, -1)

# create tensor-product space
Bθ, Br = FunctionSpaces.create_dim_wise_bspline_spaces(
    starting_points,
    box_sizes,
    num_elements,
    section_spaces,
    regularities;
    n_dofs_left=(1, 1),
    n_dofs_right=(1, 1),
)
Bθ_periodic = FunctionSpaces.GTBSplineSpace((Bθ,), [1])
TP = FunctionSpaces.TensorProductSpace((Bθ_periodic, Br))

# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0 .* r0
    geom_coeffs_0 .* r1
]

# create FEGeometry
geom = FunctionSpaces.FEGeometry(TP, geom_coeffs)
# Generate the plot
file_name = "fem_geometry_annulus_test"
run_tests(geom, file_name)

############################################################################################
#                          Lagrange-Bernstein (Square with hole)                           #
############################################################################################
deg = 1
nodes = Points.get_input_points(Quadrature.get_nodes(Quadrature.gauss_lobatto(deg+1)))[1]
b = FunctionSpaces.Lagrange(nodes)
B1 = FunctionSpaces.BSplineSpace(
    Geometry.CartesianGeometry(([0.0, 1.0, 2.0, 3.0, 4.0],)), b, [-1, 0, 0, 0, -1]
)
GB = FunctionSpaces.GTBSplineSpace((B1,), [0])
B2 = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry(([0.0, 1.0],)), 1, [-1, -1])
TP = FunctionSpaces.TensorProductSpace((GB, B2))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0 .* r0
    geom_coeffs_0 .* r1
]
geom = Mantis.FunctionSpaces.FEGeometry(TP, geom_coeffs)
file_name = "fem_geometry_lagrange_square_test"
run_tests(geom, file_name; degree=1)

@testset "Single-patch 1D-in-3D: Spiral" verbose=true begin
    geo = Geometry.CartesianGeometry(LinRange(0.0, 4.0, 5))
    GB = FunctionSpaces.BSplineSpace(
        geo, FunctionSpaces.GeneralizedTrigonometric(2, pi/2), [-1, 1, 1, 1, -1]
    )

    geom_coeffs = [
        +0.0 -1.0 0.0
        +1.0 -1.0 0.25
        +1.0 +1.0 0.5
        -1.0 +1.0 0.75
        -1.0 -1.0 1.0
        +0.0 -1.0 1.25
    ]
    geometry = FunctionSpaces.FEGeometry(GB, geom_coeffs)

    @test Geometry.get_manifold_dim(geometry) == 1
    @test Geometry.get_image_dim(geometry) == 3
    @test Geometry.get_num_patches(geometry) == 1
    @test Geometry.get_topology(geometry) == Topology.SINGLE_PATCH_TOPOLOGY_1D
    @test Geometry.get_patch_id(geometry, 1) == 1
    @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
    @test Geometry.get_global_element_id(geometry, 1, 1) == 1
    @test Geometry.get_num_elements(geometry) == 4
    @test Geometry.get_num_elements(geometry, 1) == 4
    @test Geometry.get_num_elements_per_patch(geometry) == (4,)
    @test Geometry.get_element_measure(geometry, 1) == 1.0
    @test Geometry.get_element_lengths(geometry, 1) == (1.0,)
    @test Geometry.get_element_vertices(geometry, 1) == ((0.0, 1.0),)

    xi = Points.TensorProductPoints((LinRange(0.0, 1.0, 4),))
    evaluations = Dict{Int, Matrix{Float64}}(
        1 => [
            0.0 -1.0 0.0
            0.5 -0.8660254037844387 0.14174682452694512
            0.8660254037844384 -0.5 0.2790063509461096
            1.0 0.0 0.375
        ],
        2 => [
            1.0 0.0 0.375
            0.8660254037844387 0.5 0.45424682452694515
            0.5 0.8660254037844384 0.5457531754730548
            0.0 1.0 0.625
        ],
        3 => [
            0.0 1.0 0.625
            -0.5 0.8660254037844387 0.7042468245269452
            -0.8660254037844384 0.5 0.7957531754730548
            -1.0 0.0 0.875
        ],
        4 => [
            -1.0 0.0 0.875
            -0.8660254037844386 -0.5 0.9709936490538903
            -0.5 -0.8660254037844384 1.1082531754730547
            0.0 -1.0 1.25
        ],
    )
    @test @expected evaluations k -> Geometry.evaluate(geometry, k, xi) (a, b) ->
        isapprox(a, b, atol=1e-14, rtol=1e-14)

    jacobians = Dict{Int, Vector{Matrix{Float64}}}(
        1 => [
            [1.5707963267948963; 0.0; 0.3926990816987241;;],
            [1.3603495231756633; 0.7853981633974483; 0.43826215121859685;;],
            [0.7853981633974481; 1.3603495231756633; 0.36639323124631995;;],
            [0.0; 1.5707963267948966; 0.1963495408493621;;],
        ],
        2 => [
            [0.0; 1.5707963267948961; 0.19634954084936207;;],
            [-0.7853981633974483; 1.360349523175663; 0.2682184608216389;;],
            [-1.360349523175663; 0.7853981633974481; 0.26821846082163897;;],
            [-1.5707963267948966; 0.0; 0.19634954084936213;;],
        ],
        3 => [
            [-1.5707963267948961; 0.0; 0.19634954084936207;;],
            [-1.360349523175663; -0.7853981633974483; 0.26821846082163897;;],
            [-0.7853981633974481; -1.360349523175663; 0.268218460821639;;],
            [0.0; -1.5707963267948966; 0.19634954084936218;;],
        ],
        4 => [
            [0.0; -1.5707963267948961; 0.19634954084936218;;],
            [0.7853981633974482; -1.360349523175663; 0.36639323124632006;;],
            [1.360349523175663; -0.7853981633974481; 0.4382621512185969;;],
            [1.5707963267948963; 0.0; 0.39269908169872436;;],
        ],
    )
    @test @expected jacobians k -> Geometry.jacobian(geometry, k, xi) (a, b) ->
        isapprox(a, b, atol=1e-14, rtol=1e-14)
end

############################################################################################
#                                       Wavy surface                                       #
############################################################################################
deg = 2
Wt = pi / 2
b = FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Geometry.CartesianGeometry(breakpoints)
B = FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])
GB = FunctionSpaces.GTBSplineSpace((B,), [1])
b1 = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0 .* r0 [-1.0, 1.0, -1.0, 1.0]
    geom_coeffs_0 .* r1 [1.0, -1.0, 1.0, -1.0]
]
geom = FunctionSpaces.FEGeometry(TP, geom_coeffs)
file_name = "fem_geometry_wavy_surface_test"
run_tests(geom, file_name)

############################################################################################
#                                  NURBS quarter annulus                                   #
############################################################################################

deg = 2
b = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
B = FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
b1 = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = FunctionSpaces.TensorProductSpace((B, b1))
# control points for geometry
geom_coeffs_0 = [
    0.0 1.0
    1.0 1.0
    1.0 0.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0 .* r0 zeros(3)
    geom_coeffs_0 .* r1 zeros(3)
]
geom = FunctionSpaces.FEGeometry(TP, geom_coeffs)
file_name = "fem_geometry_nurbs_quarter_annulus_test"
# run_tests(geom, file_name)

############################################################################################
#                                      NURBS annulus                                       #
############################################################################################

deg = 2
b = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
br = FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
B = (br, br, br, br)
GB = FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0 .* r0 zeros(4)
    geom_coeffs_0 .* r1 zeros(4)
]
geom = FunctionSpaces.FEGeometry(TP, geom_coeffs)
file_name = "fem_geometry_nurbs_annulus_test"
# run_tests(geom, file_name)

############################################################################################
#                                   NURBS wavy surface                                     #
############################################################################################

deg = 2
b = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
br = FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
B = (br, br, br, br)
GB = FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])
TP = FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 = [
    +1.0 -1.0
    +1.0 +1.0
    -1.0 +1.0
    -1.0 -1.0
]
r0 = 1
r1 = 2
geom_coeffs = [
    geom_coeffs_0 .* r0 [-1.0, 1.0, -1.0, 1.0]
    geom_coeffs_0 .* r1 [1.0, -1.0, 1.0, -1.0]
]
geom = FunctionSpaces.FEGeometry(TP, geom_coeffs)
file_name = "fem_geometry_nurbs_wavy_surface_test"
# run_tests(geom, file_name)

# ############################################################################################
# #                                NURBS vs GTB basis annulus                               #
# ############################################################################################

# # B-spline and NURBS GTB spline spaces on the circle
# deg = 2
# b = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
# br = FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
# Bsp = FunctionSpaces.GTBSplineSpace((b, b, b, b), [1, 1, 1, 1])
# Nurbs = FunctionSpaces.GTBSplineSpace((br, br, br, br), [1, 1, 1, 1])

# # GTB spline space on the radial direction
# Wt = pi / 2
# gt = FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
# B = FunctionSpaces.BSplineSpace(
#     Geometry.CartesianGeometry([0.0, 1.0, 2.0, 3.0, 4.0]), gt, [-1, 1, 1, 1, -1]
# )
# GTB = FunctionSpaces.GTBSplineSpace((B,), [1])

# b1 = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])

# TP_bsp = FunctionSpaces.TensorProductSpace((Bsp, b1))
# TP_nurbs = FunctionSpaces.TensorProductSpace((Nurbs, b1))
# TP_gtb = FunctionSpaces.TensorProductSpace((GTB, b1))

# # control points for geometry
# geom_coeffs_0 = [
#     +1.0 -1.0
#     +1.0 +1.0
#     -1.0 +1.0
#     -1.0 -1.0
# ]
# r0 = 1
# r1 = 2
# geom_coeffs = [
#     geom_coeffs_0 .* r0 zeros(4)
#     geom_coeffs_0 .* r1 zeros(4)
# ]

# # NURBS annulus with B-spline and NURBS bases
# geom = FunctionSpaces.FEGeometry(TP_nurbs, geom_coeffs)
# file_name = "fem_geometry_nurbs_bsp_basis_test"
# # run_tests(geom, file_name)

# geom = FunctionSpaces.FEGeometry(TP_gtb, geom_coeffs)
# file_name = "fem_geometry_nurbs_gtb_basis_test"
# # run_tests(geom, file_name)

end

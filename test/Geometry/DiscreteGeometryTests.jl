module DiscreteGeometryTests

using Mantis

import ReadVTK
import LinearAlgebra

using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")

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
    starting_points, box_sizes, num_elements, section_spaces, regularities, (1, 1), (1, 1)
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

# create DiscreteGeometry
geom = FunctionSpaces.DiscreteGeometry(TP, geom_coeffs)
# Generate the plot
file_name = "fem_geometry_annulus_test"
run_tests(geom, file_name)

############################################################################################
#                          Lagrange-Bernstein (Square with hole)                           #
############################################################################################
deg = 1
b = FunctionSpaces.LobattoLegendre(deg)
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
geom = Mantis.FunctionSpaces.DiscreteGeometry(TP, geom_coeffs)
file_name = "fem_geometry_lagrange_square_test"
run_tests(geom, file_name; degree=1)

############################################################################################
#                                          Spiral                                          #
############################################################################################
deg = 2
Wt = pi / 2
b = FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Geometry.CartesianGeometry(breakpoints)
GB = FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])

# control points for geometry
geom_coeffs = [
    +0.0 -1.0 0.0
    +1.0 -1.0 0.25
    +1.0 +1.0 0.5
    -1.0 +1.0 0.75
    -1.0 -1.0 1.0
    +0.0 -1.0 1.25
]
geom = FunctionSpaces.DiscreteGeometry(GB, geom_coeffs)
file_name = "fem_geometry_spiral_test"
run_tests(geom, file_name)

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
    geom_coeffs_0.*r0 [-1.0, 1.0, -1.0, 1.0]
    geom_coeffs_0.*r1 [1.0, -1.0, 1.0, -1.0]
]
geom = FunctionSpaces.DiscreteGeometry(TP, geom_coeffs)
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
    geom_coeffs_0.*r0 zeros(3)
    geom_coeffs_0.*r1 zeros(3)
]
geom = FunctionSpaces.DiscreteGeometry(TP, geom_coeffs)
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
    geom_coeffs_0.*r0 zeros(4)
    geom_coeffs_0.*r1 zeros(4)
]
geom = FunctionSpaces.DiscreteGeometry(TP, geom_coeffs)
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
    geom_coeffs_0.*r0 [-1.0, 1.0, -1.0, 1.0]
    geom_coeffs_0.*r1 [1.0, -1.0, 1.0, -1.0]
]
geom = FunctionSpaces.DiscreteGeometry(TP, geom_coeffs)
file_name = "fem_geometry_nurbs_wavy_surface_test"
# run_tests(geom, file_name)

############################################################################################
#                                NURBS vs GTB basis annulus                               #
############################################################################################

# B-spline and NURBS GTB spline spaces on the circle
deg = 2
b = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), deg, [-1, -1])
br = FunctionSpaces.RationalFESpace(b, [1, 1 / sqrt(2), 1])
Bsp = FunctionSpaces.GTBSplineSpace((b, b, b, b), [1, 1, 1, 1])
Nurbs = FunctionSpaces.GTBSplineSpace((br, br, br, br), [1, 1, 1, 1])

# GTB spline space on the radial direction
Wt = pi / 2
gt = FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
B = FunctionSpaces.BSplineSpace(
    Geometry.CartesianGeometry([0.0, 1.0, 2.0, 3.0, 4.0]), gt, [-1, 1, 1, 1, -1]
)
GTB = FunctionSpaces.GTBSplineSpace((B,), [1])

b1 = FunctionSpaces.BSplineSpace(Geometry.CartesianGeometry([0.0, 1.0]), 1, [-1, -1])

TP_bsp = FunctionSpaces.TensorProductSpace((Bsp, b1))
TP_nurbs = FunctionSpaces.TensorProductSpace((Nurbs, b1))
TP_gtb = FunctionSpaces.TensorProductSpace((GTB, b1))

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
    geom_coeffs_0.*r0 zeros(4)
    geom_coeffs_0.*r1 zeros(4)
]

# NURBS annulus with B-spline and NURBS bases
geom = FunctionSpaces.DiscreteGeometry(TP_nurbs, geom_coeffs)
file_name = "fem_geometry_nurbs_bsp_basis_test"
# run_tests(geom, file_name)

geom = FunctionSpaces.DiscreteGeometry(TP_gtb, geom_coeffs)
file_name = "fem_geometry_nurbs_gtb_basis_test"
# run_tests(geom, file_name)

end
